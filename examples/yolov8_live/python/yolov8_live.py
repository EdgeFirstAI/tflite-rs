#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

"""Live YOLOv8 camera inference with edgefirst-tflite and Wayland display.

Captures NV12 or YUYV frames from a camera through the libcamera Python
bindings, runs YOLOv8 detection or instance segmentation with
edgefirst-tflite, and shows the result in a Wayland window. Overlays are
composited onto the camera frame by the EdgeFirst HAL (edgefirst-image).

Pipeline::

    libcamera (NV12 | YUYV DMA-BUF)
      -> HAL import            (cached per camera buffer)
      -> HAL convert           (letterbox into the delegate's input DMA-BUF)
      -> TFLite invoke         (Neutron / VxDelegate NPU, or CPU)
      -> Decoder.draw_onto     (decode + composite onto an RGBA DMA-BUF canvas)
      -> pywayland             (canvas DMA-BUF -> wl_buffer -> compositor)

Display uses the ``zwp_linux_dmabuf_v1`` protocol to hand the canvas to the
compositor directly -- no EGL, OpenGL or GStreamer on the client side.

Models:
    Official pre-trained models are published in the EdgeFirst model zoo on
    Hugging Face:

      Detection:    https://huggingface.co/EdgeFirst/yolov8-det
      Segmentation: https://huggingface.co/EdgeFirst/yolov8-seg

      # i.MX95 Neutron
      curl -LO https://huggingface.co/EdgeFirst/yolov8-seg/resolve/main/imx95/yolov8n-seg-int8-smart.imx95.tflite

      # i.MX8MP VxDelegate, or CPU
      curl -LO https://huggingface.co/EdgeFirst/yolov8-det/resolve/main/tflite/yolov8n-det-int8-smart.tflite

    The decoder is configured from the ``edgefirst.json`` schema embedded in
    each export, as in the still-image ``yolov8.py`` example.

Usage::

    python yolov8_live.py <model.tflite> [--delegate PATH] [--threshold N]
        [--iou N] [--width N] [--height N] [--camera-name ID]
        [--format nv12|yuyv] [--color-mode class|instance|track]

    # i.MX95 Neutron, from an SSH session on the board
    XDG_RUNTIME_DIR=/run/user/0 WAYLAND_DISPLAY=wayland-0 \\
    LIBCAMERA_PIPELINES_MATCH_LIST=nxp/neo \\
        python yolov8_live.py yolov8n-seg-int8-smart.imx95.tflite \\
        --delegate /usr/lib/libneutron_delegate.so

    On i.MX95 the ``imx8-isi`` pipeline handler claims the ISI before the
    NeoISP handler and only offers raw Bayer, so restrict libcamera to
    ``nxp/neo`` to get NV12/YUYV from the ISP.

Requirements:
    pip install "edgefirst-tflite>=0.10.3" numpy pywayland \\
        "edgefirst-decoder>=0.38.1" "edgefirst-image>=0.38.1" "edgefirst-tensor>=0.38.1"

    The libcamera Python bindings ship with libcamera; on the board create
    the virtual environment with ``--system-site-packages`` to see them.
    pywayland builds from source and needs the libwayland-client headers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import select
import signal
import sys
import time

import edgefirst.decoder as ef_decoder
import edgefirst.image as ef_image
import edgefirst.tensor as ef_tensor
import numpy as np
from edgefirst_tflite import Interpreter, ModelArchive, load_delegate

try:
    import libcamera
except ImportError as exc:
    sys.exit(
        f"libcamera Python bindings not available: {exc}\n"
        "They ship with libcamera; create the venv with --system-site-packages."
    )

try:
    # Every protocol type is imported here, including those only named
    # through the registry, so a partial install fails at startup rather than
    # deep inside the camera loop.
    from pywayland.client import Display as WlDisplay
    from pywayland.protocol.linux_dmabuf_unstable_v1 import (  # noqa: F401
        ZwpLinuxBufferParamsV1,
        ZwpLinuxDmabufV1,
    )
    from pywayland.protocol.wayland import (  # noqa: F401
        WlBuffer,
        WlCallback,
        WlCompositor,
        WlSurface,
    )
    from pywayland.protocol.xdg_shell import (  # noqa: F401
        XdgSurface,
        XdgToplevel,
        XdgWmBase,
    )
except ImportError as exc:
    sys.exit(f"pywayland not available: {exc}\nInstall: pip install pywayland")


# ── Wayland display (direct DMA-BUF submission) ──────────────────────────────

# DRM fourcc ABGR8888 ('AB24'): R, G, B, A bytes in memory order, the layout of
# the HAL's PixelFormat.Rgba.
DRM_FORMAT_ABGR8888 = 0x34324241


class WaylandDisplay:
    """An xdg-shell toplevel window that presents DMA-BUF RGBA buffers.

    Every frame is committed as soon as it is drawn. The compositor shows the
    newest one at each refresh and releases those it replaced, so the display
    never waits on the pipeline or the pipeline on the display.
    """

    def __init__(self, width: int, height: int, stride: int, title: str):
        self.width = width
        self.height = height
        self.stride = stride
        self._closed = False
        self._configured = False
        self._buffers: dict[int, WlBuffer] = {}
        # The fds of buffers the compositor holds: set on submit and cleared
        # by wl_buffer.release. Drawing into one of these tears the picture.
        self._busy: set[int] = set()
        self._compositor = None
        self._wm_base = None
        self._dmabuf = None

        self._display = WlDisplay()
        self._display.connect()
        registry = self._display.get_registry()
        registry.dispatcher["global"] = self._on_global
        self._display.roundtrip()
        if not (self._compositor and self._wm_base and self._dmabuf):
            raise RuntimeError(
                "compositor lacks wl_compositor, xdg_wm_base or "
                "zwp_linux_dmabuf_v1"
            )

        self._surface = self._compositor.create_surface()
        self._xdg_surface = self._wm_base.get_xdg_surface(self._surface)
        self._xdg_surface.dispatcher["configure"] = self._on_configure
        self._toplevel = self._xdg_surface.get_toplevel()
        self._toplevel.set_title(title)
        self._toplevel.set_app_id("edgefirst-yolov8-live")
        self._toplevel.dispatcher["close"] = self._on_close
        self._surface.commit()
        while not self._configured:
            self._display.dispatch(block=True)

    def _on_global(self, registry, name, interface, version):
        if interface == "wl_compositor":
            self._compositor = registry.bind(name, WlCompositor, min(version, 4))
        elif interface == "xdg_wm_base":
            self._wm_base = registry.bind(name, XdgWmBase, min(version, 1))
            self._wm_base.dispatcher["ping"] = lambda wm, serial: wm.pong(serial)
        elif interface == "zwp_linux_dmabuf_v1":
            self._dmabuf = registry.bind(name, ZwpLinuxDmabufV1, min(version, 3))

    def _on_configure(self, xdg_surface, serial):
        xdg_surface.ack_configure(serial)
        self._configured = True

    def _on_close(self, *_args):
        self._closed = True

    def _buffer_for(self, fd: int) -> WlBuffer:
        buf = self._buffers.get(fd)
        if buf is None:
            params = self._dmabuf.create_params()
            params.add(fd, 0, 0, self.stride, 0, 0)
            buf = params.create_immed(
                self.width, self.height, DRM_FORMAT_ABGR8888, 0
            )
            params.destroy()
            buf.dispatcher["release"] = lambda _buf: self._busy.discard(fd)
            self._buffers[fd] = buf
        return buf

    def _dispatch_pending(self):
        # dispatch(block=False) only drains events already queued; buffer
        # releases and close events arrive on the socket, which a blocking
        # dispatch reads, so it is called only once the socket is readable.
        while select.select([self._display.get_fd()], [], [], 0)[0]:
            self._display.dispatch(block=True)
        self._display.dispatch(block=False)

    def free_buffer(self, fds) -> int | None:
        """The index of a buffer in *fds* that is free to draw into, or
        ``None`` when the compositor still reads every buffer or the window
        has been closed."""
        self._dispatch_pending()
        if self._closed:
            return None
        return next((i for i, fd in enumerate(fds) if fd not in self._busy), None)

    def present(self, fd: int) -> None:
        """Show the RGBA DMA-BUF *fd*, which ``free_buffer`` returned."""
        self._surface.attach(self._buffer_for(fd), 0, 0)
        self._surface.damage_buffer(0, 0, self.width, self.height)
        self._busy.add(fd)
        self._surface.commit()
        self._display.flush()

    def is_open(self) -> bool:
        self._dispatch_pending()
        self._display.flush()
        return not self._closed

    def destroy(self):
        if self._display:
            self._display.disconnect()
            self._display = None


# ── Model setup ──────────────────────────────────────────────────────────────


def resolve_schema(model_path, input_details, output_details):
    """Return the decoder schema: the embedded ``edgefirst.json``, otherwise
    one inferred from a stock Ultralytics export's own signals. See
    ``resolve_schema`` in the still-image ``yolov8.py`` for the rationale."""
    try:
        archive = ModelArchive(path=model_path)
    except Exception:
        archive = None

    metadata = {}
    if archive is not None:
        try:
            schema = json.loads(archive.edgefirst_json())
        except Exception:
            schema = None
        if schema is not None:
            print("  Schema:  edgefirst.json embedded")
            return schema
        for name in archive.entry_names():
            try:
                metadata[name] = archive.read_to_string(name)
            except Exception:
                pass

    def signals(details):
        described = []
        for det in details:
            scale, zero_point = det["quantization"]
            quant = ([float(scale)], [int(zero_point)]) if scale != 0.0 else None
            described.append(
                (
                    det["name"],
                    [int(s) for s in det["shape"]],
                    str(det["dtype"]),
                    quant,
                )
            )
        return described

    try:
        inferred = ef_decoder.infer_ultralytics_schema(
            "tflite", signals(input_details), signals(output_details), metadata
        )
    except ValueError as exc:
        sys.exit(
            "model carries no embedded edgefirst.json, and its schema could "
            f"not be inferred: {exc}"
        )
    print(f"  Schema:  inferred ({inferred.description})")
    return inferred.schema


def compute_letterbox(src_w, src_h, dst_w, dst_h):
    """Normalised ``(x0, y0, x1, y1)`` of the letterboxed image inside the
    model input.

    ``convert(letterbox=...)`` places the image itself; ``draw_onto`` needs
    the same placement to map boxes and masks back onto the camera frame.
    Mirrors the HAL's ``letterbox_rect`` exactly, including its rounding:
    ``floor(x + 0.5)`` matches Rust's half-away-from-zero ``round``, where
    Python's ``round`` would round half to even.
    """
    src_aspect = src_w / src_h
    if src_aspect > dst_w / dst_h:
        new_w, new_h = dst_w, max(1, math.floor(dst_w / src_aspect + 0.5))
    else:
        new_w, new_h = max(1, math.floor(dst_h * src_aspect + 0.5)), dst_h
    x = (dst_w - new_w) // 2
    y = (dst_h - new_h) // 2
    return (x / dst_w, y / dst_h, (x + new_w) / dst_w, (y + new_h) / dst_h)


NP_DTYPES = {"int8": np.int8, "uint8": np.uint8, "float32": np.float32, "int32": np.int32}


class OutputBuffers:
    """HAL tensors mirroring the TFLite outputs, refreshed after each invoke.

    The per-scale decoder reads quantization from the tensors themselves,
    so each integer output carries its TFLite (scale, zero_point).
    """

    def __init__(self, output_details):
        self.tensors = []
        self._dtypes = []
        for det in output_details:
            dtype = str(det["dtype"])
            if dtype not in NP_DTYPES:
                sys.exit(f"unsupported output dtype {dtype}")
            tensor = ef_tensor.Tensor([int(s) for s in det["shape"]], dtype=dtype)
            scale, zero_point = det["quantization"]
            if scale != 0.0:
                tensor.set_quantization_per_tensor(scale, zero_point)
            self.tensors.append(tensor)
            self._dtypes.append(NP_DTYPES[dtype])

    def sync_from(self, interpreter):
        for i, (tensor, dtype) in enumerate(zip(self.tensors, self._dtypes)):
            src = interpreter.get_output_tensor(i)
            with tensor.map("write") as view:
                np.copyto(np.frombuffer(view, dtype=dtype).reshape(src.shape), src)


# ── Camera frames ────────────────────────────────────────────────────────────


def hal_colorimetry(color_space):
    """The HAL colorimetry for a libcamera stream's color space.

    Passing it to the import matters: an untagged YUV frame is converted with
    a guess made from its height (BT.709, limited range for HD), while ISPs
    commonly deliver sYCC (BT.601, full range). The mismatch stretches the
    contrast and shifts the hue of every frame the model sees.
    """
    if color_space is None:
        return None
    space = {
        "Smpte170m": ef_tensor.ColorSpace.Smpte170m,
        "Rec709": ef_tensor.ColorSpace.Bt709,
        "Rec2020": ef_tensor.ColorSpace.Bt2020,
    }.get(color_space.primaries.name)
    transfer = {
        "Linear": ef_tensor.ColorTransfer.Linear,
        "Srgb": ef_tensor.ColorTransfer.Srgb,
        "Rec709": ef_tensor.ColorTransfer.Bt709,
    }.get(color_space.transferFunction.name)
    encoding = {
        "Rec601": ef_tensor.ColorEncoding.Bt601,
        "Rec709": ef_tensor.ColorEncoding.Bt709,
        "Rec2020": ef_tensor.ColorEncoding.Bt2020,
    }.get(color_space.ycbcrEncoding.name)
    range_ = {
        "Full": ef_tensor.ColorRange.Full,
        "Limited": ef_tensor.ColorRange.Limited,
    }.get(color_space.range.name)
    return ef_tensor.Colorimetry(space, transfer, encoding, range_)



class FrameCache:
    """HAL image tensors for libcamera's buffers, imported once per buffer.

    ``FrameBufferAllocator`` hands out a fixed pool whose DMA-BUFs live as
    long as the camera, so each is imported on first use and indexed by the
    request cookie. The import dups the fds, which keeps the buffers alive
    but not their pixels: a request goes back to the camera only once its
    frame has been read.
    """

    def __init__(self, capacity, width, height, stride, fmt, colorimetry):
        self._entries = [None] * capacity
        self.width = width
        self.height = height
        self.stride = stride
        self.fmt = fmt
        self.colorimetry = colorimetry

    def get_or_import(self, index, processor, framebuffer):
        tensor = self._entries[index]
        if tensor is None:
            planes = framebuffer.planes
            chroma = {}
            # NV12 may arrive as a second plane: a separate DMA-BUF, or the
            # same one at an offset past the luma rows.
            if len(planes) >= 2 and self.fmt == ef_image.PixelFormat.Nv12:
                chroma = {
                    "chroma_fd": planes[1].fd,
                    "chroma_stride": self.stride,
                    "chroma_offset": planes[1].offset or None,
                }
            tensor = processor.import_image(
                planes[0].fd,
                self.width,
                self.height,
                self.fmt,
                stride=self.stride,
                offset=planes[0].offset or None,
                colorimetry=self.colorimetry,
                **chroma,
            )
            self._entries[index] = tensor
        return tensor


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live YOLOv8 camera inference with edgefirst-tflite",
    )
    ap.add_argument("model", help="Path to .tflite model")
    ap.add_argument("--delegate", help="Path to delegate .so")
    # Higher than yolov8.py's 0.25: live video wants fewer false positives per
    # frame to keep the overlay stable.
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--width", type=int, default=1920, help="Camera width")
    ap.add_argument("--height", type=int, default=1080, help="Camera height")
    ap.add_argument("--camera-name", help="libcamera camera ID")
    ap.add_argument("--format", default="nv12", choices=["nv12", "yuyv"])
    ap.add_argument(
        "--color-mode", default="class", choices=["class", "instance", "track"]
    )
    args = ap.parse_args()

    cam_w, cam_h = args.width, args.height
    libcam_fmt, hal_fmt = {
        "nv12": (libcamera.formats.NV12, ef_image.PixelFormat.Nv12),
        "yuyv": (libcamera.formats.YUYV, ef_image.PixelFormat.Yuyv),
    }[args.format]
    color_mode = {
        "class": ef_image.ColorMode.Class,
        "instance": ef_image.ColorMode.Instance,
        "track": ef_image.ColorMode.Track,
    }[args.color_mode]

    # ── 1. Model, delegate, interpreter ──────────────────────────────────
    delegates = []
    if args.delegate:
        delegates.append(load_delegate(args.delegate))
        print(f"Delegate: {args.delegate}")
    interpreter = Interpreter(
        model_path=args.model,
        experimental_delegates=delegates or None,
        num_threads=4,
    )
    print(f"Model: {args.model}")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_shape = [int(s) for s in input_details[0]["shape"]]
    in_dtype = str(input_details[0]["dtype"])
    print(f"  input[0]: shape={in_shape} dtype={in_dtype}")
    is_nchw = in_shape[1] in (1, 3) and in_shape[3] not in (1, 3)
    if is_nchw or in_dtype not in ("int8", "uint8"):
        sys.exit("unsupported input; use an EdgeFirst model zoo export (NHWC int8)")
    in_h, in_w = in_shape[1], in_shape[2]

    # ── 2. Decoder ───────────────────────────────────────────────────────
    schema = resolve_schema(args.model, input_details, output_details)
    is_segmentation = any(o.get("type") == "protos" for o in schema["outputs"])
    decoder = ef_decoder.Decoder(
        schema, score_threshold=args.threshold, iou_threshold=args.iou
    )
    print(f"  Mode:    {'segmentation' if is_segmentation else 'detection'}")

    # ── 3. HAL processor, model I/O and display canvas ───────────────────
    # Preprocessing and overlay drawing use separate processors. Sharing one
    # corrupts the next frame's model input on i.MX95 (Mali): state left by
    # draw_onto garbles the following convert into the packed-RGB input
    # buffer, letterbox padding included, and detection quietly degrades.
    processor = ef_image.ImageProcessor()
    overlay_processor = ef_image.ImageProcessor()

    dmabuf = interpreter.dmabuf()
    info = None
    if dmabuf is not None:
        try:
            info = dmabuf.tensor_info(0)
        except Exception as exc:
            print(f"  Input:   delegate DMA-BUF unavailable ({exc})")
    else:
        print("  Input:   delegate DMA-BUF unavailable (no delegate DMA-BUF API)")
    if info is not None:
        model_input = processor.import_image(
            info["fd"],
            in_w,
            in_h,
            ef_image.PixelFormat.Rgb,
            in_dtype,
            offset=info.get("offset"),
        )
        staging = None
        print("  Input:   delegate DMA-BUF (GPU → NPU, zero-copy)")
    else:
        model_input = processor.create_image(
            in_w, in_h, ef_image.PixelFormat.Rgb, in_dtype, access="read"
        )
        staging = np.zeros((1, in_h, in_w, 3), dtype=NP_DTYPES[in_dtype])
        print("  Input:   staging copy (GPU → staging → TFLite arena)")

    outputs = OutputBuffers(output_details)
    pad = (114, 114, 114, 255)
    letterbox = compute_letterbox(cam_w, cam_h, in_w, in_h)

    # Each frame is drawn into a canvas the compositor has released, so it
    # never displays a frame mid-draw. It can hold three: the one on screen,
    # one waiting for the next refresh and the newest commit.
    canvases = [
        processor.create_image(cam_w, cam_h, ef_image.PixelFormat.Rgba)
        for _ in range(4)
    ]
    for canvas in canvases:
        if canvas.memory != ef_tensor.TensorMemory.DMABUF:
            sys.exit(f"canvas is {canvas.memory}, but Wayland needs a DMA-BUF")
    canvas_fds = [canvas.fd for canvas in canvases]
    canvas_stride = canvases[0].row_stride or cam_w * 4

    # ── 4. Camera ────────────────────────────────────────────────────────
    manager = libcamera.CameraManager.singleton()
    cameras = manager.cameras
    if args.camera_name:
        camera = next((c for c in cameras if c.id == args.camera_name), None)
    else:
        camera = cameras[0] if cameras else None
    if camera is None:
        sys.exit(f"camera not found; available: {[c.id for c in cameras]}")
    print(f"Camera: {camera.id}")
    camera.acquire()

    config = camera.generate_configuration([libcamera.StreamRole.VideoRecording])
    stream_cfg = config.at(0)
    stream_cfg.pixel_format = libcam_fmt
    stream_cfg.size = libcamera.Size(cam_w, cam_h)
    status = config.validate()
    # libcamera may adjust the buffer count, stride or color space, which the
    # import follows. The canvas, window and letterbox are all sized from the
    # request, so a different size or format would be drawn misregistered.
    if (
        status == libcamera.CameraConfiguration.Status.Invalid
        or (stream_cfg.size.width, stream_cfg.size.height) != (cam_w, cam_h)
        or stream_cfg.pixel_format != libcam_fmt
    ):
        sys.exit(
            f"camera cannot capture {cam_w}x{cam_h} {args.format}; it offered "
            f"{stream_cfg.size} {stream_cfg.pixel_format}. On i.MX95 set "
            "LIBCAMERA_PIPELINES_MATCH_LIST=nxp/neo to use the ISP."
        )
    camera.configure(config)

    stream = stream_cfg.stream
    allocator = libcamera.FrameBufferAllocator(camera)
    allocator.allocate(stream)
    buffers = allocator.buffers(stream)
    print(
        f"  Capture: {cam_w}x{cam_h} {args.format}, stride {stream_cfg.stride}, "
        f"{stream_cfg.color_space}, "
        f"{len(buffers)} buffers"
    )
    requests = []
    for i, fb in enumerate(buffers):
        request = camera.create_request(i)
        request.add_buffer(stream, fb)
        requests.append(request)
    frames = FrameCache(
        len(buffers),
        cam_w,
        cam_h,
        stream_cfg.stride,
        hal_fmt,
        hal_colorimetry(stream_cfg.color_space),
    )

    # ── 5. Display ───────────────────────────────────────────────────────
    display = WaylandDisplay(
        cam_w, cam_h, canvas_stride, "EdgeFirst TFLite YOLOv8 Live"
    )

    running = True

    def _on_sigint(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _on_sigint)

    # ── 6. Live loop ─────────────────────────────────────────────────────
    # GL composites the frame's native NV12/YUYV layout as the overlay
    # background. G2D and the CPU copy it byte-for-byte and refuse a format
    # that differs from the RGBA canvas; the first refused draw switches to
    # converting each frame to RGBA first.
    rgba_background = None
    frame_count = 0
    t_start = time.monotonic()
    try:
        camera.start()
        for request in requests:
            camera.queue_request(request)
        completed: queue.SimpleQueue = queue.SimpleQueue()
        print("Running -- close the window or press Ctrl+C to stop\n")

        report_every = 30
        window = [0.0] * 6
        dropped = 0
        detections = 0
        skipped = 0

        while running and display.is_open():
            t0 = time.monotonic()

            # Take the newest completed frame, returning older ones unprocessed.
            while completed.empty():
                if not select.select([manager.event_fd], [], [], 5.0)[0]:
                    sys.exit("no frame within 5 s")
                for request in manager.get_ready_requests():
                    completed.put(request)
            request = completed.get()
            while not completed.empty():
                request.reuse()
                camera.queue_request(request)
                request = completed.get()
                dropped += 1
            t1 = time.monotonic()

            frame = frames.get_or_import(request.cookie, processor, request.buffers[stream])
            t2 = time.monotonic()

            # The camera would overwrite the frame if its buffer were queued
            # again before the convert and the draw have read it.
            try:
                processor.convert(frame, model_input, letterbox=pad)
                if staging is None:
                    dmabuf.sync_for_device(0)
                else:
                    model_input.normalize_to_numpy(staging[0])
                    interpreter.set_tensor(0, staging)
                t3 = time.monotonic()

                # A signal arriving inside invoke() interrupts the delegate's
                # blocking wait and fails the inference, so SIGINT is held off
                # until it returns.
                signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
                try:
                    interpreter.invoke()
                finally:
                    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
                outputs.sync_from(interpreter)
                t4 = time.monotonic()

                back = display.free_buffer(canvas_fds)
                if back is not None:
                    background = frame
                    if rgba_background is not None:
                        overlay_processor.convert(frame, rgba_background)
                        background = rgba_background
                    try:
                        _, scores, _ = decoder.draw_onto(
                            overlay_processor,
                            outputs.tensors,
                            canvases[back],
                            background=background,
                            content_region=letterbox,
                            color_mode=color_mode,
                        )
                    except RuntimeError:
                        if rgba_background is not None:
                            raise
                        rgba_background = processor.create_image(
                            cam_w, cam_h, ef_image.PixelFormat.Rgba
                        )
                        continue
                    detections = len(scores)
                else:
                    skipped += 1
                t5 = time.monotonic()

                if back is not None:
                    display.present(canvas_fds[back])
                t6 = time.monotonic()
            finally:
                request.reuse()
                camera.queue_request(request)

            for i, d in enumerate((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)):
                window[i] += d
            frame_count += 1
            if frame_count % report_every == 0:
                fps = frame_count / (time.monotonic() - t_start)
                pull, imp, pre, inf, draw, disp = (
                    w / report_every * 1000.0 for w in window
                )
                print(
                    f"\r  FPS: {fps:5.1f}  pull:{pull:5.1f} imp:{imp:4.1f} "
                    f"pre:{pre:4.1f} inf:{inf:5.1f} draw:{draw:5.1f} "
                    f"disp:{disp:4.1f} ms  drop:{dropped} skip:{skipped} det:{detections:<3}",
                    end="",
                    flush=True,
                )
                window = [0.0] * 6
    finally:
        # ── 7. Shutdown ──────────────────────────────────────────────────
        # Explicit, ordered teardown: left to the garbage collector, the
        # pywayland proxies can outlive their display and crash on destroy.
        print()
        camera.stop()
        camera.release()
        for fd in canvas_fds:
            os.close(fd)
        display.destroy()
    if frame_count > 0:
        elapsed = time.monotonic() - t_start
        print(
            f"Processed {frame_count} frames in {elapsed:.1f}s "
            f"({frame_count / elapsed:.1f} FPS average)"
        )


if __name__ == "__main__":
    main()
