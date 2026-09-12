#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

"""YOLOv8 Object Detection & Segmentation with edgefirst-tflite.

End-to-end YOLOv8 inference using edgefirst-tflite for model execution and the
modular EdgeFirst HAL packages for the rest of the pipeline: edgefirst-codec
(JPEG/PNG decode), edgefirst-tensor (zero-copy tensor allocation),
edgefirst-image (preprocessing and overlay rendering), and edgefirst-decoder
(YOLO decoding via the high-level Decoder API).

The HAL packages share the PEP 420 ``edgefirst.*`` namespace, so each one is
installed separately and imported as ``edgefirst.<name>``.

The decoder is configured from the model's embedded ``edgefirst.json``
schema, so fused, logical-split and per-scale FPN-split ("smart") exports all
work without manual output classification. A model without that schema falls
back to inferring the layout from tensor shapes, which cannot express the
per-scale form.

Models:
    Official pre-trained models are published in the EdgeFirst model zoo on
    Hugging Face:

      Detection:    https://huggingface.co/EdgeFirst/yolov8-det
      Segmentation: https://huggingface.co/EdgeFirst/yolov8-seg

    Each repository ships a ``tflite/`` directory for i.MX8MP and other
    VxDelegate/CPU targets and an ``imx95/`` directory of ``.imx95.tflite``
    exports compiled for the i.MX95 Neutron NPU, in n/s/m sizes.

      curl -LO https://huggingface.co/EdgeFirst/yolov8-det/resolve/main/tflite/yolov8n-det-int8-smart.tflite
      curl -LO https://huggingface.co/EdgeFirst/yolov8-seg/resolve/main/imx95/yolov8n-seg-int8-smart.imx95.tflite

    A stock Ultralytics export also runs here unmodified -- no EdgeFirst
    tooling and no ``edgefirst.json``:

      yolo export model=yolov8n.pt format=tflite int8=True imgsz=640

    Its schema is inferred from the model's own signals (see
    ``resolve_schema``). Note that Ultralytics' LiteRT exports keep PyTorch's
    NCHW layout and a float32 boundary, so they take the CPU staging path
    rather than zero-copy, and are correspondingly slower.

Usage:
    python yolov8.py <model.tflite> <image.jpg> [options]

Examples:
    # CPU-only detection
    python yolov8.py yolov8n-det-int8-smart.tflite zidane.jpg

    # Detection on i.MX8MP, 5 warmup + 100 benchmark iterations
    python yolov8.py yolov8n-det-int8-smart.tflite zidane.jpg \
        --delegate /usr/lib/libvx_delegate.so --warmup 5 --iters 100 --save

    # Segmentation on i.MX95 Neutron
    python yolov8.py yolov8n-seg-int8-smart.imx95.tflite zidane.jpg \
        --delegate /usr/lib/libneutron_delegate.so --warmup 5 --iters 100 --save

Requirements:
    pip install edgefirst-tflite>=0.10.1 numpy \
        edgefirst-codec>=0.31.0 edgefirst-decoder>=0.31.0 \
        edgefirst-image>=0.31.0 edgefirst-tensor>=0.31.0

    edgefirst-tflite 0.10.1 is the floor for stock Ultralytics exports
    specifically: they store constant buffers outside the flatbuffer, which
    earlier versions could not resolve through the TFLite C API, and the model
    aborted at invoke with "Input tensor N lacks data". EdgeFirst-converted
    models run on older versions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

# ── COCO class labels ─────────────────────────────────────────────────────────

COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_outputs(output_details: list[dict]) -> list:
    """Classify TFLite output tensors into edgefirst.decoder Output objects."""
    from edgefirst.decoder import Output

    has_protos = False
    proto_channels = 0
    has_split_boxes = False

    for det in output_details:
        shape = tuple(det["shape"])
        if len(shape) == 4:
            has_protos = True
            proto_channels = min(shape[1:])
        elif len(shape) >= 2:
            feat = shape[1] if len(shape) == 3 else shape[0]
            if feat == 4:
                has_split_boxes = True

    outputs = []
    for det in output_details:
        shape = list(det["shape"])
        quant = det.get("quantization_parameters", {})
        scale = quant.get("scales", [0.0])
        zero_point = quant.get("zero_points", [0])
        scale = float(scale[0]) if len(scale) > 0 else 0.0
        zero_point = int(zero_point[0]) if len(zero_point) > 0 else 0
        is_quantized = str(det.get("dtype", "float32")) != "float32"

        if len(shape) == 4:
            out = Output.protos(shape=shape)
        elif has_split_boxes:
            feat = shape[1] if len(shape) == 3 else shape[0]
            if feat == 4:
                out = Output.boxes(shape=shape)
            elif has_protos and proto_channels > 0 and feat == proto_channels:
                out = Output.mask_coefficients(shape=shape)
            else:
                out = Output.scores(shape=shape)
        else:
            out = Output.detection(shape=shape)

        if is_quantized and scale != 0.0:
            out.with_quantization(scale, zero_point)

        outputs.append(out)

    return outputs


def resolve_schema(model_path, input_details, output_details):
    """Resolve the decoder schema and class labels for a model.

    Three sources, in descending order of authority:

    1. **``edgefirst.json``**, embedded by the EdgeFirst converter in a ZIP
       trailer appended to the flatbuffer. It describes the output layout
       exactly, including the per-scale FPN children a "smart" export needs.
    2. **Inference from the model's own signals**, for a stock Ultralytics
       export that carries no ``edgefirst.json``.
       ``infer_ultralytics_schema`` reads the shapes, dtypes and quantization
       of the boundary tensors together with Ultralytics' ``metadata.json``
       envelope (``names``, ``task``) and reconstructs the equivalent schema.
       This is what lets a stock ``yolo export format=tflite`` model run here
       unmodified.
    3. **Nothing** -- the caller falls back to classifying the outputs by
       shape, which handles fused and logical-split layouts only.

    Returns ``(schema_dict_or_None, labels, description)``.
    """
    from edgefirst.decoder import infer_ultralytics_schema
    from edgefirst_tflite import ModelArchive

    # A model may carry a ZIP trailer holding `metadata.json` but no
    # `edgefirst.json` -- that is exactly the stock Ultralytics case -- so a
    # missing EdgeFirst schema falls through to inference rather than failing.
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
            try:
                labels = archive.labels()
            except Exception:
                labels = []
            return schema, labels, "edgefirst.json embedded"

        # Collect every metadata string the model offers and let inference
        # scan them. Ultralytics writes its envelope as `metadata.json`; the
        # inference looks for whichever value parses as JSON carrying `names`,
        # so offering all of them avoids encoding a guess about the exporter.
        for name in archive.entry_names():
            try:
                metadata[name] = archive.read_to_string(name)
            except Exception:
                pass

    def signals(details):
        """Describe boundary tensors as (name, shape, dtype, quantization)."""
        described = []
        for det in details:
            scale, zero_point = det["quantization"]
            # A scale of zero is TFLite's "not quantized" encoding.
            quant = ([float(scale)], [int(zero_point)]) if scale != 0.0 else None
            described.append((
                det["name"],
                [int(s) for s in det["shape"]],
                str(det["dtype"]),
                quant,
            ))
        return described

    try:
        inferred = infer_ultralytics_schema(
            # The box convention follows the container, and TFLite exports are
            # normalized to [0, 1]. Naming the source is what lets inference
            # avoid guessing it.
            "tflite",
            signals(input_details),
            signals(output_details),
            metadata,
        )
    except ValueError:
        # No recognizable Ultralytics signature. Not an error: the caller has
        # a shape-based fallback, and reporting here would be noise for the
        # EdgeFirst models that never reach this branch.
        return None, [], None

    return inferred.schema, inferred.labels, f"inferred ({inferred.description})"


def load_image(processor, path, fmt, access="none"):
    """Decode an image file into a GPU-backed tensor in `fmt`.

    edgefirst.codec decodes into a pre-allocated tensor in the source's native
    pixel format (JPEG → NV12, PNG → RGB/RGBA), so peek the header to size a
    native-format tensor, decode into it, then let the processor convert to
    `fmt`.

    `access` declares what the *CPU* will do with the result. The default
    "none" keeps hardware pipelines eligible for vendor tile compression;
    anything the script later maps itself — normalize_to_numpy(), save_jpeg(),
    map() — must declare "read", "write" or "readwrite" or the map fails.
    """
    import edgefirst.codec as codec

    info = codec.Tensor.peek_image_info_file(path)
    # When the source decodes straight to `fmt` (a PNG already in RGBA) this
    # buffer is returned as-is, so it must carry the caller's `access`, not
    # just the "write" the decode itself needs.
    decodes_to_fmt = info.format == fmt
    native = processor.create_image(
        info.width, info.height, info.format, "uint8",
        access=access if decodes_to_fmt else "write",
    )
    codec.decode_file_into(native, path)
    if native.format == fmt:
        return native

    out = processor.create_image(
        info.width, info.height, fmt, "uint8", access=access,
    )
    processor.convert(native, out)
    return out


def compute_letterbox(src_w, src_h, dst_w, dst_h):
    """Compute the letterbox destination rect preserving aspect ratio.

    Mirrors `letterbox_rect` in edgefirst-image exactly, including its
    rounding. `convert(letterbox=...)` places the image itself; this only
    recovers where it put it, for `letterbox_norm` and the inverse box
    transform. Scaling by `int(src * scale)` instead rounds the other way for
    roughly half of all source sizes, which lands masks a row off and scales
    printed boxes by the wrong factor.

    `floor(x + 0.5)`, not `round()`: Rust's `f64::round` goes half away from
    zero while Python's `round` is banker's rounding, and the two disagree on
    exact halves.
    """
    if src_w == 0 or src_h == 0:
        return 0, 0, dst_w, dst_h
    src_aspect = src_w / src_h
    dst_aspect = dst_w / dst_h
    if src_aspect > dst_aspect:
        new_w, new_h = dst_w, max(1, math.floor(dst_w / src_aspect + 0.5))
    else:
        new_w, new_h = max(1, math.floor(dst_h * src_aspect + 0.5)), dst_h
    left = max(0, dst_w - new_w) // 2
    top = max(0, dst_h - new_h) // 2
    return left, top, new_w, new_h


def percentile(values, p):
    """Compute percentile from a sorted array."""
    idx = min(int(np.ceil((len(values) - 1) * p)), len(values) - 1)
    return values[idx]


def print_stats(label, timings):
    """Print timing statistics for a set of iterations."""
    n = len(timings["total"])
    if n == 0:
        return

    stages = ("preprocess", "infer", "copy", "decode", "materialize",
              "render", "total")

    if n == 1:
        print(f"--- {label} ---")
        for name in stages:
            vals = timings[name]
            if not vals:
                continue
            lbl = name.capitalize() + ":"
            print(f"  {lbl:<14}{vals[0]:.1f}ms")
        return

    print(f"--- {label} ({n} iterations) ---")
    print("                    min      max      avg      p95      p99")
    for name in stages:
        vals = timings[name]
        if not vals:
            continue
        s = sorted(vals)
        mn, mx = s[0], s[-1]
        avg = sum(s) / len(s)
        p95 = percentile(s, 0.95)
        p99 = percentile(s, 0.99)
        lbl = name.capitalize()
        print(f"  {lbl:<10} {mn:>7.1f} {mx:>8.1f} {avg:>8.1f} {p95:>8.1f} {p99:>8.1f} ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 detection & segmentation with edgefirst-tflite",
    )
    parser.add_argument("model", help="Path to .tflite model")
    parser.add_argument("image", help="Path to input image (JPEG/PNG)")
    parser.add_argument("--delegate", help="Path to delegate .so")
    parser.add_argument("--save", action="store_true", help="Save overlay image")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Score threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of warmup iterations (default: 0)")
    parser.add_argument("--iters", type=int, default=1,
                        help="Number of benchmark iterations (default: 1)")
    args = parser.parse_args()

    # ── 1. Load library, model, delegate, interpreter ────────────────
    t_load = time.perf_counter()

    from edgefirst_tflite import Interpreter, load_delegate

    delegates = []
    if args.delegate:
        delegate = load_delegate(args.delegate)
        print(f"Delegate: {args.delegate}")
        if delegate.has_dmabuf:
            print("  DMA-BUF: available")
        if delegate.has_camera_adaptor:
            print("  CameraAdaptor: available")
        delegates.append(delegate)

    interpreter = Interpreter(
        model_path=args.model,
        experimental_delegates=delegates if delegates else None,
        num_threads=4,
    )

    load_time = (time.perf_counter() - t_load) * 1000
    print(f"Model: {args.model}")

    # ── 2. Inspect tensors ───────────────────────────────────────────
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Inputs:  {len(input_details)}")
    print(f"Outputs: {len(output_details)}")

    inp = input_details[0]
    in_shape = tuple(int(s) for s in inp["shape"])
    in_dtype = str(inp["dtype"])

    # Channel axis, not assumed. The EdgeFirst converter emits NHWC, but a
    # stock Ultralytics LiteRT export keeps PyTorch's NCHW, and reading H and W
    # off the wrong axes is silent: it builds a 3-pixel-wide letterbox, feeds
    # the model noise and reports zero detections rather than an error.
    is_nchw = (
        len(in_shape) == 4 and in_shape[1] in (1, 3) and in_shape[3] not in (1, 3)
    )
    if is_nchw:
        in_h, in_w = in_shape[2], in_shape[3]
    else:
        in_h, in_w = in_shape[1], in_shape[2]
    print(f"  input[0]: shape={list(in_shape)} dtype={in_dtype} "
          f"layout={'NCHW' if is_nchw else 'NHWC'}")

    for i, det in enumerate(output_details):
        print(f"  output[{i}]: shape={list(det['shape'])} dtype={det['dtype']}")

    # ── 3. Auto-detect model type and build Decoder ──────────────────
    from edgefirst.decoder import Decoder, DecoderVersion, Nms

    schema, labels, schema_desc = resolve_schema(
        args.model, input_details, output_details,
    )
    if not labels:
        labels = COCO

    if schema is not None:
        # The model's own layout, either declared or inferred. This is the
        # only path that handles per-scale ("smart") exports -- which is every
        # model in the EdgeFirst zoo -- and it mirrors the Rust example's
        # DecoderBuilder::with_schema.
        is_segmentation = any(
            o.get("type") == "protos" for o in schema.get("outputs", [])
        )
        decoder = Decoder(
            schema,
            score_threshold=args.threshold,
            iou_threshold=args.iou,
            nms=Nms.ClassAgnostic,
        )
        print(f"  Schema: {schema_desc}, {len(labels)} labels")
    else:
        # Last resort: classify the outputs by shape. Fused and logical-split
        # only -- a smart export lands here as InvalidConfig("Invalid Yolo
        # model outputs").
        is_segmentation = any(len(d["shape"]) == 4 for d in output_details)
        decoder = Decoder.new_from_outputs(
            classify_outputs(output_details),
            score_threshold=args.threshold,
            iou_threshold=args.iou,
            nms=Nms.ClassAgnostic,
            decoder_version=DecoderVersion.Yolov8,
        )
        print("  Schema: none available, classified from output shapes")

    print(f"  Mode: {'segmentation' if is_segmentation else 'detection'}")

    # ── 4. Setup image preprocessing ─────────────────────────────────
    from edgefirst.image import ColorMode, Flip, ImageProcessor, Rotation
    from edgefirst.tensor import PixelFormat, Tensor

    # Pre-allocate HAL output tensors (reused each iteration, like Rust OutputBuffers).
    DTYPE_MAP = {"int8": "int8", "uint8": "uint8", "float32": "float32", "int32": "int32"}
    NP_DTYPE_MAP = {"int8": np.int8, "uint8": np.uint8, "float32": np.float32, "int32": np.int32}
    output_tensors = []
    for det in output_details:
        shape = [int(s) for s in det["shape"]]
        dt_str = str(det["dtype"])
        dt = DTYPE_MAP.get(dt_str, "float32")
        tensor = Tensor(shape, dtype=dt)
        # The schema decoder reads quantization off the tensors it is handed,
        # per scale, the way the Rust example's OutputBuffers does. Without
        # this a smart model fails with QuantMissing { role: "boxes" }.
        scale, zero_point = det["quantization"]
        if scale != 0.0:
            tensor.set_quantization_per_tensor(scale, zero_point)
        output_tensors.append(tensor)

    t_init = time.perf_counter()

    processor = ImageProcessor()

    src = load_image(processor, args.image, PixelFormat.Rgba, access="read")
    img_w, img_h = src.width, src.height
    print(f"Image: {img_w}x{img_h}")

    dst_dtype = "int8" if in_dtype == "int8" else "uint8"
    left, top, new_w, new_h = compute_letterbox(img_w, img_h, in_w, in_h)

    # Create preprocessing destination buffer (reused each iteration).
    #
    # Path A: DMA-BUF + integer model → zero-copy GPU → NPU
    # Path B: No DMA-BUF → copy via normalize_to_numpy + set_tensor
    dmabuf = interpreter.dmabuf()
    # An NCHW model cannot take the zero-copy path: the GPU writes interleaved
    # pixels, so the channel-first buffer has to be built on the CPU.
    use_dmabuf = dmabuf is not None and in_dtype != "float32" and not is_nchw

    if use_dmabuf:
        # Try portable HAL tensor_info API first (Neutron, future delegates),
        # fall back to VxDelegate legacy request/bind_to_tensor path.
        # If both fail (e.g. multi-partition Neutron), fall back to CPU staging.
        info = None
        try:
            info = dmabuf.tensor_info(0)
        except Exception:
            pass

        if info is not None:
            dst_img = processor.import_image(
                info["fd"], in_w, in_h, PixelFormat.Rgb, dst_dtype,
                offset=info.get("offset"),
            )
            print("  Input: HAL DMA-BUF zero-copy (GPU → DMA-BUF → NPU)")
        else:
            try:
                buf_size = in_h * in_w * 3
                handle, desc = dmabuf.request(0, "delegate", buf_size)
                dmabuf.bind_to_tensor(handle, 0)
                dst_img = processor.import_image(
                    desc["fd"], in_w, in_h, PixelFormat.Rgb, dst_dtype,
                )
                print("  Input: VxDelegate DMA-BUF (legacy, GPU → DMA-BUF → NPU)")
            except Exception:
                use_dmabuf = False

    if not use_dmabuf:
        dst_img = processor.create_image(
            in_w, in_h, PixelFormat.Rgb, dst_dtype, access="readwrite",
        )
        print("  Input: CPU staging (GPU → staging → TFLite arena)")

    # Pre-allocate numpy buffer for CPU path.
    if not use_dmabuf:
        if in_dtype == "int8":
            input_array = np.zeros((in_h, in_w, 3), dtype=np.int8)
        else:
            input_array = np.zeros((in_h, in_w, 3), dtype=np.uint8)

    init_time = (time.perf_counter() - t_init) * 1000

    # Pre-allocate overlay for rendering (reused across iterations).
    overlay = (
        load_image(processor, args.image, PixelFormat.Rgba, access="readwrite")
        if args.save
        else None
    )

    # Precompute normalised letterbox rect for materialize_masks / draw_decoded_masks.
    letterbox_norm = (left / in_w, top / in_h,
                      (left + new_w) / in_w, (top + new_h) / in_h)

    # ── 5. Iteration runner ──────────────────────────────────────────
    def run_iterations(n, render=False):
        timings = {
            "preprocess": [], "infer": [], "copy": [], "decode": [],
            "materialize": [], "render": [], "total": [],
        }
        boxes = scores = classes = masks = None

        for iteration in range(n):
            t_total = time.perf_counter()

            # Preprocess: GPU convert + copy to input tensor.
            t0 = time.perf_counter()
            processor.convert(
                src, dst_img,
                rotation=Rotation.Rotate0,
                flip=Flip.NoFlip,
                letterbox=(114, 114, 114, 255),  # YOLO gray pad
            )
            if use_dmabuf:
                dmabuf.sync_for_device(0)
            else:
                dst_img.normalize_to_numpy(input_array)
                if in_dtype == "float32":
                    staged = input_array.astype(np.float32) / 255.0
                else:
                    staged = input_array
                if is_nchw:
                    # HWC → CHW. `set_tensor` copies the buffer flat, so the
                    # array has to be made contiguous in the new order rather
                    # than merely reshaped.
                    staged = np.ascontiguousarray(staged.transpose(2, 0, 1))
                interpreter.set_tensor(0, staged)
            timings["preprocess"].append((time.perf_counter() - t0) * 1000)

            # Infer
            t0 = time.perf_counter()
            interpreter.invoke()
            timings["infer"].append((time.perf_counter() - t0) * 1000)

            # Copy output tensors from TFLite arena → HAL tensors.
            t0 = time.perf_counter()
            for oi, det in enumerate(output_details):
                np_arr = interpreter.get_output_tensor(oi)
                np_dtype = NP_DTYPE_MAP.get(str(det["dtype"]), np.float32)
                with output_tensors[oi].map() as m:
                    dst = np.frombuffer(m, dtype=np_dtype).reshape(np_arr.shape)
                    np.copyto(dst, np_arr)
            timings["copy"].append((time.perf_counter() - t0) * 1000)

            # Decode: decode_proto returns proto data for seg models (None for det).
            t0 = time.perf_counter()
            boxes, scores, classes, proto_data = decoder.decode_proto(
                output_tensors,
            )

            # Normalize if in pixel space.
            if len(scores) > 0 and np.max(boxes) > 2.0:
                boxes[:, [0, 2]] /= in_w
                boxes[:, [1, 3]] /= in_h
            timings["decode"].append((time.perf_counter() - t0) * 1000)

            # Materialize: CPU dot-product of mask coefficients × protos → bitmaps.
            t0 = time.perf_counter()
            if proto_data is not None:
                masks = processor.materialize_masks(
                    boxes, scores, classes, proto_data,
                    letterbox=letterbox_norm,
                )
            else:
                masks = []
            timings["materialize"].append((time.perf_counter() - t0) * 1000)

            # Render overlay on the last iteration only.  Rendering mid-loop
            # triggers a GPU/NPU sync issue on i.MX95 (Mali-G310 + Neutron
            # shared DMA-BUF): draw_decoded_masks queues GPU work that races
            # with the next convert → sync_for_device → invoke cycle.
            if render and overlay is not None and iteration == n - 1:
                t0 = time.perf_counter()
                processor.draw_decoded_masks(
                    overlay, boxes, scores, classes,
                    seg=masks, background=src,
                    letterbox=letterbox_norm,
                    color_mode=ColorMode.Instance,
                )
                timings["render"].append((time.perf_counter() - t0) * 1000)

            timings["total"].append((time.perf_counter() - t_total) * 1000)

        return boxes, scores, classes, masks, timings

    # ── 6. Warmup ────────────────────────────────────────────────────
    if args.warmup > 0:
        print(f"\nRunning {args.warmup} warmup iterations...")
        _, _, _, _, warmup_timings = run_iterations(args.warmup)
        print()
        print_stats("Warmup", warmup_timings)

    # ── 7. Benchmark ─────────────────────────────────────────────────
    boxes, scores, classes, masks, bench_timings = run_iterations(
        args.iters, render=args.save,
    )

    # ── 8. Print detections (from last iteration) ────────────────────
    num_detections = len(scores) if scores is not None else 0
    print(f"\n--- Detections ({num_detections}) ---")
    # Inverse letterbox transform: map normalised model coords → original image pixels.
    lx0 = left / in_w
    ly0 = top / in_h
    inv_lw = in_w / new_w
    inv_lh = in_h / new_h
    for i in range(num_detections):
        label = int(classes[i])
        name = labels[label] if label < len(labels) else "?"
        score = float(scores[i]) * 100.0
        bx0 = max(0.0, min(1.0, float(boxes[i, 0])))
        by0 = max(0.0, min(1.0, float(boxes[i, 1])))
        bx1 = max(0.0, min(1.0, float(boxes[i, 2])))
        by1 = max(0.0, min(1.0, float(boxes[i, 3])))
        x1 = max(0.0, min(float(img_w), (bx0 - lx0) * inv_lw * img_w))
        y1 = max(0.0, min(float(img_h), (by0 - ly0) * inv_lh * img_h))
        x2 = max(0.0, min(float(img_w), (bx1 - lx0) * inv_lw * img_w))
        y2 = max(0.0, min(float(img_h), (by1 - ly0) * inv_lh * img_h))
        print(f"  {name:>12} ({label:2d}): {score:5.1f}%  "
              f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    # ── 9. Save overlay (once) ───────────────────────────────────────
    if overlay is not None:
        stem, _ext = os.path.splitext(os.path.basename(args.image))
        out_dir = os.path.dirname(args.image) or "."
        out_path = os.path.join(out_dir, f"{stem}_overlay.jpg")
        overlay.save_jpeg(out_path, 95)
        print(f"  Saved: {out_path}")

    # ── 10. Print timing ─────────────────────────────────────────────
    print()
    print(f"  Load+init:  {load_time + init_time:.1f}ms"
          f"  (model + image \u2192 DMA-BUF, one-time)")
    print()
    print_stats(f"Benchmark (iters={args.iters})", bench_timings)


if __name__ == "__main__":
    main()
