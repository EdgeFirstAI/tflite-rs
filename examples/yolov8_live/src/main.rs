// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! # Live `YOLOv8` Camera Inference with `edgefirst-tflite`
//!
//! Captures NV12 or YUYV frames from a camera through libcamera, runs `YOLOv8`
//! detection or instance segmentation with `edgefirst-tflite`, and shows the
//! result in a native Wayland window. Overlays are composited onto the camera
//! frame by the `EdgeFirst` HAL (`edgefirst-image`).
//!
//! The pipeline is intentionally single-threaded and synchronous so each
//! stage's cost is visible in the per-frame timings.
//!
//! ## Pipeline
//!
//! ```text
//! libcamera (NV12 | YUYV DMA-BUF)
//!   -> HAL import            (cached per camera buffer)
//!   -> HAL convert           (letterbox into the delegate's input DMA-BUF)
//!   -> TFLite invoke         (Neutron / VxDelegate NPU, or CPU)
//!   -> HAL draw_masks        (decode + composite onto an RGBA DMA-BUF canvas)
//!   -> Wayland               (canvas DMA-BUF -> wl_buffer -> compositor)
//! ```
//!
//! Every hop is a DMA-BUF hand-off except the model outputs, which are copied
//! out of the `TFLite` arena into HAL tensors (a few hundred KiB per frame).
//! Display uses `zwp_linux_dmabuf_v1` to give the canvas to the compositor
//! directly — no EGL or GL on the client side.
//!
//! ## Models
//!
//! Official pre-trained models are published in the `EdgeFirst` model zoo on
//! Hugging Face:
//!
//! - Detection: <https://huggingface.co/EdgeFirst/yolov8-det>
//! - Segmentation: <https://huggingface.co/EdgeFirst/yolov8-seg>
//!
//! ```sh
//! # i.MX95 Neutron
//! curl -LO https://huggingface.co/EdgeFirst/yolov8-det/resolve/main/imx95/yolov8n-det-int8-smart.imx95.tflite
//! curl -LO https://huggingface.co/EdgeFirst/yolov8-seg/resolve/main/imx95/yolov8n-seg-int8-smart.imx95.tflite
//!
//! # i.MX8MP VxDelegate, or CPU
//! curl -LO https://huggingface.co/EdgeFirst/yolov8-det/resolve/main/tflite/yolov8n-det-int8-smart.tflite
//! ```
//!
//! The decoder is configured from the `edgefirst.json` schema embedded in
//! each export, the same way as the still-image `yolov8` example.
//!
//! ## Usage
//!
//! ```text
//! yolov8-live <model.tflite> [--delegate <path>] [--threshold N] [--iou N]
//!             [--width N] [--height N] [--camera-name ID]
//!             [--format nv12|yuyv] [--color-mode class|instance|track]
//! ```
//!
//! ```sh
//! # i.MX95 Neutron, run from an SSH session on the board
//! XDG_RUNTIME_DIR=/run/user/0 WAYLAND_DISPLAY=wayland-0 \
//! LIBCAMERA_PIPELINES_MATCH_LIST=nxp/neo \
//!     yolov8-live yolov8n-seg-int8-smart.imx95.tflite \
//!     --delegate /usr/lib/libneutron_delegate.so
//! ```
//!
//! On i.MX95 the `imx8-isi` pipeline handler claims the ISI before the
//! `NeoISP` handler does, and offers only raw Bayer that the HAL cannot
//! convert. Restricting libcamera to `nxp/neo` routes the sensor through the
//! ISP, which produces NV12 and YUYV.
//!
//! ## Requirements
//!
//! - libcamera >= 0.4 (headers at build time; cross builds must use headers
//!   matching the target's libcamera, since the bindings wrap its C++ API)
//! - A Wayland compositor (Weston) offering `zwp_linux_dmabuf_v1`

mod error;

use std::collections::{BTreeMap, HashMap};
use std::os::fd::{AsRawFd as _, BorrowedFd};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use edgefirst_decoder::{
    infer_ultralytics_schema,
    schema::{DType as SchemaDType, LogicalType, Quantization as SchemaQuantization, SchemaV2},
    Decoder, DecoderBuilder, ModelSignals, ModelSource, TensorInfo,
};
use edgefirst_image::{
    ColorMode, ConvertOptions, ImageProcessor, ImageProcessorTrait as _, MaskOverlay,
};
use edgefirst_tensor::{
    ColorEncoding, ColorRange, ColorSpace, ColorTransfer, Colorimetry, CpuAccess, DType,
    PixelFormat, PlaneDescriptor, Quantization, TensorDyn, TensorMemory,
};
use edgefirst_tflite::{
    archive::ModelArchive, metadata::Metadata, Delegate, Interpreter, Library, Model, Tensor,
    TensorType,
};
use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer as _,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    request::ReuseFlag,
    stream::StreamRole,
};
use wayland_client::protocol::{wl_buffer, wl_callback, wl_compositor, wl_registry, wl_surface};
use wayland_client::{delegate_noop, Connection, Dispatch, EventQueue, QueueHandle};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1, zwp_linux_dmabuf_v1,
};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

// ── Arguments ────────────────────────────────────────────────────────────────

struct Args {
    model: PathBuf,
    delegate: Option<PathBuf>,
    threshold: f32,
    iou: f32,
    width: usize,
    height: usize,
    camera_name: Option<String>,
    format: PixelFormat,
    color_mode: ColorMode,
}

fn usage(program: &str) -> ! {
    eprintln!(
        "Usage: {program} <model.tflite> [--delegate <path>] [--threshold N] [--iou N] \
         [--width N] [--height N] [--camera-name ID] [--format nv12|yuyv] \
         [--color-mode class|instance|track]"
    );
    std::process::exit(1);
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with('-') {
        usage(&args[0]);
    }
    let mut parsed = Args {
        model: PathBuf::from(&args[1]),
        delegate: None,
        // Higher than the still-image example's 0.25: live video wants fewer
        // false positives per frame to keep the overlay stable.
        threshold: 0.50,
        iou: 0.45,
        width: 1920,
        height: 1080,
        camera_name: None,
        format: PixelFormat::Nv12,
        color_mode: ColorMode::Class,
    };
    let mut i = 2;
    macro_rules! next_arg {
        ($flag:expr) => {{
            i += 1;
            if i >= args.len() {
                eprintln!("missing value for {}", $flag);
                std::process::exit(1);
            }
            args[i].as_str()
        }};
    }
    macro_rules! parse_value {
        ($flag:expr) => {
            next_arg!($flag).parse().unwrap_or_else(|_| {
                eprintln!("invalid {} value", $flag);
                std::process::exit(1);
            })
        };
    }
    while i < args.len() {
        match args[i].as_str() {
            "--delegate" => parsed.delegate = Some(PathBuf::from(next_arg!("--delegate"))),
            "--threshold" => parsed.threshold = parse_value!("--threshold"),
            "--iou" => parsed.iou = parse_value!("--iou"),
            "--width" => parsed.width = parse_value!("--width"),
            "--height" => parsed.height = parse_value!("--height"),
            "--camera-name" => parsed.camera_name = Some(next_arg!("--camera-name").to_string()),
            "--format" => {
                parsed.format = match next_arg!("--format") {
                    "nv12" => PixelFormat::Nv12,
                    "yuyv" | "yuy2" => PixelFormat::Yuyv,
                    other => {
                        eprintln!("unknown --format {other} (expected nv12 or yuyv)");
                        std::process::exit(1);
                    }
                };
            }
            "--color-mode" => {
                parsed.color_mode = match next_arg!("--color-mode") {
                    "class" => ColorMode::Class,
                    "instance" => ColorMode::Instance,
                    "track" => ColorMode::Track,
                    other => {
                        eprintln!(
                            "unknown --color-mode {other} (expected class, instance or track)"
                        );
                        std::process::exit(1);
                    }
                };
            }
            "-h" | "--help" => usage(&args[0]),
            other => eprintln!("Unknown argument: {other}"),
        }
        i += 1;
    }
    parsed
}

// ── Wayland display (direct DMA-BUF submission, no EGL/GL) ───────────────────

/// DRM fourcc `ABGR8888`: R, G, B, A bytes in memory order, which is the
/// layout of the HAL's [`PixelFormat::Rgba`].
const DRM_FORMAT_ABGR8888: u32 = u32::from_le_bytes(*b"AB24");

/// Event-loop state for the Wayland connection.
#[derive(Debug, Default)]
struct DisplayState {
    compositor: Option<wl_compositor::WlCompositor>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    dmabuf: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    surface: Option<wl_surface::WlSurface>,
    configured: bool,
    closed: bool,
    /// `true` when the compositor is ready for the next frame. Cleared on
    /// submit and set again by the `wl_surface.frame` callback.
    frame_done: bool,
    /// One `wl_buffer` per DMA-BUF fd, imported on first submit.
    buffers: HashMap<i32, wl_buffer::WlBuffer>,
}

/// An XDG toplevel window that presents DMA-BUF RGBA buffers.
struct WaylandDisplay {
    conn: Connection,
    queue: EventQueue<DisplayState>,
    state: DisplayState,
    width: i32,
    height: i32,
}

impl WaylandDisplay {
    /// Connect via `$WAYLAND_DISPLAY`, bind the globals and wait for the
    /// window's first configure.
    fn new(width: usize, height: usize, title: &str) -> Result<Self> {
        let width = i32::try_from(width).map_err(|_| Error::display("width out of range"))?;
        let height = i32::try_from(height).map_err(|_| Error::display("height out of range"))?;
        let conn = Connection::connect_to_env()
            .map_err(|e| Error::display(format!("no Wayland compositor: {e}")))?;
        let mut queue = conn.new_event_queue();
        let qh = queue.handle();
        conn.display().get_registry(&qh, ());

        let mut state = DisplayState {
            frame_done: true,
            ..DisplayState::default()
        };
        queue
            .roundtrip(&mut state)
            .map_err(|e| Error::display(format!("registry roundtrip: {e}")))?;

        let compositor = state
            .compositor
            .as_ref()
            .ok_or_else(|| Error::display("compositor lacks wl_compositor"))?;
        let wm_base = state
            .wm_base
            .as_ref()
            .ok_or_else(|| Error::display("compositor lacks xdg_wm_base"))?;
        if state.dmabuf.is_none() {
            return Err(Error::display("compositor lacks zwp_linux_dmabuf_v1"));
        }

        let surface = compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_title(title.to_string());
        toplevel.set_app_id("edgefirst-yolov8-live".to_string());
        surface.commit();
        state.surface = Some(surface);

        while !state.configured {
            queue
                .blocking_dispatch(&mut state)
                .map_err(|e| Error::display(format!("dispatch: {e}")))?;
        }

        Ok(Self {
            conn,
            queue,
            state,
            width,
            height,
        })
    }

    /// Present the RGBA DMA-BUF `fd`, whose rows are `stride` bytes apart.
    ///
    /// Paced by `wl_surface.frame`: while the compositor has not consumed the
    /// previous frame this is a no-op, so a fast pipeline drops frames at the
    /// display rather than queueing them. Returns `false` once the window has
    /// been closed.
    fn present(&mut self, fd: BorrowedFd<'_>, stride: u32) -> Result<bool> {
        self.dispatch_pending()?;
        if self.state.closed {
            return Ok(false);
        }
        if !self.state.frame_done {
            return Ok(true);
        }

        let qh = self.queue.handle();
        let (w, h) = (self.width, self.height);
        let raw = fd.as_raw_fd();
        let dmabuf = self.state.dmabuf.as_ref().expect("bound in new()");
        let buffer = self.state.buffers.entry(raw).or_insert_with(|| {
            let params = dmabuf.create_params(&qh, ());
            params.add(fd, 0, 0, stride, 0, 0);
            let buffer = params.create_immed(
                w,
                h,
                DRM_FORMAT_ABGR8888,
                zwp_linux_buffer_params_v1::Flags::empty(),
                &qh,
                (),
            );
            params.destroy();
            buffer
        });

        let surface = self.state.surface.as_ref().expect("created in new()");
        surface.attach(Some(buffer), 0, 0);
        surface.damage_buffer(0, 0, w, h);
        surface.frame(&qh, ());
        self.state.frame_done = false;
        surface.commit();
        self.conn
            .flush()
            .map_err(|e| Error::display(format!("flush: {e}")))?;
        Ok(true)
    }

    /// Process pending events and report whether the window is still open.
    fn is_open(&mut self) -> Result<bool> {
        self.dispatch_pending()?;
        Ok(!self.state.closed)
    }

    fn dispatch_pending(&mut self) -> Result<()> {
        // `dispatch_pending` only drains what is already queued; reading the
        // socket first is what lets frame callbacks and close events arrive
        // without blocking.
        if let Some(guard) = self.queue.prepare_read() {
            match guard.read() {
                Ok(_) => {}
                Err(wayland_client::backend::WaylandError::Io(e))
                    if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(Error::display(format!("read: {e}"))),
            }
        }
        self.queue
            .dispatch_pending(&mut self.state)
            .map_err(|e| Error::display(format!("dispatch: {e}")))?;
        self.conn
            .flush()
            .map_err(|e| Error::display(format!("flush: {e}")))?;
        Ok(())
    }
}

impl Dispatch<wl_registry::WlRegistry, ()> for DisplayState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        (): &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
                    state.compositor = Some(registry.bind(name, version.min(4), qh, ()));
                }
                "xdg_wm_base" => {
                    state.wm_base = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwp_linux_dmabuf_v1" => {
                    state.dmabuf = Some(registry.bind(name, version.min(3), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for DisplayState {
    fn event(
        _: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for DisplayState {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surface.ack_configure(serial);
            state.configured = true;
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for DisplayState {
    fn event(
        state: &mut Self,
        _: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_toplevel::Event::Close = event {
            state.closed = true;
        }
    }
}

impl Dispatch<wl_callback::WlCallback, ()> for DisplayState {
    fn event(
        state: &mut Self,
        _: &wl_callback::WlCallback,
        event: wl_callback::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_callback::Event::Done { .. } = event {
            state.frame_done = true;
        }
    }
}

delegate_noop!(DisplayState: ignore wl_compositor::WlCompositor);
delegate_noop!(DisplayState: ignore wl_surface::WlSurface);
delegate_noop!(DisplayState: ignore wl_buffer::WlBuffer);
delegate_noop!(DisplayState: ignore zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1);
delegate_noop!(DisplayState: ignore zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1);

// ── Camera frames ────────────────────────────────────────────────────────────

/// The HAL colorimetry for a libcamera stream's color space.
///
/// Passing it to the import matters: an untagged YUV frame is converted with
/// a guess made from its height (BT.709, limited range for HD), while ISPs
/// commonly deliver sYCC (BT.601, full range). The mismatch stretches the
/// contrast and shifts the hue of every frame the model sees.
fn colorimetry(cs: libcamera::color_space::ColorSpace) -> Colorimetry {
    use libcamera::color_space::{Primaries, Range, TransferFunction, YcbcrEncoding};
    Colorimetry {
        space: match cs.primaries {
            Primaries::Raw => None,
            Primaries::Smpte170m => Some(ColorSpace::Smpte170m),
            Primaries::Rec709 => Some(ColorSpace::Bt709),
            Primaries::Rec2020 => Some(ColorSpace::Bt2020),
        },
        transfer: Some(match cs.transfer_function {
            TransferFunction::Linear => ColorTransfer::Linear,
            TransferFunction::Srgb => ColorTransfer::Srgb,
            TransferFunction::Rec709 => ColorTransfer::Bt709,
        }),
        encoding: match cs.ycbcr_encoding {
            YcbcrEncoding::None => None,
            YcbcrEncoding::Rec601 => Some(ColorEncoding::Bt601),
            YcbcrEncoding::Rec709 => Some(ColorEncoding::Bt709),
            YcbcrEncoding::Rec2020 => Some(ColorEncoding::Bt2020),
        },
        range: Some(match cs.range {
            Range::Full => ColorRange::Full,
            Range::Limited => ColorRange::Limited,
        }),
    }
}

/// libcamera's fourcc for a HAL capture format.
fn libcamera_format(format: PixelFormat) -> libcamera::pixel_format::PixelFormat {
    let fourcc = if format == PixelFormat::Yuyv {
        *b"YUYV"
    } else {
        *b"NV12"
    };
    libcamera::pixel_format::PixelFormat::new(u32::from_le_bytes(fourcc), 0)
}

/// HAL image tensors for libcamera's buffers, imported once per buffer.
///
/// `FrameBufferAllocator` hands out a fixed pool whose DMA-BUFs live as long
/// as the camera, so each is imported on first use and indexed by the request
/// cookie. The import dups the fds, so a request can be requeued as soon as
/// its frame has been looked up.
struct FrameCache {
    entries: Vec<Option<TensorDyn>>,
    width: usize,
    height: usize,
    stride: usize,
    format: PixelFormat,
    colorimetry: Option<Colorimetry>,
}

impl FrameCache {
    fn get_or_import(
        &mut self,
        index: usize,
        processor: &ImageProcessor,
        framebuffer: &FrameBuffer,
    ) -> Result<&TensorDyn> {
        let slot = self
            .entries
            .get_mut(index)
            .ok_or_else(|| Error::camera(format!("request cookie {index} out of range")))?;
        if slot.is_none() {
            let planes = framebuffer.planes();
            let luma = planes
                .get(0)
                .ok_or_else(|| Error::camera("frame buffer has no planes"))?;
            // SAFETY: the fd belongs to a libcamera FrameBuffer that outlives
            // this call; PlaneDescriptor::new dups it, so the imported tensor
            // does not depend on the borrow.
            let primary = PlaneDescriptor::new(unsafe { BorrowedFd::borrow_raw(luma.fd()) })?
                .with_stride(self.stride);
            // NV12 may arrive as a second plane: a separate DMA-BUF, or the
            // same one at an offset past the luma rows.
            let chroma = match planes.get(1) {
                Some(plane) if self.format == PixelFormat::Nv12 => {
                    // SAFETY: as for the luma plane above.
                    let mut desc =
                        PlaneDescriptor::new(unsafe { BorrowedFd::borrow_raw(plane.fd()) })?
                            .with_stride(self.stride);
                    if let Some(offset) = plane.offset().filter(|&o| o > 0) {
                        desc = desc.with_offset(offset);
                    }
                    Some(desc)
                }
                _ => None,
            };
            *slot = Some(processor.import_image(
                primary,
                chroma,
                self.width,
                self.height,
                self.format,
                DType::U8,
                self.colorimetry,
            )?);
        }
        Ok(slot.as_ref().expect("imported above"))
    }
}

// ── Model setup ──────────────────────────────────────────────────────────────

fn tflite_dtype(tt: TensorType) -> Result<DType> {
    match tt {
        TensorType::Float32 => Ok(DType::F32),
        TensorType::Int8 => Ok(DType::I8),
        TensorType::UInt8 => Ok(DType::U8),
        TensorType::Int32 => Ok(DType::I32),
        other => Err(Error::unsupported(format!("output dtype: {other:?}"))),
    }
}

/// Maps a `TFLite` element type onto the schema's model-boundary `DType`.
fn schema_dtype(ty: TensorType) -> Result<SchemaDType> {
    Ok(match ty {
        TensorType::Float32 => SchemaDType::Float32,
        TensorType::Float16 => SchemaDType::Float16,
        TensorType::Int8 => SchemaDType::Int8,
        TensorType::UInt8 => SchemaDType::Uint8,
        TensorType::Int16 => SchemaDType::Int16,
        TensorType::Int32 => SchemaDType::Int32,
        other => {
            return Err(Error::unsupported(format!(
                "tensor dtype {other:?} is not a supported model boundary type"
            )))
        }
    })
}

/// Describes one model boundary tensor for schema inference.
fn tensor_info(tensor: &Tensor<'_>) -> Result<TensorInfo> {
    let qp = tensor.quantization_params();
    let dtype = schema_dtype(tensor.tensor_type())?;
    Ok(TensorInfo {
        name: tensor.name().to_string(),
        shape: tensor.shape()?,
        dtype,
        // A scale of zero is TFLite's "not quantized" encoding, not a scale.
        quantization: (qp.scale != 0.0).then(|| SchemaQuantization {
            scale: vec![qp.scale],
            zero_point: Some(vec![qp.zero_point]),
            axis: None,
            dtype: Some(dtype),
        }),
    })
}

/// Resolves the decoder schema: the embedded `edgefirst.json` when present,
/// otherwise one inferred from a stock Ultralytics export's own signals. See
/// `load_schema` in the `yolov8` example for the full rationale.
fn load_schema(model_data: &[u8], interpreter: &Interpreter<'_>) -> Result<SchemaV2> {
    let mut archive = ModelArchive::new(model_data).ok();
    if let Some(archive) = archive.as_mut() {
        if let Ok(edgefirst_json) = archive.edgefirst_json() {
            println!("  Schema:  edgefirst.json embedded");
            return Ok(SchemaV2::parse_json(&edgefirst_json)?);
        }
    }

    let mut metadata = BTreeMap::new();
    if let Some(archive) = archive.as_mut() {
        for name in archive
            .entry_names()
            .map(str::to_string)
            .collect::<Vec<_>>()
        {
            if let Ok(value) = archive.read_to_string(&name) {
                metadata.insert(name, value);
            }
        }
    }
    if let Some(description) = Metadata::from_model_bytes(model_data).description {
        metadata.insert("description".to_string(), description);
    }

    let signals = ModelSignals {
        source: ModelSource::TfLite,
        inputs: interpreter
            .inputs()?
            .iter()
            .map(tensor_info)
            .collect::<Result<Vec<_>>>()?,
        outputs: interpreter
            .outputs()?
            .iter()
            .map(tensor_info)
            .collect::<Result<Vec<_>>>()?,
        metadata,
    };
    let inferred = infer_ultralytics_schema(&signals).map_err(|e| {
        Error::unsupported(format!(
            "model carries no embedded edgefirst.json, and its schema could not be inferred: {e}"
        ))
    })?;
    println!("  Schema:  inferred ({})", inferred.description);
    Ok(inferred.schema)
}

/// HAL tensors mirroring the `TFLite` outputs, refreshed after each invoke.
struct OutputBuffers(Vec<TensorDyn>);

impl OutputBuffers {
    fn allocate(interpreter: &Interpreter<'_>) -> Result<Self> {
        let outputs = interpreter.outputs()?;
        let mut tensors = Vec::with_capacity(outputs.len());
        for t in &outputs {
            let dtype = tflite_dtype(t.tensor_type())?;
            let mut td = TensorDyn::new(&t.shape()?, dtype, Some(TensorMemory::Mem), None)?;
            // The per-scale decoder reads quantization from the tensor itself,
            // not the schema. Float tensors reject set_quantization.
            if dtype != DType::F32 {
                let qp = t.quantization_params();
                td.set_quantization(Quantization::from((qp.scale, qp.zero_point)))?;
            }
            tensors.push(td);
        }
        Ok(Self(tensors))
    }

    fn sync_from(&mut self, interpreter: &Interpreter<'_>) -> Result<()> {
        let outputs = interpreter.outputs()?;
        for (td, t) in self.0.iter_mut().zip(outputs.iter()) {
            let mut map = td.map_bytes(CpuAccess::Write)?;
            map.copy_from_slice(t.as_bytes()?);
        }
        Ok(())
    }

    fn refs(&self) -> Vec<&TensorDyn> {
        self.0.iter().collect()
    }
}

/// Where the letterboxed frame is rendered for the model.
enum ModelInput {
    /// The delegate's input DMA-BUF, imported as a HAL image: the GPU writes
    /// the buffer the NPU reads.
    DmaBuf(TensorDyn),
    /// A HAL image copied into the `TFLite` arena each frame.
    Staging(TensorDyn),
}

impl ModelInput {
    /// The image `convert` renders into.
    fn image(&mut self) -> &mut TensorDyn {
        match self {
            Self::DmaBuf(t) | Self::Staging(t) => t,
        }
    }

    /// Make the rendered frame visible to the model.
    fn submit(&self, interpreter: &mut Interpreter<'_>) -> Result<()> {
        match self {
            Self::DmaBuf(_) => {
                interpreter
                    .delegate(0)
                    .and_then(Delegate::dmabuf)
                    .ok_or_else(|| Error::unsupported("delegate DMA-BUF API disappeared"))?
                    .sync_for_device(0)?;
            }
            Self::Staging(image) => {
                let map = image.map_bytes(CpuAccess::Read)?;
                interpreter.inputs_mut()?[0]
                    .as_bytes_mut()?
                    .copy_from_slice(&map);
            }
        }
        Ok(())
    }
}

/// Bind the model input: the delegate's DMA-BUF when it exposes one, a
/// staging image otherwise.
fn bind_model_input(
    interpreter: &Interpreter<'_>,
    processor: &ImageProcessor,
    in_w: usize,
    in_h: usize,
    dtype: DType,
) -> Result<ModelInput> {
    let info = match interpreter.delegate(0).and_then(Delegate::dmabuf) {
        Some(dmabuf) => dmabuf.tensor_info(0).map_err(|e| e.to_string()),
        None => Err("no delegate DMA-BUF API".to_string()),
    };
    if let Ok(info) = info.as_ref().inspect_err(|reason| {
        println!("  Input:   delegate DMA-BUF unavailable ({reason})");
    }) {
        // SAFETY: the fd is owned by the delegate for the interpreter's
        // lifetime; PlaneDescriptor::new dups it.
        let plane = PlaneDescriptor::new(unsafe { BorrowedFd::borrow_raw(info.fd) })?
            .with_offset(info.offset);
        let image =
            processor.import_image(plane, None, in_w, in_h, PixelFormat::Rgb, dtype, None)?;
        println!("  Input:   delegate DMA-BUF (GPU \u{2192} NPU, zero-copy)");
        return Ok(ModelInput::DmaBuf(image));
    }
    let image =
        processor.create_image(in_w, in_h, PixelFormat::Rgb, dtype, None, CpuAccess::Read)?;
    println!("  Input:   staging copy (GPU \u{2192} staging \u{2192} TFLite arena)");
    Ok(ModelInput::Staging(image))
}

/// The overlay background for a camera frame. GL composites the frame's
/// native NV12 or YUYV layout directly; G2D and the CPU copy the background
/// byte-for-byte, so without GL the frame is converted to the canvas's RGBA
/// format first.
fn overlay_background<'a>(
    processor: &mut ImageProcessor,
    frame: &'a TensorDyn,
    rgba: &'a mut Option<TensorDyn>,
) -> Result<&'a TensorDyn> {
    match rgba {
        Some(background) => {
            processor.convert(frame, background, None)?;
            Ok(background)
        }
        None => Ok(frame),
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

/// Frames per timing report.
const REPORT_EVERY: u32 = 30;

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn main() -> Result<()> {
    let args = parse_args();
    let (cam_w, cam_h) = (args.width, args.height);

    // ── 1. Model, delegate, interpreter ─────────────────────────────────
    let lib = Library::new()?;
    let model_path = args
        .model
        .to_str()
        .ok_or_else(|| Error::unsupported("model path is not UTF-8"))?;
    let model = Model::from_file(&lib, model_path)?;
    println!("Model: {}", args.model.display());

    let mut builder = Interpreter::builder(&lib)?.num_threads(4);
    if let Some(ref path) = args.delegate {
        builder = builder.delegate(Delegate::load(path)?);
        println!("Delegate: {}", path.display());
    }
    let mut interpreter = builder.build(&model)?;

    let (in_h, in_w, input_dtype) = {
        let inputs = interpreter.inputs()?;
        let input = &inputs[0];
        let shape = input.shape()?;
        println!("  input[0]: {input}");
        let nchw = shape.len() == 4 && matches!(shape[1], 1 | 3) && !matches!(shape[3], 1 | 3);
        if nchw {
            return Err(Error::unsupported(
                "NCHW input; use an EdgeFirst model zoo export (NHWC int8)",
            ));
        }
        let dtype = match input.tensor_type() {
            TensorType::Int8 => DType::I8,
            TensorType::UInt8 => DType::U8,
            other => {
                return Err(Error::unsupported(format!(
                    "{other:?} input; use an EdgeFirst model zoo export (NHWC int8)"
                )))
            }
        };
        (shape[1], shape[2], dtype)
    };

    // ── 2. Decoder ──────────────────────────────────────────────────────
    let schema = load_schema(model.data(), &interpreter)?;
    let is_segmentation = schema
        .outputs
        .iter()
        .any(|o| matches!(o.type_, Some(LogicalType::Protos)));
    let decoder: Decoder = DecoderBuilder::new()
        .with_schema(schema)
        .with_score_threshold(args.threshold)
        .with_iou_threshold(args.iou)
        .build()?;
    println!(
        "  Mode:    {}",
        if is_segmentation {
            "segmentation"
        } else {
            "detection"
        }
    );

    // ── 3. HAL processor, model I/O and display canvas ──────────────────
    // Preprocessing and overlay drawing use separate processors. Sharing one
    // corrupts the next frame's model input on i.MX95 (Mali): state left by
    // draw_masks garbles the following convert into the packed-RGB input
    // buffer, letterbox padding included, and detection quietly degrades.
    let mut processor = ImageProcessor::new()?;
    let mut overlay_processor = ImageProcessor::new()?;
    let mut model_input = bind_model_input(&interpreter, &processor, in_w, in_h, input_dtype)?;
    let mut output_bufs = OutputBuffers::allocate(&interpreter)?;
    let letterbox = ConvertOptions::new().letterbox([114, 114, 114, 255]);

    // The canvas is what the compositor scans, so it has to be a DMA-BUF;
    // `ReadWrite` covers the CPU renderer used when GL is unavailable.
    let mut canvas = processor
        .create_image(
            cam_w,
            cam_h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        )
        .map_err(|e| Error::display(format!("cannot allocate a DMA-BUF canvas: {e}")))?;
    let canvas_fd = canvas.clone_fd()?;
    let canvas_stride = canvas
        .as_u8()
        .and_then(edgefirst_tensor::Tensor::effective_row_stride)
        .unwrap_or(cam_w * 4);
    let canvas_stride =
        u32::try_from(canvas_stride).map_err(|_| Error::display("canvas stride out of range"))?;
    let mut rgba_background = if processor.opengl.is_none() {
        Some(processor.create_image(
            cam_w,
            cam_h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            CpuAccess::ReadWrite,
        )?)
    } else {
        None
    };

    // ── 4. Camera ───────────────────────────────────────────────────────
    let manager = CameraManager::new()?;
    let cameras = manager.cameras();
    let camera = match args.camera_name {
        Some(ref name) => cameras.iter().find(|c| c.id() == name.as_str()),
        None => cameras.get(0),
    };
    let Some(camera) = camera else {
        let available: Vec<String> = cameras.iter().map(|c| c.id().to_string()).collect();
        return Err(Error::camera(format!(
            "camera not found; available: [{}]",
            available.join(", ")
        )));
    };
    println!("Camera: {}", camera.id());
    let mut camera = camera.acquire()?;

    let mut config = camera
        .generate_configuration(&[StreamRole::VideoRecording])
        .ok_or_else(|| Error::camera("cannot generate a configuration"))?;
    {
        let mut stream_cfg = config.get_mut(0).expect("one stream requested");
        stream_cfg.set_pixel_format(libcamera_format(args.format));
        stream_cfg.set_size(libcamera::geometry::Size {
            width: u32::try_from(cam_w).map_err(|_| Error::camera("width out of range"))?,
            height: u32::try_from(cam_h).map_err(|_| Error::camera("height out of range"))?,
        });
    }
    match config.validate() {
        CameraConfigurationStatus::Valid => {}
        CameraConfigurationStatus::Adjusted => {
            // The canvas, window and letterbox are all sized from the request,
            // so an adjusted size would be drawn misregistered.
            let cfg = config.get(0).expect("one stream requested");
            let size = cfg.get_size();
            return Err(Error::camera(format!(
                "camera cannot capture {cam_w}x{cam_h} {:?}; it offered {}x{} {:?}",
                args.format,
                size.width,
                size.height,
                cfg.get_pixel_format()
            )));
        }
        CameraConfigurationStatus::Invalid => {
            return Err(Error::camera(format!(
                "camera cannot produce {:?}; on i.MX95 set \
                 LIBCAMERA_PIPELINES_MATCH_LIST=nxp/neo to capture through the ISP",
                args.format
            )))
        }
    }
    camera.configure(&mut config)?;
    let (stream, stride, color_space) = {
        let cfg = config.get(0).expect("one stream requested");
        let stream = cfg
            .stream()
            .ok_or_else(|| Error::camera("configured stream missing"))?;
        (stream, cfg.get_stride() as usize, cfg.get_color_space())
    };

    let mut allocator = FrameBufferAllocator::new(&camera);
    let buffers = allocator.alloc(&stream)?;
    let n_buffers = buffers.len();
    println!(
        "  Capture: {cam_w}x{cam_h} {:?}, stride {stride}, {n_buffers} buffers, {color_space:?}",
        args.format
    );
    let mut requests = Vec::with_capacity(n_buffers);
    for (i, buffer) in buffers.into_iter().enumerate() {
        let mut request = camera
            .create_request(Some(i as u64))
            .ok_or_else(|| Error::camera("cannot create a request"))?;
        request.add_buffer(&stream, buffer)?;
        requests.push(request);
    }
    let mut frames = FrameCache {
        entries: (0..n_buffers).map(|_| None).collect(),
        width: cam_w,
        height: cam_h,
        stride,
        format: args.format,
        colorimetry: color_space.map(colorimetry),
    };

    let (tx, rx) = mpsc::channel();
    camera.on_request_completed(move |request| {
        let _ = tx.send(request);
    });

    // ── 5. Display ──────────────────────────────────────────────────────
    let mut display = WaylandDisplay::new(cam_w, cam_h, "EdgeFirst TFLite YOLOv8 Live")?;

    // ── 6. Live loop ────────────────────────────────────────────────────
    camera.start(None)?;
    for request in requests {
        camera.queue_request(request).map_err(|(_, e)| e)?;
    }
    println!("Running -- close the window or press Ctrl+C to stop\n");

    let t_start = Instant::now();
    let mut frame_count: u64 = 0;
    let mut window = [Duration::ZERO; 6];
    let mut dropped: u64 = 0;

    while display.is_open()? {
        let t0 = Instant::now();

        // Take the newest completed frame, returning any older ones to the
        // camera unprocessed.
        let mut request = rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| Error::camera("no frame within 5 s"))?;
        while let Ok(newer) = rx.try_recv() {
            request.reuse(ReuseFlag::REUSE_BUFFERS);
            camera.queue_request(request).map_err(|(_, e)| e)?;
            request = newer;
            dropped += 1;
        }
        let t1 = Instant::now();

        let index = usize::try_from(request.cookie())
            .map_err(|_| Error::camera("request cookie out of range"))?;
        let buffer: &FrameBuffer = request
            .buffer(&stream)
            .ok_or_else(|| Error::camera("completed request has no buffer"))?;
        let frame = frames.get_or_import(index, &processor, buffer)?;
        // The cached tensor holds its own fd, so the buffer can go back to the
        // camera now; keeping requests queued avoids starving the ISP.
        request.reuse(ReuseFlag::REUSE_BUFFERS);
        camera.queue_request(request).map_err(|(_, e)| e)?;
        let t2 = Instant::now();

        processor.convert(frame, model_input.image(), Some(letterbox))?;
        model_input.submit(&mut interpreter)?;
        let t3 = Instant::now();

        interpreter.invoke()?;
        output_bufs.sync_from(&interpreter)?;
        let t4 = Instant::now();

        let background = overlay_background(&mut overlay_processor, frame, &mut rgba_background)?;
        let overlay = MaskOverlay::default()
            .with_background(background)
            .with_convert_options(&letterbox, cam_w, cam_h, in_w, in_h)
            .with_color_mode(args.color_mode);
        let detections =
            overlay_processor.draw_masks(&decoder, &output_bufs.refs(), &mut canvas, overlay)?;
        let t5 = Instant::now();

        display.present(std::os::fd::AsFd::as_fd(&canvas_fd), canvas_stride)?;
        let t6 = Instant::now();

        for (acc, d) in
            window
                .iter_mut()
                .zip([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5])
        {
            *acc += d;
        }
        frame_count += 1;
        if frame_count.is_multiple_of(u64::from(REPORT_EVERY)) {
            let fps = frame_count as f64 / t_start.elapsed().as_secs_f64();
            let [pull, import, preprocess, infer, draw, present] =
                window.map(|d| (d / REPORT_EVERY).as_secs_f64() * 1000.0);
            eprint!(
                "\r  FPS: {fps:5.1}  pull:{pull:5.1} imp:{import:4.1} pre:{preprocess:4.1} \
                 inf:{infer:5.1} draw:{draw:5.1} disp:{present:4.1} ms  \
                 drop:{dropped} det:{:<3}",
                detections.len()
            );
            window = [Duration::ZERO; 6];
        }
    }

    eprintln!();
    camera.stop()?;
    if frame_count > 0 {
        let elapsed = t_start.elapsed().as_secs_f64();
        println!(
            "Processed {frame_count} frames in {elapsed:.1}s ({:.1} FPS average)",
            frame_count as f64 / elapsed
        );
    }
    Ok(())
}
