// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! # Live `YOLOv8` Camera Inference with `edgefirst-tflite`
//!
//! Captures NV12 or YUYV frames from a camera through libcamera, runs `YOLOv8`
//! detection or instance segmentation with `edgefirst-tflite`, and shows the
//! result in a native Wayland window. Overlays are composited onto the camera
//! frame by the `EdgeFirst` HAL (`edgefirst-image`).
//!
//! ## Pipeline
//!
//! ```text
//! libcamera (NV12 | YUYV DMA-BUF)
//!   -> HAL import            (cached per camera buffer)
//!   -> HAL convert           (letterbox into the delegate's input DMA-BUF)
//!   -> TFLite invoke         (Neutron / VxDelegate NPU, or CPU)
//!   -> HAL decode            (boxes, and mask coefficients for segmentation)
//!   -> HAL draw              (composite onto an RGBA DMA-BUF canvas)
//!   -> Wayland               (canvas DMA-BUF -> wl_buffer -> compositor)
//! ```
//!
//! The stages run on separate threads so consecutive frames overlap (see
//! [`pipeline`]); the status line reports each stage's time per frame.
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
//!             [--nms auto|agnostic|aware] [--fullscreen]
//!             [--class-colors RRGGBB,...] [--line-width N] [--slots N]
//!             [--exposure US --gain N] [--auto-exposure]
//! ```
//!
//! `--nms auto` follows the policy in the model's `edgefirst.json`; `agnostic`
//! also suppresses overlapping boxes of different classes, which removes the
//! double-labelled objects a model unsure between two classes produces.
//!
//! `--fullscreen` asks the compositor for a fullscreen window and renders at
//! the output's resolution; the camera frame is scaled to fill it, so capture
//! at the display's aspect ratio to keep the picture undistorted.
//!
//! `--class-colors` sets the colour of each class, in label order, as hex
//! `RRGGBB` values. `--line-width` sets the box outline width in screen
//! pixels (default 3).
//!
//! `--slots` sets the number of interpreter instances the letterbox and
//! inference stages alternate between (default 2; 1 runs them back to back).
//!
//! `--exposure` (microseconds) and `--gain` (analogue gain), given together,
//! turn the camera's automatic exposure off and hold the sensor at fixed
//! values, which stops the frame-to-frame brightness changes the AGC makes.
//! Mains lighting flickers at twice the mains frequency, so an exposure of
//! whole flicker periods (8333 or 16666 µs for 60 Hz, 10000 µs for 50 Hz)
//! collects the same light in every frame. `--auto-exposure` cancels both, so
//! it can undo them when it follows them on the command line. A warning names
//! any control the camera lacks, and the second status line's frames are
//! checked against the requested values. Each status line reports the range
//! of exposure and gain the frames carried since the previous one.
//!
//! Some pipelines report the requested values in the frames rather than the
//! sensor's, and some sensors run an automatic exposure of their own that
//! libcamera leaves on, so steady values on the status line with a
//! flickering picture point at the sensor. The AR0234 on Phytec i.MX95 boards
//! is one; turn its own exposure control off before starting:
//!
//! ```sh
//! v4l2-ctl -d /dev/cam-csi1-port0 -c auto_exposure=1,autogain_analogue=0
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
mod pipeline;

use std::collections::{BTreeMap, HashMap};
use std::io::IsTerminal as _;
use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd, OwnedFd};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use edgefirst_decoder::{
    infer_ultralytics_schema,
    schema::{DType as SchemaDType, LogicalType, Quantization as SchemaQuantization, SchemaV2},
    Decoder, DecoderBuilder, ModelSignals, ModelSource, Nms, TensorInfo,
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
    control::{ControlEntry as _, ControlError, ControlList, MergePolicy},
    controls::{AeEnable, AnalogueGain, AnalogueGainMode, ExposureTime, ExposureTimeMode},
    framebuffer::AsFrameBuffer as _,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    request::ReuseFlag,
    stream::StreamRole,
    utils::UniquePtr,
};
use wayland_client::protocol::{wl_buffer, wl_callback, wl_compositor, wl_registry, wl_surface};
use wayland_client::{delegate_noop, Connection, Dispatch, EventQueue, QueueHandle};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1, zwp_linux_dmabuf_v1,
};
use wayland_protocols::wp::viewporter::client::{wp_viewport, wp_viewporter};
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
    nms: Nms,
    fullscreen: bool,
    class_colors: Vec<[u8; 4]>,
    line_width: Option<usize>,
    slots: usize,
    exposure: Option<i32>,
    gain: Option<f32>,
}

fn usage(program: &str) -> ! {
    eprintln!(
        "Usage: {program} <model.tflite> [--delegate <path>] [--threshold N] [--iou N] \
         [--width N] [--height N] [--camera-name ID] [--format nv12|yuyv] \
         [--color-mode class|instance|track] [--nms auto|agnostic|aware] [--fullscreen] \
         [--class-colors RRGGBB,...] [--line-width N] [--slots N] \
         [--exposure US --gain N] [--auto-exposure]"
    );
    std::process::exit(1);
}

#[allow(clippy::too_many_lines)]
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
        nms: Nms::Auto,
        fullscreen: false,
        class_colors: Vec::new(),
        line_width: None,
        slots: 2,
        exposure: None,
        gain: None,
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
            "--nms" => {
                parsed.nms = match next_arg!("--nms") {
                    "auto" => Nms::Auto,
                    "agnostic" => Nms::ClassAgnostic,
                    "aware" => Nms::ClassAware,
                    other => {
                        eprintln!("unknown --nms {other} (expected auto, agnostic or aware)");
                        std::process::exit(1);
                    }
                };
            }
            "--fullscreen" => parsed.fullscreen = true,
            "--class-colors" => {
                parsed.class_colors = parse_class_colors(next_arg!("--class-colors"));
            }
            "--line-width" => {
                let width: usize = parse_value!("--line-width");
                if width == 0 {
                    eprintln!("--line-width must be at least 1");
                    std::process::exit(1);
                }
                parsed.line_width = Some(width);
            }
            "--slots" => parsed.slots = parse_value!("--slots"),
            "--exposure" => parsed.exposure = Some(parse_value!("--exposure")),
            "--gain" => parsed.gain = Some(parse_value!("--gain")),
            "--auto-exposure" => {
                parsed.exposure = None;
                parsed.gain = None;
            }
            "-h" | "--help" => usage(&args[0]),
            other => eprintln!("WARNING: unknown argument {other}, ignored"),
        }
        i += 1;
    }
    if parsed.exposure.is_some() != parsed.gain.is_some() {
        eprintln!("--exposure and --gain must be given together");
        std::process::exit(1);
    }
    parsed
}

/// Palette entries the HAL accepts in `set_class_colors`.
const MAX_CLASS_COLORS: usize = 20;

/// Parses an opaque `RRGGBB` colour, with or without a leading `#`.
fn parse_color(text: &str) -> Option<[u8; 4]> {
    let hex = text.trim().trim_start_matches('#');
    if hex.len() != 6 {
        return None;
    }
    let [_, red, green, blue] = u32::from_str_radix(hex, 16).ok()?.to_be_bytes();
    Some([red, green, blue, 255])
}

/// Parses `--class-colors`: comma-separated colours, at most
/// [`MAX_CLASS_COLORS`] of them.
fn parse_class_colors(value: &str) -> Vec<[u8; 4]> {
    value
        .split(',')
        .map(parse_color)
        .collect::<Option<Vec<_>>>()
        .filter(|colors| colors.len() <= MAX_CLASS_COLORS)
        .unwrap_or_else(|| {
            eprintln!(
                "invalid --class-colors {value} (expected up to {MAX_CLASS_COLORS} \
                 comma-separated RRGGBB values)"
            );
            std::process::exit(1);
        })
}

// ── Wayland display (direct DMA-BUF submission, no EGL/GL) ───────────────────

/// DRM fourcc `ABGR8888`: R, G, B, A bytes in memory order, which is the
/// layout of the HAL's [`PixelFormat::Rgba`].
const DRM_FORMAT_ABGR8888: u32 = u32::from_le_bytes(*b"AB24");

/// Event-loop state for the Wayland connection.
#[derive(Debug, Default)]
struct DisplayState {
    compositor: Option<wl_compositor::WlCompositor>,
    viewporter: Option<wp_viewporter::WpViewporter>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    dmabuf: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    surface: Option<wl_surface::WlSurface>,
    configured: bool,
    /// Size the compositor asked for in the last `xdg_toplevel.configure`;
    /// zero when it left the choice to the client.
    configured_size: (i32, i32),
    closed: bool,
    /// `true` when the compositor is ready for the next frame. Cleared on
    /// submit and set again by the `wl_surface.frame` callback.
    frame_done: bool,
    /// One `wl_buffer` per DMA-BUF fd, imported on first submit.
    buffers: HashMap<i32, wl_buffer::WlBuffer>,
}

/// Buffers of another size than the window, which the compositor scales.
struct Scaled {
    size: (i32, i32),
    _viewport: wp_viewport::WpViewport,
}

/// An XDG toplevel window that presents DMA-BUF RGBA buffers.
struct WaylandDisplay {
    conn: Connection,
    queue: EventQueue<DisplayState>,
    state: DisplayState,
    width: i32,
    height: i32,
    scaled: Option<Scaled>,
}

impl WaylandDisplay {
    /// Connect via `$WAYLAND_DISPLAY`, bind the globals and wait for the
    /// window's first configure.
    ///
    /// With `fullscreen`, the window takes the size the compositor configures
    /// (the output's resolution) rather than `width` x `height`.
    fn new(width: usize, height: usize, title: &str, fullscreen: bool) -> Result<Self> {
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
        if fullscreen {
            toplevel.set_fullscreen(None);
        }
        surface.commit();
        state.surface = Some(surface);

        while !state.configured {
            queue
                .blocking_dispatch(&mut state)
                .map_err(|e| Error::display(format!("dispatch: {e}")))?;
        }

        let (width, height) = match state.configured_size {
            (w, h) if fullscreen && w > 0 && h > 0 => (w, h),
            _ => (width, height),
        };
        Ok(Self {
            conn,
            queue,
            state,
            width,
            height,
            scaled: None,
        })
    }

    /// Present `width` x `height` buffers scaled to the window, returning
    /// `false` when the compositor cannot scale them.
    fn scale_buffers(&mut self, width: usize, height: usize) -> Result<bool> {
        let Some(viewporter) = self.state.viewporter.as_ref() else {
            return Ok(false);
        };
        let width = i32::try_from(width).map_err(|_| Error::display("width out of range"))?;
        let height = i32::try_from(height).map_err(|_| Error::display("height out of range"))?;
        let surface = self.state.surface.as_ref().expect("created in new()");
        let viewport = viewporter.get_viewport(surface, &self.queue.handle(), ());
        viewport.set_destination(self.width, self.height);
        self.scaled = Some(Scaled {
            size: (width, height),
            _viewport: viewport,
        });
        Ok(true)
    }

    /// The window size in pixels, which is the size buffers must be.
    fn size(&self) -> (usize, usize) {
        (
            self.width.unsigned_abs() as usize,
            self.height.unsigned_abs() as usize,
        )
    }

    /// Present the RGBA DMA-BUF `fd`, whose rows are `stride` bytes apart,
    /// returning whether it was handed to the compositor.
    ///
    /// Paced by `wl_surface.frame`: while the compositor has not consumed the
    /// previous frame this is a no-op, so a fast pipeline drops frames at the
    /// display rather than queueing them. A window that has been closed
    /// presents nothing; [`Self::is_open`] reports it.
    fn present(&mut self, fd: BorrowedFd<'_>, stride: u32) -> Result<bool> {
        self.dispatch_pending()?;
        if self.state.closed || !self.state.frame_done {
            return Ok(false);
        }

        let qh = self.queue.handle();
        let (w, h) = self
            .scaled
            .as_ref()
            .map_or((self.width, self.height), |scaled| scaled.size);
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
                "wp_viewporter" => {
                    state.viewporter = Some(registry.bind(name, 1, qh, ()));
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
        match event {
            xdg_toplevel::Event::Close => state.closed = true,
            xdg_toplevel::Event::Configure { width, height, .. } => {
                state.configured_size = (width, height);
            }
            _ => {}
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
delegate_noop!(DisplayState: ignore wp_viewporter::WpViewporter);
delegate_noop!(DisplayState: ignore wp_viewport::WpViewport);
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
/// cookie. A tensor is taken out while its frame is in the pipeline and put
/// back once the frame has been displayed.
struct FrameCache {
    entries: Vec<Option<TensorDyn>>,
    width: usize,
    height: usize,
    stride: usize,
    format: PixelFormat,
    colorimetry: Option<Colorimetry>,
}

impl FrameCache {
    fn take(
        &mut self,
        index: usize,
        processor: &ImageProcessor,
        framebuffer: &FrameBuffer,
    ) -> Result<TensorDyn> {
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
        Ok(slot.take().expect("imported above"))
    }

    fn put(&mut self, index: usize, image: TensorDyn) {
        if let Some(slot) = self.entries.get_mut(index) {
            *slot = Some(image);
        }
    }
}

/// Camera controls that hold the sensor at a fixed exposure and gain, for
/// `--exposure` and `--gain`, or `None` to leave the AGC in charge.
///
/// Controls the camera does not offer are left out with a warning naming the
/// option they leave without effect, and the range of each one it does offer
/// is printed to help choose values. A gain outside the camera's range, which
/// it clamps, is warned about too. The exposure range is not checked: some
/// sensors advertise a maximum below exposures they apply.
fn manual_exposure(
    camera: &libcamera::camera::ActiveCamera<'_>,
    exposure: Option<i32>,
    gain: Option<f32>,
) -> Result<Option<UniquePtr<ControlList>>> {
    if exposure.is_none() && gain.is_none() {
        return Ok(None);
    }
    let offered = |id: u32, name: &str, effect: &str| {
        if let Ok(info) = camera.controls().find(id) {
            println!("  Control: {name} {:?}..{:?}", info.min(), info.max());
            true
        } else {
            eprintln!("WARNING: the camera has no {name} control, so {effect}");
            false
        }
    };
    // Cameras without the mode controls select manual values through AeEnable.
    let has = |id: u32| camera.controls().find(id).is_ok();
    let control_error = |e: ControlError| Error::camera(format!("exposure controls: {e}"));
    let mut controls = ControlList::new();
    if offered(
        AeEnable::ID,
        "AeEnable",
        "automatic exposure may override --exposure and --gain",
    ) {
        controls.set(AeEnable(false)).map_err(control_error)?;
    }
    if let Some(us) = exposure {
        if has(ExposureTimeMode::ID) {
            controls
                .set(ExposureTimeMode::Manual)
                .map_err(control_error)?;
        }
        if offered(ExposureTime::ID, "ExposureTime", "--exposure has no effect") {
            controls.set(ExposureTime(us)).map_err(control_error)?;
        }
    }
    if let Some(gain) = gain {
        if has(AnalogueGainMode::ID) {
            controls
                .set(AnalogueGainMode::Manual)
                .map_err(control_error)?;
        }
        if offered(AnalogueGain::ID, "AnalogueGain", "--gain has no effect") {
            controls.set(AnalogueGain(gain)).map_err(control_error)?;
        }
        let range = camera
            .controls()
            .find(AnalogueGain::ID)
            .ok()
            .and_then(|info| {
                Some((
                    f32::try_from(info.min()).ok()?,
                    f32::try_from(info.max()).ok()?,
                ))
            });
        if let Some((lo, hi)) = range.filter(|&(lo, hi)| !(lo..=hi).contains(&gain)) {
            eprintln!(
                "WARNING: --gain {gain} is outside the camera's range of {lo} to {hi}, \
                 so the camera clamps it"
            );
        }
    }
    Ok(Some(controls))
}

/// The range of exposure time and analogue gain the sensor used over a
/// status-line window, from each frame's metadata.
#[derive(Debug, Default)]
struct ExposureRange {
    exposure: Option<(i32, i32)>,
    gain: Option<(f32, f32)>,
}

impl ExposureRange {
    fn add(&mut self, metadata: &ControlList) {
        if let Ok(ExposureTime(us)) = metadata.get::<ExposureTime>() {
            self.exposure = Some(
                self.exposure
                    .map_or((us, us), |(lo, hi)| (lo.min(us), hi.max(us))),
            );
        }
        if let Ok(AnalogueGain(g)) = metadata.get::<AnalogueGain>() {
            self.gain = Some(self.gain.map_or((g, g), |(lo, hi)| (lo.min(g), hi.max(g))));
        }
    }
}

impl std::fmt::Display for ExposureRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.exposure {
            Some((lo, hi)) if lo == hi => write!(f, "exp:{lo}")?,
            Some((lo, hi)) => write!(f, "exp:{lo}-{hi}")?,
            None => write!(f, "exp:--")?,
        }
        match self.gain {
            Some((lo, hi)) if (hi - lo).abs() < 0.005 => write!(f, " gain:{lo:.2}"),
            Some((lo, hi)) => write!(f, " gain:{lo:.2}-{hi:.2}"),
            None => write!(f, " gain:--"),
        }
    }
}

/// Warn when the frames show an exposure or gain other than the one asked
/// for, which means the camera ignored or clamped the setting; print the
/// values the frames report when they match. Some pipelines report the
/// requested values rather than the sensor's, so a match is not proof.
fn check_exposure(requested: (Option<i32>, Option<f32>), used: &ExposureRange) {
    /// Relative difference allowed for the sensor's step size: exposure is set
    /// in whole lines and gain in register steps.
    const TOLERANCE: f64 = 0.03;
    let off = |want: f64, (lo, hi): (f64, f64)| {
        (lo - want).abs() > want * TOLERANCE || (hi - want).abs() > want * TOLERANCE
    };
    let span = |lo: String, hi: String| if lo == hi { lo } else { format!("{lo}-{hi}") };
    let mut ok = true;
    if let Some(want) = requested.0 {
        match used.exposure {
            None => {
                eprintln!("WARNING: the camera reports no ExposureTime, so --exposure {want} cannot be confirmed");
                ok = false;
            }
            Some((lo, hi)) if off(f64::from(want), (f64::from(lo), f64::from(hi))) => {
                eprintln!("WARNING: the camera ran at {} µs, not --exposure {want}; it ignored or clamped the setting",
                    span(lo.to_string(), hi.to_string()));
                ok = false;
            }
            Some(_) => {}
        }
    }
    if let Some(want) = requested.1 {
        match used.gain {
            None => {
                eprintln!("WARNING: the camera reports no AnalogueGain, so --gain {want} cannot be confirmed");
                ok = false;
            }
            Some((lo, hi)) if off(f64::from(want), (f64::from(lo), f64::from(hi))) => {
                eprintln!("WARNING: the camera ran at gain {}, not --gain {want}; it ignored or clamped the setting",
                    span(format!("{lo:.2}"), format!("{hi:.2}")));
                ok = false;
            }
            Some(_) => {}
        }
    }
    if ok && (requested.0.is_some() || requested.1.is_some()) {
        println!("  Exposure: frames report {used}");
    }
}

/// Return a request's buffer to the camera for another capture.
fn requeue(
    camera: &libcamera::camera::ActiveCamera<'_>,
    mut request: libcamera::request::Request,
) -> Result<()> {
    request.reuse(ReuseFlag::REUSE_BUFFERS);
    camera.queue_request(request).map_err(|(_, e)| e)?;
    Ok(())
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

/// An RGBA DMA-BUF of `width` x `height` for the compositor, with its fd
/// and row stride in bytes.
///
/// With GL only the GPU and the compositor touch it, and declaring no CPU
/// access lets the HAL take it from uncached memory, which devices map
/// without maintaining the CPU cache over it. The CPU renderer used without
/// GL needs to read and write it.
fn display_buffer(
    processor: &ImageProcessor,
    width: usize,
    height: usize,
) -> Result<(TensorDyn, OwnedFd, usize)> {
    let access = if processor.opengl.is_some() {
        CpuAccess::None
    } else {
        CpuAccess::ReadWrite
    };
    let image = processor
        .create_image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            access,
        )
        .map_err(|e| Error::display(format!("cannot allocate a DMA-BUF canvas: {e}")))?;
    let stride = image
        .as_u8()
        .and_then(edgefirst_tensor::Tensor::effective_row_stride)
        .unwrap_or(width * 4);
    let fd = image.clone_fd()?;
    Ok((image, fd, stride))
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

    if args.slots == 0 {
        return Err(Error::unsupported("--slots must be at least 1"));
    }
    // One interpreter, each with its own delegate instance, per slot.
    let mut interpreters = Vec::with_capacity(args.slots);
    for _ in 0..args.slots {
        let mut builder = Interpreter::builder(&lib)?.num_threads(4);
        if let Some(ref path) = args.delegate {
            builder = builder.delegate(Delegate::load(path)?);
        }
        interpreters.push(builder.build(&model)?);
    }
    if let Some(ref path) = args.delegate {
        println!("Delegate: {} ({} slots)", path.display(), args.slots);
    }
    let interpreter = &interpreters[0];

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
    let schema = load_schema(model.data(), interpreter)?;
    let is_segmentation = schema
        .outputs
        .iter()
        .any(|o| matches!(o.type_, Some(LogicalType::Protos)));
    let decoder: Decoder = DecoderBuilder::new()
        .with_schema(schema)
        .with_score_threshold(args.threshold)
        .with_iou_threshold(args.iou)
        .with_nms(Some(args.nms))
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
    let processor = ImageProcessor::new()?;
    let mut overlay_processor = ImageProcessor::new()?;
    if !args.class_colors.is_empty() {
        overlay_processor.set_class_colors(&args.class_colors)?;
    }
    let mut slots = Vec::with_capacity(interpreters.len());
    for (id, interpreter) in interpreters.iter().enumerate() {
        let input = bind_model_input(interpreter, &processor, in_w, in_h, input_dtype)?;
        slots.push(pipeline::Slot::new(id, input));
    }
    // One output buffer each for the npu thread, the hand-off to decode and
    // the decode thread, so neither waits on the other for a buffer.
    let output_pool = (0..3)
        .map(|_| OutputBuffers::allocate(&interpreters[0]))
        .collect::<Result<Vec<_>>>()?;
    let letterbox = ConvertOptions::new().letterbox([114, 114, 114, 255]);

    // The window, and so the canvas, is the capture size unless the
    // compositor sized it for fullscreen.
    let mut display = WaylandDisplay::new(
        cam_w,
        cam_h,
        "EdgeFirst TFLite YOLOv8 Live",
        args.fullscreen,
    )?;
    let (disp_w, disp_h) = display.size();
    println!("Display: {disp_w}x{disp_h}");
    if (disp_w * cam_h).abs_diff(disp_h * cam_w) * 100 > disp_w * cam_h {
        eprintln!(
            "  warning: capture {cam_w}x{cam_h} and display {disp_w}x{disp_h} differ in \
             aspect ratio; the picture is stretched to fill the screen"
        );
    }
    // With GL, a canvas the size of the camera frame lets the HAL draw over
    // the frame itself, and the compositor scales the canvas to the window as
    // it composites; otherwise each frame is first converted to the window's
    // size.
    let (canvas_w, canvas_h) = if (disp_w, disp_h) != (cam_w, cam_h)
        && overlay_processor.opengl.is_some()
        && display.scale_buffers(cam_w, cam_h)?
    {
        (cam_w, cam_h)
    } else {
        (disp_w, disp_h)
    };
    if (canvas_w, canvas_h) != (disp_w, disp_h) {
        println!("Canvas:  {canvas_w}x{canvas_h}, scaled by the compositor");
    }
    if let Some(width) = args.line_width {
        // The outline is drawn in canvas pixels; keep its width on screen.
        let width = u32::try_from((width * canvas_w).div_ceil(disp_w).max(1))
            .map_err(|_| Error::unsupported("--line-width out of range"))?;
        overlay_processor.set_box_line_width(width)?;
    }

    // The canvas is what the compositor scans, so it has to be a DMA-BUF.
    // Two canvases: each frame is drawn into the one the compositor is not
    // showing, so it never displays a frame mid-draw.
    let [(canvas_a, fd_a, canvas_stride), (canvas_b, fd_b, _)] = [
        display_buffer(&processor, canvas_w, canvas_h)?,
        display_buffer(&processor, canvas_w, canvas_h)?,
    ];
    let mut canvases = [canvas_a, canvas_b];
    let canvas_fds = [fd_a, fd_b];
    let mut back = 0;
    let canvas_stride =
        u32::try_from(canvas_stride).map_err(|_| Error::display("canvas stride out of range"))?;
    // Drawing composites onto a background of the canvas's size and, without
    // GL, of its RGBA format, so either difference needs a converted copy of
    // each frame.
    let mut rgba_background =
        if processor.opengl.is_none() || (canvas_w, canvas_h) != (cam_w, cam_h) {
            Some(processor.create_image(
                canvas_w,
                canvas_h,
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

    let (events_tx, events) = mpsc::channel();
    let camera_events = events_tx.clone();
    camera.on_request_completed(move |request| {
        let _ = camera_events.send(pipeline::Event::Frame(request));
    });

    // ── 5. Live loop ────────────────────────────────────────────────────
    // Applied at start and on the first round of requests; the pipeline
    // keeps manual exposure and gain until a request changes them.
    let manual = manual_exposure(&camera, args.exposure, args.gain)?;
    camera.start(manual.as_deref()).map_err(|e| {
        if manual.is_some() {
            Error::camera(format!("camera rejected --exposure/--gain controls: {e}"))
        } else {
            e.into()
        }
    })?;
    for mut request in requests {
        if let Some(manual) = manual.as_deref() {
            request
                .controls_mut()
                .merge(manual, MergePolicy::OverwriteExisting);
        }
        camera.queue_request(request).map_err(|(_, e)| e)?;
    }
    println!("Running -- close the window or press Ctrl+C to stop\n");

    // Pipeline channels. Each is bounded by what it holds: one frame
    // waiting to be letterboxed, the slots, one frame between npu and
    // decode, and the output-buffer pool.
    let (frame_tx, frame_rx) = mpsc::sync_channel::<pipeline::Frame>(1);
    let (free_tx, free_rx) = mpsc::channel();
    for slot in slots {
        free_tx.send(slot).expect("receiver is alive");
    }
    let (prepared_tx, prepared_rx) = mpsc::sync_channel(1);
    let (pool_tx, pool_rx) = mpsc::channel();
    for buffers in output_pool {
        pool_tx.send(buffers).expect("receiver is alive");
    }
    let (inferred_tx, inferred_rx) = mpsc::sync_channel(1);

    let t_start = Instant::now();
    let mut window_start = t_start;
    let mut frame_count: u64 = 0;
    // pre, infer, copy, decode, background, draw, present, latency
    let mut window = [Duration::ZERO; 8];
    let mut dropped: u64 = 0;
    let mut exposure_range = ExposureRange::default();
    let mut status_lines: u32 = 0;
    let stderr_is_terminal = std::io::stderr().is_terminal();
    let mut last_detections = 0;
    // Requests whose frames are in the pipeline, by buffer index.
    let mut in_flight: HashMap<usize, libcamera::request::Request> = HashMap::new();

    std::thread::scope(|scope| -> Result<()> {
        let decoder = &decoder;
        let stages = [
            scope.spawn(move || {
                pipeline::preprocess(processor, letterbox, &frame_rx, &free_rx, &prepared_tx)
            }),
            scope.spawn(move || {
                pipeline::infer(interpreters, &prepared_rx, &free_tx, &pool_rx, &inferred_tx)
            }),
            scope.spawn(move || {
                pipeline::decode(decoder, is_segmentation, &inferred_rx, &pool_tx, &events_tx)
            }),
        ];

        let looped = (|| -> Result<()> {
            while display.is_open()? {
                // Everything that arrived while the last frame was drawn:
                // decoded frames are drawn in order, and of the camera
                // frames only the newest enters the pipeline.
                let first = events
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(|_| Error::camera("no frame within 5 s"))?;
                let mut newest = None;
                for event in std::iter::once(first).chain(events.try_iter()) {
                    match event {
                        pipeline::Event::Frame(request) => {
                            exposure_range.add(request.metadata());
                            if let Some(older) = newest.replace(request) {
                                requeue(&camera, older)?;
                                dropped += 1;
                            }
                        }
                        pipeline::Event::Decoded(result) => {
                            let pipeline::Decoded {
                                frame,
                                detections,
                                masks,
                                protos,
                                times,
                            } = *result;
                            let t0 = Instant::now();
                            let background = overlay_background(
                                &mut overlay_processor,
                                &frame.image,
                                &mut rgba_background,
                            )?;
                            let t1 = Instant::now();
                            let overlay = MaskOverlay::default()
                                .with_background(background)
                                .with_convert_options(&letterbox, cam_w, cam_h, in_w, in_h)
                                .with_color_mode(args.color_mode);
                            if let Some(ref protos) = protos {
                                overlay_processor.draw_proto_masks(
                                    &mut canvases[back],
                                    &detections,
                                    protos,
                                    overlay,
                                )?;
                            } else {
                                overlay_processor.draw_decoded_masks(
                                    &mut canvases[back],
                                    &detections,
                                    &masks,
                                    overlay,
                                )?;
                            }
                            let t2 = Instant::now();
                            if display.present(canvas_fds[back].as_fd(), canvas_stride)? {
                                back = 1 - back;
                            }
                            let t3 = Instant::now();

                            // The frame has been drawn, so its buffer can be
                            // captured into again.
                            let latency = frame.entered.elapsed();
                            frames.put(frame.index, frame.image);
                            if let Some(request) = in_flight.remove(&frame.index) {
                                requeue(&camera, request)?;
                            }

                            for (acc, d) in window.iter_mut().zip([
                                times.preprocess,
                                times.infer,
                                times.copy,
                                times.decode,
                                t1 - t0,
                                t2 - t1,
                                t3 - t2,
                                latency,
                            ]) {
                                *acc += d;
                            }
                            last_detections = detections.len();
                            frame_count += 1;
                            if frame_count.is_multiple_of(u64::from(REPORT_EVERY)) {
                                let fps =
                                    f64::from(REPORT_EVERY) / window_start.elapsed().as_secs_f64();
                                window_start = Instant::now();
                                let [pre, infer, copy, decode, bg, draw, present, latency] =
                                    window.map(|d| (d / REPORT_EVERY).as_secs_f64() * 1000.0);
                                let status = format!(
                                    "  FPS: {fps:5.1}  pre:{pre:4.1} inf:{infer:5.1} \
                                     cpy:{copy:4.1} dec:{decode:5.1} bg:{bg:4.1} draw:{draw:5.1} \
                                     disp:{present:4.1} lat:{latency:5.1} ms  \
                                     drop:{dropped} det:{last_detections:<3} {exposure_range}"
                                );
                                // The first window includes frames captured before the
                                // controls took effect.
                                status_lines += 1;
                                if status_lines == 2 {
                                    if stderr_is_terminal {
                                        eprintln!();
                                    }
                                    check_exposure((args.exposure, args.gain), &exposure_range);
                                }
                                // A terminal redraws one line in place; a log, such as
                                // the journal, only records complete lines.
                                if stderr_is_terminal {
                                    eprint!("\r{status}  ");
                                } else {
                                    eprintln!("{status}");
                                }
                                exposure_range = ExposureRange::default();
                                window = [Duration::ZERO; 8];
                            }
                        }
                    }
                }

                // Feed the newest frame to the pipeline, or give it back to
                // the camera when the pipeline is still busy.
                if let Some(request) = newest {
                    let index = usize::try_from(request.cookie())
                        .map_err(|_| Error::camera("request cookie out of range"))?;
                    let buffer: &FrameBuffer = request
                        .buffer(&stream)
                        .ok_or_else(|| Error::camera("completed request has no buffer"))?;
                    let image = frames.take(index, &overlay_processor, buffer)?;
                    let frame = pipeline::Frame {
                        index,
                        image,
                        entered: Instant::now(),
                    };
                    match frame_tx.try_send(frame) {
                        Ok(()) => {
                            in_flight.insert(index, request);
                        }
                        Err(mpsc::TrySendError::Full(frame)) => {
                            frames.put(frame.index, frame.image);
                            requeue(&camera, request)?;
                            dropped += 1;
                        }
                        // A stage has stopped; its error is collected below.
                        Err(mpsc::TrySendError::Disconnected(_)) => break,
                    }
                }
            }
            Ok(())
        })();

        // Closing the first channel winds the stages down in order.
        drop(frame_tx);
        for stage in stages {
            stage
                .join()
                .map_err(|_| Error::unsupported("a pipeline thread panicked"))??;
        }
        looped
    })?;

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
