// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! # `YOLOv8` Object Detection & Segmentation with `edgefirst-tflite`
//!
//! End-to-end `YOLOv8` inference using `edgefirst-tflite` for model execution
//! and `edgefirst-hal` for image preprocessing, YOLO decoding (via high-level
//! `Decoder` API), and overlay rendering.
//!
//! Supports both detection-only (`yolov8n`) and instance segmentation
//! (`yolov8n-seg`) models. The decoder is configured from the model's
//! embedded `edgefirst.json` schema (extracted from the ZIP archive that the
//! `EdgeFirst` converter appends to the `.tflite` flatbuffer), so all three
//! YOLO output layouts — fused, logical-split, and per-scale FPN-split —
//! work transparently without manual output classification.
//!
//! Supports i.MX8MP (`VxDelegate` with DMA-BUF zero-copy and `CameraAdaptor`),
//! i.MX95 (Neutron NPU with HAL DMA-BUF zero-copy), and CPU-only inference.
//! Uses the portable HAL Delegate API (`tensor_info`, `import_image`) where
//! available, with a deprecated `VxDelegate`-specific fallback.
//!
//! ## Usage
//!
//! ```text
//! yolov8 <model.tflite> <image.jpg> [--delegate <path>] [--save]
//!        [--threshold N] [--iou N] [--warmup N] [--iters N]
//! ```
//!
//! ## Examples
//!
//! ```sh
//! # CPU-only inference
//! cargo run -p yolov8 -- model.tflite image.jpg
//!
//! # Benchmark with 5 warmup + 100 iterations
//! yolov8 model.tflite image.jpg --delegate /usr/lib/libvx_delegate.so --warmup 5 --iters 100 --save
//! ```

mod error;

use std::os::fd::BorrowedFd;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use edgefirst_hal::{
    decoder::{
        schema::{LogicalType, SchemaV2},
        DecoderBuilder, DetectBox, ProtoData, Segmentation,
    },
    image::{
        load_image, save_jpeg, ColorMode, Crop, Flip, ImageProcessor, ImageProcessorTrait as _,
        MaskOverlay, MaskResolution, Rect, Rotation,
    },
    tensor::{
        DType, PixelFormat, PlaneDescriptor, Quantization, TensorDyn, TensorMapTrait as _,
        TensorMemory, TensorTrait as _,
    },
};
use edgefirst_tflite::{
    archive::ModelArchive, Delegate, DelegateOptions, Interpreter, Library, Model, TensorType,
};

// ── Arguments ────────────────────────────────────────────────────────────────

struct Args {
    model: PathBuf,
    image: PathBuf,
    delegate: Option<PathBuf>,
    save: bool,
    threshold: f32,
    iou: f32,
    warmup: usize,
    iters: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model.tflite> <image.jpg> [--delegate <path>] [--save] \
             [--threshold N] [--iou N] [--warmup N] [--iters N]",
            args[0]
        );
        std::process::exit(1);
    }
    let mut delegate = None;
    let mut threshold = 0.25;
    let mut iou = 0.45;
    let mut save = false;
    let mut warmup = 0;
    let mut iters = 1;
    let mut i = 3;
    macro_rules! next_arg {
        ($flag:expr) => {{
            i += 1;
            if i >= args.len() {
                eprintln!("missing value for {}", $flag);
                std::process::exit(1);
            }
            &args[i]
        }};
    }
    while i < args.len() {
        match args[i].as_str() {
            "--delegate" => delegate = Some(PathBuf::from(next_arg!("--delegate"))),
            "--save" => save = true,
            "--threshold" => {
                threshold = next_arg!("--threshold")
                    .parse()
                    .expect("invalid --threshold value");
            }
            "--iou" => iou = next_arg!("--iou").parse().expect("invalid --iou value"),
            "--warmup" => {
                warmup = next_arg!("--warmup")
                    .parse()
                    .expect("invalid --warmup value");
            }
            "--iters" => iters = next_arg!("--iters").parse().expect("invalid --iters value"),
            other => eprintln!("Unknown argument: {other}"),
        }
        i += 1;
    }
    Args {
        model: args[1].clone().into(),
        image: args[2].clone().into(),
        delegate,
        save,
        threshold,
        iou,
        warmup,
        iters,
    }
}

// ── Letterbox ────────────────────────────────────────────────────────────────

/// Compute a letterbox crop that preserves aspect ratio with gray padding.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn compute_letterbox(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Crop {
    let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale) as usize;
    let new_h = (src_h as f32 * scale) as usize;
    Crop::new()
        .with_dst_rect(Some(Rect::new(
            (dst_w - new_w) / 2,
            (dst_h - new_h) / 2,
            new_w,
            new_h,
        )))
        .with_dst_color(Some([114, 114, 114, 255])) // YOLO gray
}

// ── Timing / statistics ─────────────────────────────────────────────────────

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Per-stage timings (ms) for each pipeline iteration.
struct IterTimings {
    preprocess: Vec<f64>,
    infer: Vec<f64>,
    copy: Vec<f64>,
    decode: Vec<f64>,
    materialize: Vec<f64>,
    render: Vec<f64>,
    total: Vec<f64>,
}

impl IterTimings {
    fn with_capacity(n: usize) -> Self {
        Self {
            preprocess: Vec::with_capacity(n),
            infer: Vec::with_capacity(n),
            copy: Vec::with_capacity(n),
            decode: Vec::with_capacity(n),
            materialize: Vec::with_capacity(n),
            render: Vec::with_capacity(n),
            total: Vec::with_capacity(n),
        }
    }

    fn print_stats(&self, label: &str) {
        let n = self.total.len();
        if n == 0 {
            return;
        }
        if n == 1 {
            println!("--- {label} ---");
            println!("  Preprocess:   {:.1}ms", self.preprocess[0]);
            println!("  Infer:        {:.1}ms", self.infer[0]);
            println!("  Copy:         {:.1}ms", self.copy[0]);
            println!("  Decode:       {:.1}ms", self.decode[0]);
            println!("  Materialize:  {:.1}ms", self.materialize[0]);
            if !self.render.is_empty() {
                println!("  Render:       {:.1}ms", self.render[0]);
            }
            println!("  Total:        {:.1}ms", self.total[0]);
            return;
        }
        println!("--- {label} ({n} iterations) ---");
        println!("                    min      max      avg      p95      p99");
        print_row("Preprocess", &self.preprocess);
        print_row("Infer", &self.infer);
        print_row("Copy", &self.copy);
        print_row("Decode", &self.decode);
        print_row("Materialize", &self.materialize);
        if !self.render.is_empty() {
            print_row("Render", &self.render);
        }
        print_row("Total", &self.total);
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = ((sorted.len() as f64 - 1.0) * p) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[allow(clippy::cast_precision_loss)]
fn print_row(label: &str, values: &[f64]) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let max = *sorted.last().unwrap();
    let avg = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p95 = percentile(&sorted, 0.95);
    let p99 = percentile(&sorted, 0.99);
    println!("  {label:<10} {min:>7.1} {max:>8.1} {avg:>8.1} {p95:>8.1} {p99:>8.1} ms");
}

// ── Output buffers / model input / pipeline helpers ──────────────────────────

fn tflite_dtype(tt: TensorType) -> Result<DType> {
    match tt {
        TensorType::Float32 => Ok(DType::F32),
        TensorType::Int8 => Ok(DType::I8),
        TensorType::UInt8 => Ok(DType::U8),
        TensorType::Int32 => Ok(DType::I32),
        other => Err(Error::unsupported(format!("output dtype: {other:?}"))),
    }
}

/// Pre-allocated HAL tensors for `TFLite` outputs (updated in-place each frame).
struct OutputBuffers(Vec<TensorDyn>);

impl OutputBuffers {
    fn allocate(interpreter: &Interpreter<'_>) -> Result<Self> {
        let outputs = interpreter.outputs()?;
        let mut tensors = Vec::with_capacity(outputs.len());
        for t in &outputs {
            let shape = t.shape()?;
            let dtype = tflite_dtype(t.tensor_type())?;
            let mut td = TensorDyn::new(&shape, dtype, Some(TensorMemory::Mem), None)?;
            // The HAL per-scale decoder reads quantization from the tensor
            // itself (not from the schema), so propagate the TFLite tensor's
            // (scale, zero_point) onto each integer output buffer. Float
            // outputs reject set_quantization — skip them.
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
            match tflite_dtype(t.tensor_type())? {
                DType::F32 => {
                    td.as_f32_mut()
                        .expect("F32")
                        .map()?
                        .as_mut_slice()
                        .copy_from_slice(t.as_slice::<f32>()?);
                }
                DType::I8 => {
                    td.as_i8_mut()
                        .expect("I8")
                        .map()?
                        .as_mut_slice()
                        .copy_from_slice(t.as_slice::<i8>()?);
                }
                DType::U8 => {
                    td.as_u8_mut()
                        .expect("U8")
                        .map()?
                        .as_mut_slice()
                        .copy_from_slice(t.as_slice::<u8>()?);
                }
                DType::I32 => {
                    td.as_i32_mut()
                        .expect("I32")
                        .map()?
                        .as_mut_slice()
                        .copy_from_slice(t.as_slice::<i32>()?);
                }
                _ => return Err(Error::unsupported("output dtype")),
            }
        }
        Ok(())
    }

    fn refs(&self) -> Vec<&TensorDyn> {
        self.0.iter().collect()
    }
}

/// Model input: GPU-rendered DMA-BUF (zero-copy to NPU) or CPU staging buffer.
enum ModelInput {
    DmaBuf(TensorDyn),
    Staging(TensorDyn),
}

/// Normalize detection boxes to [0,1] if the model outputs pixel coordinates.
#[allow(clippy::cast_precision_loss)]
fn maybe_normalize_boxes(detections: &mut [DetectBox], in_w: usize, in_h: usize) {
    if let Some(first) = detections.first() {
        if first.bbox.xmax > 2.0 || first.bbox.ymax > 2.0 {
            let iw = in_w as f32;
            let ih = in_h as f32;
            for d in detections.iter_mut() {
                d.bbox.xmin /= iw;
                d.bbox.ymin /= ih;
                d.bbox.xmax /= iw;
                d.bbox.ymax /= ih;
            }
        }
    }
}

// ── Pipeline helpers ─────────────────────────────────────────────────────────

/// GPU-convert `src` into the model input tensor for one frame.
#[allow(clippy::too_many_arguments)]
fn preprocess_step(
    src: &TensorDyn,
    model_input: &mut ModelInput,
    processor: &mut ImageProcessor,
    interpreter: &mut Interpreter<'_>,
    letterbox: Crop,
    use_dmabuf: bool,
    input_type: TensorType,
) -> Result<()> {
    match model_input {
        ModelInput::DmaBuf(dst) => {
            processor.convert(src, dst, Rotation::None, Flip::None, letterbox)?;
            if use_dmabuf {
                interpreter
                    .delegate(0)
                    .expect("delegate")
                    .dmabuf()
                    .expect("dmabuf")
                    .sync_for_device(0)?;
            }
        }
        ModelInput::Staging(staging) => {
            processor.convert(src, staging, Rotation::None, Flip::None, letterbox)?;
            let mut inputs = interpreter.inputs_mut()?;
            let inp = &mut inputs[0];
            match input_type {
                TensorType::Int8 => {
                    inp.copy_from_slice(staging.as_i8().expect("i8").map()?.as_slice())?;
                }
                TensorType::UInt8 => {
                    inp.copy_from_slice(staging.as_u8().expect("u8").map()?.as_slice())?;
                }
                TensorType::Float32 => {
                    let map = staging.as_u8().expect("u8 staging for f32").map()?;
                    let f32_slice = inp.as_mut_slice::<f32>()?;
                    for (d, &s) in f32_slice.iter_mut().zip(map.as_slice().iter()) {
                        *d = f32::from(s) / 255.0;
                    }
                }
                other => return Err(Error::unsupported(format!("input type: {other:?}"))),
            }
        }
    }
    Ok(())
}

// ── Pipeline iteration ───────────────────────────────────────────────────────

/// Run `n` iterations of preprocess -> infer -> copy -> decode -> render,
/// returning the last iteration's detections and per-stage timings.
#[allow(clippy::too_many_arguments)]
fn run_iterations(
    n: usize,
    interpreter: &mut Interpreter<'_>,
    decoder: &edgefirst_hal::decoder::Decoder,
    processor: &mut ImageProcessor,
    src: &TensorDyn,
    model_input: &mut ModelInput,
    output_bufs: &mut OutputBuffers,
    mut dst: Option<&mut TensorDyn>,
    letterbox: Crop,
    in_w: usize,
    in_h: usize,
    use_dmabuf: bool,
    input_type: TensorType,
) -> Result<(Vec<DetectBox>, IterTimings)> {
    let mut timings = IterTimings::with_capacity(n);
    let mut detections: Vec<DetectBox> = Vec::with_capacity(100);
    // Precompute the normalised letterbox rect once for use in materialize_segmentations.
    let letterbox_norm = MaskOverlay::default()
        .with_letterbox_crop(&letterbox, in_w, in_h)
        .letterbox;

    for _ in 0..n {
        detections.clear();
        let t_total = Instant::now();

        let t_pre = Instant::now();
        preprocess_step(
            src,
            model_input,
            processor,
            interpreter,
            letterbox,
            use_dmabuf,
            input_type,
        )?;
        timings.preprocess.push(ms(t_pre.elapsed()));

        let t_inf = Instant::now();
        interpreter.invoke()?;
        timings.infer.push(ms(t_inf.elapsed()));

        let t_copy = Instant::now();
        output_bufs.sync_from(interpreter)?;
        timings.copy.push(ms(t_copy.elapsed()));

        // decode_proto populates detections + returns prototype data for seg models.
        // For detection-only models it returns None; fall back to decode().
        let t_decode = Instant::now();
        let refs = output_bufs.refs();
        let mut fallback_masks: Vec<Segmentation> = Vec::new();
        let proto: Option<ProtoData> = decoder.decode_proto(&refs, &mut detections)?;
        if proto.is_none() {
            decoder.decode(&refs, &mut detections, &mut fallback_masks)?;
        }
        maybe_normalize_boxes(&mut detections, in_w, in_h);
        timings.decode.push(ms(t_decode.elapsed()));

        // materialize: CPU dot-product of mask coefficients × proto tensor → bitmaps.
        // For detection-only models this is a no-op (empty proto → empty masks).
        // draw_decoded_masks is then called on ImageProcessor so the GL backend
        // renders the bitmaps onto the DMA-BUF dst tensor.
        let t_mat = Instant::now();
        let masks: Vec<Segmentation> = if let Some(ref proto_data) = proto {
            processor.materialize_masks(
                &detections,
                proto_data,
                letterbox_norm,
                MaskResolution::Proto,
            )?
        } else {
            fallback_masks
        };
        timings.materialize.push(ms(t_mat.elapsed()));

        if let Some(ref mut d) = dst {
            let overlay = MaskOverlay::default()
                .with_background(src)
                .with_letterbox_crop(&letterbox, in_w, in_h)
                .with_color_mode(ColorMode::Instance);
            let t_render = Instant::now();
            processor.draw_decoded_masks(d, &detections, &masks, overlay)?;
            timings.render.push(ms(t_render.elapsed()));
        }

        timings.total.push(ms(t_total.elapsed()));
    }

    Ok((detections, timings))
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines, clippy::similar_names)]
fn main() -> Result<()> {
    let args = parse_args();

    // ── 1. Load TFLite library, model, delegate, interpreter ────────
    let t_load = Instant::now();

    let lib = Library::new()?;
    let model = Model::from_file(&lib, args.model.to_str().unwrap_or("model.tflite"))?;
    println!("Model: {}", args.model.display());

    let mut use_camera_adaptor = false;
    let use_dmabuf;

    let delegate = if let Some(ref delegate_path) = args.delegate {
        let opts = DelegateOptions::new().option("camera_adaptor", "rgba");
        let d = Delegate::load_with_options(delegate_path, &opts)?;
        println!("Delegate: {}", delegate_path.display());

        if d.has_camera_adaptor() {
            if let Some(adaptor) = d.camera_adaptor() {
                if adaptor.is_format_supported("rgba") {
                    use_camera_adaptor = true;
                }
            }
        }

        use_dmabuf = d.has_dmabuf();
        if use_dmabuf {
            println!("  DMA-BUF: available");
        }

        Some(d)
    } else {
        use_dmabuf = false;
        None
    };

    let mut builder = Interpreter::builder(&lib)?.num_threads(4);
    if let Some(d) = delegate {
        builder = builder.delegate(d);
    }
    let mut interpreter = builder.build(&model)?;

    let load_time = t_load.elapsed();

    println!("Inputs:  {}", interpreter.input_count());
    println!("Outputs: {}", interpreter.output_count());

    // ── 2. Inspect input tensor ─────────────────────────────────────
    let (in_h, in_w, input_type, _input_quant) = {
        let inputs = interpreter.inputs()?;
        let input = &inputs[0];
        let shape = input.shape()?;
        let tt = input.tensor_type();
        let qp = input.quantization_params();
        let h = shape[1];
        let w = shape[2];
        println!(
            "  input[0]: {} (scale={}, zero_point={})",
            input, qp.scale, qp.zero_point
        );
        (h, w, tt, qp)
    };

    if use_camera_adaptor {
        println!("  CameraAdaptor: enabled (RGBA \u{2192} RGB on NPU)");
    }

    // ── 3. Print outputs and build Decoder from embedded edgefirst.json ───
    // Emit the same per-output diagnostic lines the manual path used to
    // print, so the output stays familiar across the refactor.
    {
        let outputs = interpreter.outputs()?;
        for (i, tensor) in outputs.iter().enumerate() {
            let qp = tensor.quantization_params();
            println!(
                "  output[{i}]: {} (scale={}, zero_point={})",
                tensor, qp.scale, qp.zero_point
            );
        }
    }

    // The model has no embedded EdgeFirst metadata archive when this returns
    // an error — propagate as `Error::Tflite` (its inner cause already carries
    // the missing-archive message).
    let mut archive = ModelArchive::new(model.data())?;
    let edgefirst_json = archive.edgefirst_json()?;
    let labels = archive.labels().unwrap_or_default();
    println!(
        "  Schema:  edgefirst.json embedded ({} bytes), labels.txt: {} entries",
        edgefirst_json.len(),
        labels.len(),
    );

    // Parse the JSON into a SchemaV2 so we can both feed the builder via the
    // schema-aware path (`with_schema` preserves per-scale FPN children that
    // the merge pipeline relies on for DFL decode) and inspect the logical
    // outputs to decide segmentation vs detection mode without depending on
    // the runtime `Decoder::model_type()` debug formatting.
    let schema = SchemaV2::parse_json(&edgefirst_json)?;
    let is_segmentation = schema
        .outputs
        .iter()
        .any(|o| matches!(o.type_, Some(LogicalType::Protos)));

    let decoder = DecoderBuilder::new()
        .with_schema(schema)
        .with_score_threshold(args.threshold)
        .with_iou_threshold(args.iou)
        .build()?;

    let mode = if is_segmentation {
        "segmentation"
    } else {
        "detection"
    };
    println!("  Mode: {mode}");

    // ── 4. Create ImageProcessor and load source image into DMA-BUF ────
    let t_init = Instant::now();
    let mut processor = ImageProcessor::new()?;

    let image_bytes = std::fs::read(&args.image)?;
    let cpu_img = load_image(&image_bytes, Some(PixelFormat::Rgba), None)?;
    let img_w = cpu_img.width().expect("loaded image must have width");
    let img_h = cpu_img.height().expect("loaded image must have height");
    println!("Image: {img_w}x{img_h}");

    let mut src_rgba = processor.create_image(img_w, img_h, PixelFormat::Rgba, DType::U8, None)?;
    processor.convert(
        &cpu_img,
        &mut src_rgba,
        Rotation::None,
        Flip::None,
        Crop::new(),
    )?;
    drop(cpu_img);

    let letterbox = compute_letterbox(img_w, img_h, in_w, in_h);

    // ── 5. Bind model input (DMA-BUF or staging buffer) ─────────────
    // CameraAdaptor (RGBA input, NPU converts to RGB) is only valid on the DMA-BUF
    // path. CPU staging always needs RGB with the correct dtype for the model.
    let (input_fmt, input_dtype) = if use_camera_adaptor && use_dmabuf {
        (PixelFormat::Rgba, DType::U8)
    } else {
        let dt = if input_type == TensorType::Int8 {
            DType::I8
        } else {
            DType::U8
        };
        (PixelFormat::Rgb, dt)
    };

    let mut model_input: ModelInput = if use_dmabuf && input_type != TensorType::Float32 {
        let delegate_ref = interpreter.delegate(0).expect("delegate not found");
        let dmabuf_api = delegate_ref.dmabuf().expect("DMA-BUF not available");

        if let Ok(info) = dmabuf_api.tensor_info(0) {
            // SAFETY: info.fd is owned by the delegate; PlaneDescriptor dups it.
            let pd = PlaneDescriptor::new(unsafe { BorrowedFd::borrow_raw(info.fd) })?
                .with_offset(info.offset);
            let dst = processor.import_image(pd, None, in_w, in_h, input_fmt, input_dtype)?;
            if use_camera_adaptor {
                println!(
                    "  Input: HAL DMA-BUF + CameraAdaptor \
                     (GPU \u{2192} RGBA \u{2192} DMA-BUF \u{2192} NPU \u{2192} RGB)"
                );
            } else {
                println!(
                    "  Input: HAL DMA-BUF zero-copy \
                     (GPU \u{2192} {} {:?} \u{2192} DMA-BUF \u{2192} NPU)",
                    if input_fmt == PixelFormat::Rgb {
                        "RGB"
                    } else {
                        "RGBA"
                    },
                    input_dtype,
                );
            }
            ModelInput::DmaBuf(dst)
        } else {
            // VxDelegate legacy path: configure CameraAdaptor + DMA-BUF, then
            // invalidate the graph so the first Invoke() recompiles with both.
            if use_camera_adaptor {
                if let Some(adaptor) = delegate_ref.camera_adaptor() {
                    #[allow(deprecated)]
                    adaptor.set_format(0, "rgba")?;
                }
            }
            #[allow(deprecated)]
            let buf_size = in_h * in_w * if use_camera_adaptor { 4 } else { 3 };
            #[allow(deprecated)]
            let (handle, desc) =
                dmabuf_api.request(0, edgefirst_tflite::dmabuf::Ownership::Delegate, buf_size)?;
            #[allow(deprecated)]
            dmabuf_api.bind_to_tensor(handle, 0)?;
            #[allow(deprecated)]
            dmabuf_api.invalidate_graph()?;
            // SAFETY: desc.fd is owned by VxDelegate for the interpreter's lifetime.
            #[allow(deprecated)]
            let pd = PlaneDescriptor::new(unsafe { BorrowedFd::borrow_raw(desc.fd) })?;
            let dst = processor.import_image(pd, None, in_w, in_h, input_fmt, input_dtype)?;
            if use_camera_adaptor {
                println!("  Input: VxDelegate DMA-BUF + CameraAdaptor (legacy, RGBA \u{2192} NPU)");
            } else {
                println!("  Input: VxDelegate DMA-BUF (legacy, GPU \u{2192} DMA-BUF \u{2192} NPU)");
            }
            ModelInput::DmaBuf(dst)
        }
    } else {
        let staging = processor.create_image(in_w, in_h, input_fmt, input_dtype, None)?;
        println!("  Input: CPU staging (GPU \u{2192} staging \u{2192} TFLite arena)");
        ModelInput::Staging(staging)
    };

    // ── 6. Pre-allocate output buffers and render canvas ────────────
    let mut output_bufs = OutputBuffers::allocate(&interpreter)?;

    let mut dst = if args.save {
        Some(processor.create_image(img_w, img_h, PixelFormat::Rgba, DType::U8, None)?)
    } else {
        None
    };

    let init_time = t_init.elapsed();

    // ── 7. Warmup iterations ────────────────────────────────────────
    if args.warmup > 0 {
        println!("\nRunning {} warmup iteration(s)...", args.warmup);
        let (_, warmup_timings) = run_iterations(
            args.warmup,
            &mut interpreter,
            &decoder,
            &mut processor,
            &src_rgba,
            &mut model_input,
            &mut output_bufs,
            dst.as_mut(),
            letterbox,
            in_w,
            in_h,
            use_dmabuf,
            input_type,
        )?;
        println!();
        warmup_timings.print_stats("Warmup");
    }

    // ── 8. Benchmark iterations ─────────────────────────────────────
    let (detections, bench_timings) = run_iterations(
        args.iters,
        &mut interpreter,
        &decoder,
        &mut processor,
        &src_rgba,
        &mut model_input,
        &mut output_bufs,
        dst.as_mut(),
        letterbox,
        in_w,
        in_h,
        use_dmabuf,
        input_type,
    )?;

    // ── 9. Print detections ───────────────────────────────────────────
    println!("\n--- Detections ({}) ---", detections.len());
    #[allow(clippy::cast_precision_loss)]
    {
        let (lx0, ly0, lx1, ly1) = if let Some(r) = letterbox.dst_rect {
            (
                r.left as f32 / in_w as f32,
                r.top as f32 / in_h as f32,
                (r.left + r.width) as f32 / in_w as f32,
                (r.top + r.height) as f32 / in_h as f32,
            )
        } else {
            (0.0_f32, 0.0_f32, 1.0_f32, 1.0_f32)
        };
        let inv_lw = 1.0 / (lx1 - lx0);
        let inv_lh = 1.0 / (ly1 - ly0);
        for det in &detections {
            let name = labels.get(det.label).map_or("?", String::as_str);
            let bbox = det.bbox.to_canonical();
            let x1 = (((bbox.xmin.clamp(0.0, 1.0) - lx0) * inv_lw) * img_w as f32)
                .clamp(0.0, img_w as f32);
            let y1 = (((bbox.ymin.clamp(0.0, 1.0) - ly0) * inv_lh) * img_h as f32)
                .clamp(0.0, img_h as f32);
            let x2 = (((bbox.xmax.clamp(0.0, 1.0) - lx0) * inv_lw) * img_w as f32)
                .clamp(0.0, img_w as f32);
            let y2 = (((bbox.ymax.clamp(0.0, 1.0) - ly0) * inv_lh) * img_h as f32)
                .clamp(0.0, img_h as f32);
            println!(
                "  {name:>12} ({:2}): {:5.1}%  [{:.0}, {:.0}, {:.0}, {:.0}]",
                det.label,
                det.score * 100.0,
                x1,
                y1,
                x2,
                y2,
            );
        }
    }

    // ── 10. Save overlay ──────────────────────────────────────────────
    if let Some(ref d) = dst {
        let stem = args.image.file_stem().unwrap_or_default().to_string_lossy();
        let out_path = args.image.with_file_name(format!("{stem}_overlay.jpg"));
        save_jpeg(d, out_path.to_str().unwrap(), 95)?;
        println!("  Saved: {}", out_path.display());
    }

    // ── 11. Print timing summary ─────────────────────────────────────
    println!();
    println!(
        "  Load+init:  {:.1}ms  (model + image → DMA-BUF, one-time)",
        ms(load_time) + ms(init_time)
    );
    println!();
    bench_timings.print_stats(&format!("Benchmark (iters={})", args.iters));

    Ok(())
}
