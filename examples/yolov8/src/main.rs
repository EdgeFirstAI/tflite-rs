// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! # `YOLOv8` Object Detection & Segmentation with `edgefirst-tflite`
//!
//! End-to-end `YOLOv8` inference using `edgefirst-tflite` for model execution
//! and `edgefirst-hal` for image preprocessing, YOLO decoding (via high-level
//! `Decoder` API), and overlay rendering.
//!
//! Supports both detection-only (`yolov8n`) and instance segmentation
//! (`yolov8n-seg`) models. The model type is auto-detected from output shapes:
//! a 4D output tensor indicates segmentation prototype masks.
//!
//! Supports i.MX8MP (`VxDelegate` with DMA-BUF zero-copy and
//! `CameraAdaptor`), i.MX95 (Neutron NPU delegate), and CPU-only inference.
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

use std::path::PathBuf;
use std::time::{Duration, Instant};

use edgefirst_hal::{
    decoder::{
        configs, ArrayViewDQuantized, ConfigOutput, DecoderBuilder, DecoderVersion, DetectBox, Nms,
        Segmentation,
    },
    image::{
        Crop, Flip, ImageProcessor, ImageProcessorTrait as _, Rect, Rotation, TensorImage, RGB,
        RGBA,
    },
    tensor::{TensorMapTrait as _, TensorTrait as _},
};
use edgefirst_tflite::{Delegate, Interpreter, Library, Model, TensorType};
use ndarray::{ArrayViewD, IxDyn};

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
    while i < args.len() {
        match args[i].as_str() {
            "--delegate" => {
                i += 1;
                delegate = Some(PathBuf::from(&args[i]));
            }
            "--save" => save = true,
            "--threshold" => {
                i += 1;
                threshold = args[i].parse().expect("invalid --threshold value");
            }
            "--iou" => {
                i += 1;
                iou = args[i].parse().expect("invalid --iou value");
            }
            "--warmup" => {
                i += 1;
                warmup = args[i].parse().expect("invalid --warmup value");
            }
            "--iters" => {
                i += 1;
                iters = args[i].parse().expect("invalid --iters value");
            }
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

// ── COCO labels ──────────────────────────────────────────────────────────────

const COCO: &[&str] = &[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

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

/// Collected per-iteration timings for a pipeline run.
struct IterTimings {
    infer: Vec<f64>,
    decode: Vec<f64>,
    render: Vec<f64>,
    total: Vec<f64>,
}

impl IterTimings {
    fn with_capacity(n: usize) -> Self {
        Self {
            infer: Vec::with_capacity(n),
            decode: Vec::with_capacity(n),
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
            println!("  Infer:   {:.1}ms", self.infer[0]);
            println!("  Decode:  {:.1}ms", self.decode[0]);
            if !self.render.is_empty() {
                println!("  Render:  {:.1}ms", self.render[0]);
            }
            println!("  Total:   {:.1}ms", self.total[0]);
            return;
        }
        println!("--- {label} ({n} iterations) ---");
        println!("               min      max      avg      p95      p99");
        print_row("Infer", &self.infer);
        print_row("Decode", &self.decode);
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
    let idx = ((sorted.len() as f64 - 1.0) * p).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[allow(clippy::cast_precision_loss)]
fn print_row(label: &str, values: &[f64]) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let avg = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p95 = percentile(&sorted, 0.95);
    let p99 = percentile(&sorted, 0.99);
    println!("  {label:<8} {min:>7.1} {max:>8.1} {avg:>8.1} {p95:>8.1} {p99:>8.1} ms");
}

// ── Output classification ────────────────────────────────────────────────────

/// Build a `ConfigOutput` for a `TFLite` output tensor based on its shape.
///
/// Classification heuristics for `YOLOv8` output tensors:
/// - 4D tensor → `Protos` (segmentation prototype mask tensor)
/// - 3D tensor with feature dim == 4 → `Boxes` (split detection boxes)
/// - 3D tensor with feature dim == `proto_channels` (when split + seg) →
///   `MaskCoefficients`
/// - 3D tensor in split model → `Scores`
/// - 3D tensor in combined model → `Detection` (boxes + scores fused)
fn classify_output(
    shape: &[usize],
    quant: Option<configs::QuantTuple>,
    has_split_boxes: bool,
    has_protos: bool,
    proto_channels: usize,
) -> ConfigOutput {
    if shape.len() == 4 {
        return ConfigOutput::Protos(configs::Protos {
            decoder: configs::DecoderType::Ultralytics,
            quantization: quant,
            shape: shape.to_vec(),
            dshape: Vec::new(),
        });
    }

    let feat_dim = if shape.len() == 3 { shape[1] } else { shape[0] };

    if has_split_boxes {
        if feat_dim == 4 {
            ConfigOutput::Boxes(configs::Boxes {
                decoder: configs::DecoderType::Ultralytics,
                quantization: quant,
                shape: shape.to_vec(),
                dshape: Vec::new(),
                normalized: None,
            })
        } else if has_protos && proto_channels > 0 && feat_dim == proto_channels {
            ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                decoder: configs::DecoderType::Ultralytics,
                quantization: quant,
                shape: shape.to_vec(),
                dshape: Vec::new(),
            })
        } else {
            ConfigOutput::Scores(configs::Scores {
                decoder: configs::DecoderType::Ultralytics,
                quantization: quant,
                shape: shape.to_vec(),
                dshape: Vec::new(),
            })
        }
    } else {
        ConfigOutput::Detection(configs::Detection {
            decoder: configs::DecoderType::Ultralytics,
            quantization: quant,
            shape: shape.to_vec(),
            anchors: None,
            dshape: Vec::new(),
            normalized: None,
        })
    }
}

// ── Decode helper ────────────────────────────────────────────────────────────

/// Read outputs from the interpreter and decode via the `Decoder`.
fn decode_outputs(
    interpreter: &Interpreter<'_>,
    decoder: &edgefirst_hal::decoder::Decoder,
    detections: &mut Vec<DetectBox>,
    masks: &mut Vec<Segmentation>,
    in_w: usize,
    in_h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let outputs = interpreter.outputs()?;
    let is_float = outputs
        .iter()
        .all(|t| t.tensor_type() == TensorType::Float32);

    if is_float {
        let shapes: Vec<Vec<usize>> = outputs
            .iter()
            .map(edgefirst_tflite::Tensor::shape)
            .collect::<Result<Vec<_>, _>>()?;
        let slices: Vec<&[f32]> = outputs
            .iter()
            .map(|t| t.as_slice::<f32>())
            .collect::<Result<Vec<_>, _>>()?;
        let views: Vec<ArrayViewD<f32>> = shapes
            .iter()
            .zip(slices.iter())
            .map(|(shape, data)| ArrayViewD::from_shape(IxDyn(shape), data))
            .collect::<Result<Vec<_>, _>>()?;
        decoder.decode_float(&views, detections, masks)?;
    } else {
        let shapes: Vec<Vec<usize>> = outputs
            .iter()
            .map(edgefirst_tflite::Tensor::shape)
            .collect::<Result<Vec<_>, _>>()?;
        let views: Vec<ArrayViewDQuantized> = outputs
            .iter()
            .zip(shapes.iter())
            .map(|(t, shape)| match t.tensor_type() {
                TensorType::UInt8 => Ok(ArrayViewDQuantized::UInt8(ArrayViewD::from_shape(
                    IxDyn(shape),
                    t.as_slice::<u8>()?,
                )?)),
                TensorType::Int8 => Ok(ArrayViewDQuantized::Int8(ArrayViewD::from_shape(
                    IxDyn(shape),
                    t.as_slice::<i8>()?,
                )?)),
                other => Err(format!("unsupported output type: {other:?}").into()),
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        decoder.decode_quantized(&views, detections, masks)?;
    }

    // Normalize box coordinates to [0,1] if in pixel space.
    #[allow(clippy::cast_precision_loss)]
    if decoder.normalized_boxes() != Some(true) {
        let needs_norm = detections
            .iter()
            .any(|d| d.bbox.xmax > 2.0 || d.bbox.ymax > 2.0);
        if needs_norm {
            let iw = in_w as f32;
            let ih = in_h as f32;
            for det in detections {
                det.bbox.xmin /= iw;
                det.bbox.ymin /= ih;
                det.bbox.xmax /= iw;
                det.bbox.ymax /= ih;
            }
        }
    }

    Ok(())
}

// ── Pipeline iteration ───────────────────────────────────────────────────────

/// Run n iterations of invoke → decode → draw, collecting per-stage timings.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_iterations(
    n: usize,
    interpreter: &mut Interpreter<'_>,
    decoder: &edgefirst_hal::decoder::Decoder,
    processor: &mut ImageProcessor,
    mut overlay: Option<&mut TensorImage>,
    dmabuf_handle: Option<edgefirst_tflite::dmabuf::BufferHandle>,
    in_w: usize,
    in_h: usize,
) -> Result<(Vec<DetectBox>, Vec<Segmentation>, IterTimings), Box<dyn std::error::Error>> {
    let mut timings = IterTimings::with_capacity(n);
    let mut detections = Vec::with_capacity(100);
    let mut masks = Vec::with_capacity(100);

    for _ in 0..n {
        let t_total = Instant::now();

        // Infer
        let t_inf = Instant::now();
        interpreter.invoke()?;
        timings.infer.push(ms(t_inf.elapsed()));

        // DMA-BUF sync
        if let Some(handle) = dmabuf_handle {
            let delegate_ref = interpreter.delegate(0).expect("delegate not found");
            let dmabuf = delegate_ref.dmabuf().expect("DMA-BUF not available");
            dmabuf.sync_for_cpu(handle)?;
        }

        // Decode
        let t_dec = Instant::now();
        decode_outputs(
            interpreter,
            decoder,
            &mut detections,
            &mut masks,
            in_w,
            in_h,
        )?;
        timings.decode.push(ms(t_dec.elapsed()));

        // Render (if --save)
        if let Some(ref mut ov) = overlay {
            let t_render = Instant::now();
            processor.draw_masks(ov, &detections, &masks)?;
            timings.render.push(ms(t_render.elapsed()));
        }

        timings.total.push(ms(t_total.elapsed()));
    }

    Ok((detections, masks, timings))
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    // ── 1. Load TFLite library, model, delegate, interpreter ────────
    let t_load = Instant::now();

    let lib = Library::new()?;
    let model = Model::from_file(&lib, args.model.to_str().unwrap_or("model.tflite"))?;
    println!("Model: {}", args.model.display());

    // Optionally load delegate and probe NPU features.
    //
    // --- Optimized path (i.MX8MP VxDelegate) ---
    // When CameraAdaptor is available, the delegate handles RGBA→RGB conversion
    // on the NPU.  Combined with DMA-BUF, the entire input path is zero-copy:
    //   camera RGBA buffer → DMA-BUF → NPU (format convert + inference)
    //
    // --- Fallback path (delegates without DMA-BUF/CameraAdaptor) ---
    // Without DMA-BUF, we copy preprocessed pixels into the TFLite input tensor.
    // Without CameraAdaptor, HAL ImageProcessor handles format conversion via
    // GPU (OpenGL), G2D hardware accelerator, or CPU — in that priority order.
    let mut use_camera_adaptor = false;
    let use_dmabuf;

    let delegate = if let Some(ref delegate_path) = args.delegate {
        let d = Delegate::load(delegate_path)?;
        println!("Delegate: {}", delegate_path.display());

        if d.has_camera_adaptor() {
            if let Some(adaptor) = d.camera_adaptor() {
                adaptor.set_format(0, "rgba")?;
                use_camera_adaptor = true;
                println!("  CameraAdaptor: enabled (RGBA -> RGB on NPU)");
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
    let (in_h, in_w, input_type, input_quant) = {
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

    // ── 3. Inspect outputs, auto-detect model type, build Decoder ───
    let decoder = {
        let outputs = interpreter.outputs()?;

        let mut has_protos = false;
        let mut proto_channels = 0usize;
        let mut has_split_boxes = false;

        for tensor in &outputs {
            let shape = tensor.shape()?;
            if shape.len() == 4 {
                has_protos = true;
                proto_channels = *shape[1..].iter().min().unwrap_or(&0);
            } else if shape.len() >= 2 {
                let feat_dim = if shape.len() == 3 { shape[1] } else { shape[0] };
                if feat_dim == 4 {
                    has_split_boxes = true;
                }
            }
        }

        let mut dec_builder = DecoderBuilder::default()
            .with_score_threshold(args.threshold)
            .with_iou_threshold(args.iou)
            .with_nms(Some(Nms::ClassAgnostic));

        for (i, tensor) in outputs.iter().enumerate() {
            let shape = tensor.shape()?;
            let qp = tensor.quantization_params();
            println!(
                "  output[{i}]: {} (scale={}, zero_point={})",
                tensor, qp.scale, qp.zero_point
            );

            let quant = if tensor.tensor_type() == TensorType::Float32 {
                None
            } else {
                Some(configs::QuantTuple(qp.scale, qp.zero_point))
            };

            dec_builder = dec_builder.add_output(classify_output(
                &shape,
                quant,
                has_split_boxes,
                has_protos,
                proto_channels,
            ));
        }

        dec_builder = dec_builder.with_decoder_version(DecoderVersion::Yolov8);
        dec_builder.build()?
    };

    println!(
        "  Mode: {}",
        if format!("{:?}", decoder.model_type())
            .to_lowercase()
            .contains("seg")
        {
            "segmentation"
        } else {
            "detection"
        }
    );

    // ── 4. Load and preprocess image (once) ─────────────────────────
    let image_bytes = std::fs::read(&args.image)?;
    let src = TensorImage::load(&image_bytes, Some(RGBA), None)?;
    let (img_w, img_h) = (src.width(), src.height());
    println!("Image: {img_w}x{img_h}");

    let dst_format = if use_camera_adaptor { RGBA } else { RGB };
    let mut dst = TensorImage::new(in_w, in_h, dst_format, None)?;

    let letterbox = compute_letterbox(img_w, img_h, in_w, in_h);

    let t_pre = Instant::now();
    let mut processor = ImageProcessor::new()?;
    processor.convert(&src, &mut dst, Rotation::None, Flip::None, letterbox)?;

    // ── 5. Write preprocessed pixels to input tensor (once) ─────────
    #[allow(unused_variables)]
    let dmabuf_handle = if use_camera_adaptor && use_dmabuf {
        let map = dst.tensor().map()?;
        let pixels = map.as_slice();

        let delegate_ref = interpreter.delegate(0).expect("delegate not found");
        let dmabuf = delegate_ref.dmabuf().expect("DMA-BUF not available");
        let buf_size = in_h * in_w * 4;
        let (handle, desc) =
            dmabuf.request(0, edgefirst_tflite::dmabuf::Ownership::Delegate, buf_size)?;
        dmabuf.bind_to_tensor(handle, 0)?;

        let (ptr, needs_munmap) = if let Some(p) = desc.map_ptr {
            (p.cast::<u8>(), false)
        } else {
            // SAFETY: `desc.fd` is a valid DMA-BUF file descriptor returned by
            // the delegate. We map it as shared + writable for CPU access.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    buf_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    desc.fd,
                    0,
                )
            };
            assert!(p != libc::MAP_FAILED, "failed to mmap DMA-BUF fd");
            (p.cast::<u8>(), true)
        };

        // SAFETY: `ptr` points to `buf_size` bytes of mapped DMA-BUF memory.
        // `pixels` contains exactly `buf_size` bytes of RGBA data.
        unsafe {
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr, pixels.len());
        }

        if needs_munmap {
            // SAFETY: `ptr` was obtained from mmap with size `buf_size`.
            unsafe { libc::munmap(ptr.cast(), buf_size) };
        }

        dmabuf.sync_for_device(handle)?;
        println!("  DMA-BUF + CameraAdaptor: RGBA input (zero-copy)");
        Some(handle)
    } else {
        {
            let map = dst.tensor().map()?;
            let pixels = map.as_slice();
            let mut inputs = interpreter.inputs_mut()?;
            let input = &mut inputs[0];

            match input_type {
                TensorType::Float32 => {
                    let float_data: Vec<f32> =
                        pixels.iter().map(|&v| f32::from(v) / 255.0).collect();
                    input.copy_from_slice(&float_data)?;
                }
                TensorType::UInt8 => {
                    input.copy_from_slice(pixels)?;
                }
                TensorType::Int8 => {
                    #[allow(clippy::cast_possible_wrap)]
                    let i8_data: Vec<i8> =
                        pixels.iter().map(|&v| v.wrapping_sub(128) as i8).collect();
                    input.copy_from_slice(&i8_data)?;
                }
                _ => {
                    return Err(format!(
                        "unsupported input type: {input_type:?} (scale={}, zp={})",
                        input_quant.scale, input_quant.zero_point
                    )
                    .into());
                }
            }
        }

        if use_dmabuf {
            let delegate_ref = interpreter.delegate(0).expect("delegate not found");
            let dmabuf = delegate_ref.dmabuf().expect("DMA-BUF not available");
            let byte_size = {
                let inputs = interpreter.inputs()?;
                inputs[0].byte_size()
            };
            let (handle, _desc) =
                dmabuf.request(0, edgefirst_tflite::dmabuf::Ownership::Delegate, byte_size)?;
            dmabuf.bind_to_tensor(handle, 0)?;
            dmabuf.sync_for_device(handle)?;
            println!("  DMA-BUF: bound to input tensor (zero-copy)");
            Some(handle)
        } else {
            None
        }
    };
    let preprocess_time = t_pre.elapsed();

    // Pre-allocate overlay for rendering (reused across iterations).
    let mut overlay = if args.save {
        Some(TensorImage::load(&image_bytes, Some(RGBA), None)?)
    } else {
        None
    };

    // ── 6. Warmup iterations ────────────────────────────────────────
    if args.warmup > 0 {
        println!("\nRunning {} warmup iterations...", args.warmup);
        let (_, _, warmup_timings) = run_iterations(
            args.warmup,
            &mut interpreter,
            &decoder,
            &mut processor,
            overlay.as_mut(),
            dmabuf_handle,
            in_w,
            in_h,
        )?;
        println!();
        warmup_timings.print_stats("Warmup");
    }

    // ── 7. Benchmark iterations ─────────────────────────────────────
    let (detections, _masks, bench_timings) = run_iterations(
        args.iters,
        &mut interpreter,
        &decoder,
        &mut processor,
        overlay.as_mut(),
        dmabuf_handle,
        in_w,
        in_h,
    )?;

    // ── 8. Print detections (from last iteration) ───────────────────
    println!("\n--- Detections ({}) ---", detections.len());
    #[allow(clippy::cast_precision_loss)]
    for det in &detections {
        let name = COCO.get(det.label).unwrap_or(&"?");
        let x1 = (det.bbox.xmin * img_w as f32).max(0.0);
        let y1 = (det.bbox.ymin * img_h as f32).max(0.0);
        let x2 = (det.bbox.xmax * img_w as f32).min(img_w as f32);
        let y2 = (det.bbox.ymax * img_h as f32).min(img_h as f32);
        println!(
            "  {name:>12} ({:2}): {:5.1}%  [{:.0}, {:.0}, {:.0}, {:.0}]",
            det.label,
            det.score * 100.0,
            x1,
            y1,
            x2,
            y2
        );
    }

    // ── 9. Save overlay (once) ──────────────────────────────────────
    if let Some(ref ov) = overlay {
        let stem = args.image.file_stem().unwrap_or_default().to_string_lossy();
        let out_path = args.image.with_file_name(format!("{stem}_overlay.jpg"));
        ov.save_jpeg(out_path.to_str().unwrap(), 95)?;
        println!("  Saved: {}", out_path.display());
    }

    // ── 10. Print timing ────────────────────────────────────────────
    println!();
    println!("  Load:       {:.1}ms", ms(load_time));
    println!("  Preprocess: {:.1}ms", ms(preprocess_time));
    println!();
    bench_timings.print_stats(&format!("Benchmark (iters={})", args.iters));

    Ok(())
}
