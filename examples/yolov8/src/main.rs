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
//! yolov8 <model.tflite> <image.jpg> [--delegate <path>] [--save] [--threshold N] [--iou N]
//! ```
//!
//! ## Examples
//!
//! ```sh
//! # CPU-only inference
//! cargo run -p yolov8 -- model.tflite image.jpg
//!
//! # i.MX8MP with VxDelegate
//! yolov8 yolov8n.tflite image.jpg --delegate /usr/lib/libvx_delegate.so --save
//!
//! # i.MX95 with Neutron NPU (detection)
//! yolov8 yolov8n-int8.imx95.tflite image.jpg --delegate /usr/lib/libneutron_delegate.so --save
//!
//! # i.MX95 with Neutron NPU (segmentation)
//! yolov8 yolov8n-seg-int8.imx95.tflite image.jpg --delegate /usr/lib/libneutron_delegate.so --save
//! ```

use std::path::PathBuf;
use std::time::Instant;

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
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model.tflite> <image.jpg> [--delegate <path>] [--save] [--threshold N] [--iou N]",
            args[0]
        );
        std::process::exit(1);
    }
    let mut delegate = None;
    let mut threshold = 0.25;
    let mut iou = 0.45;
    let mut save = false;
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

// ── Timing helper ────────────────────────────────────────────────────────────

fn fmt_ms(d: std::time::Duration) -> String {
    format!("{:.1}ms", d.as_secs_f64() * 1000.0)
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
    // The Decoder handles all post-processing: dequantization, DFL decoding,
    // NMS, coordinate normalization, and (for seg models) mask generation.
    let is_segmentation;
    let decoder = {
        let outputs = interpreter.outputs()?;

        // First pass: probe for protos (4D) and split boxes (feature_dim==4).
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

        is_segmentation = has_protos;

        // Second pass: classify each output and add to DecoderBuilder.
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

    if is_segmentation {
        println!("  Mode: segmentation");
    } else {
        println!("  Mode: detection");
    }

    // ── 4. Load and preprocess image ────────────────────────────────
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

    // ── 5. Write preprocessed pixels to input tensor / DMA-BUF ──────
    #[allow(unused_variables)]
    let dmabuf_handle = if use_camera_adaptor && use_dmabuf {
        // CameraAdaptor + DMA-BUF: write raw RGBA bytes to DMA-BUF.
        // The NPU handles RGBA→RGB conversion and quantization in-graph.
        let map = dst.tensor().map()?;
        let pixels = map.as_slice();

        let delegate_ref = interpreter.delegate(0).expect("delegate not found");
        let dmabuf = delegate_ref.dmabuf().expect("DMA-BUF not available");
        let buf_size = in_h * in_w * 4;
        let (handle, desc) =
            dmabuf.request(0, edgefirst_tflite::dmabuf::Ownership::Delegate, buf_size)?;
        dmabuf.bind_to_tensor(handle, 0)?;

        // Get or create a CPU-writable mapping for the DMA-BUF.
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
        // Write type-converted pixels to the input tensor.
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
                    // Shift uint8 [0,255] to int8 [-128,127]: subtract 128.
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

        // Optionally set up DMA-BUF for zero-copy inference.
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

    // ── 6. Run inference ────────────────────────────────────────────
    let t_inf = Instant::now();
    interpreter.invoke()?;
    let inference_time = t_inf.elapsed();

    // ── 7. Sync DMA-BUF back to CPU ────────────────────────────────
    if let Some(handle) = dmabuf_handle {
        let delegate_ref = interpreter.delegate(0).expect("delegate not found");
        let dmabuf = delegate_ref.dmabuf().expect("DMA-BUF not available");
        dmabuf.sync_for_cpu(handle)?;
    }

    // ── 8. Decode outputs via Decoder ───────────────────────────────
    // The Decoder handles dequantization, DFL box decoding, NMS, and
    // (for segmentation models) mask coefficient → prototype multiplication.
    let t_post = Instant::now();
    let mut detections: Vec<DetectBox> = Vec::with_capacity(100);
    let mut masks: Vec<Segmentation> = Vec::with_capacity(100);

    {
        let outputs = interpreter.outputs()?;
        let is_float = outputs
            .iter()
            .all(|t| t.tensor_type() == TensorType::Float32);

        if is_float {
            // Float path: wrap each output tensor as ArrayViewD<f32>.
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
            decoder.decode_float(&views, &mut detections, &mut masks)?;
        } else {
            // Quantized path: wrap each output as ArrayViewDQuantized.
            // Handles mixed integer types (e.g., one output i8, another u8).
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
            decoder.decode_quantized(&views, &mut detections, &mut masks)?;
        }
    }

    // Normalize box coordinates to [0,1] if the decoder output is in pixel
    // space (common with quantized Neutron models where coordinates are not
    // pre-normalized by the model).
    #[allow(clippy::cast_precision_loss)]
    if decoder.normalized_boxes() != Some(true) {
        let needs_norm = detections
            .iter()
            .any(|d| d.bbox.xmax > 2.0 || d.bbox.ymax > 2.0);
        if needs_norm {
            let iw = in_w as f32;
            let ih = in_h as f32;
            for det in &mut detections {
                det.bbox.xmin /= iw;
                det.bbox.ymin /= ih;
                det.bbox.xmax /= iw;
                det.bbox.ymax /= ih;
            }
        }
    }

    let postprocess_time = t_post.elapsed();

    // ── 9. Print detections ─────────────────────────────────────────
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

    // ── 10. Optionally save overlay ─────────────────────────────────
    let render_time = if args.save {
        let t_render = Instant::now();
        let mut overlay = TensorImage::load(&image_bytes, Some(RGBA), None)?;
        processor.draw_masks(&mut overlay, &detections, &masks)?;

        let stem = args.image.file_stem().unwrap_or_default().to_string_lossy();
        let out_path = args.image.with_file_name(format!("{stem}_overlay.jpg"));
        overlay.save_jpeg(out_path.to_str().unwrap(), 95)?;

        let elapsed = t_render.elapsed();
        println!("\n  Saved: {}", out_path.display());
        Some(elapsed)
    } else {
        None
    };

    // ── 11. Print timing ────────────────────────────────────────────
    println!("\n--- Timing ---");
    println!("  Load:        {}", fmt_ms(load_time));
    println!("  Preprocess:  {}", fmt_ms(preprocess_time));
    println!("  Inference:   {}", fmt_ms(inference_time));
    println!("  Postprocess: {}", fmt_ms(postprocess_time));
    let mut total = load_time + preprocess_time + inference_time + postprocess_time;
    if let Some(rt) = render_time {
        println!("  Render:      {}", fmt_ms(rt));
        total += rt;
    }
    println!("  Total:       {}", fmt_ms(total));

    Ok(())
}
