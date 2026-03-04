// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! # `YOLOv8` Object Detection with `edgefirst-tflite`
//!
//! End-to-end `YOLOv8` inference using `edgefirst-tflite` for model execution
//! and `edgefirst-hal` for image preprocessing, YOLO decoding, and overlay
//! rendering.
//!
//! Supports both i.MX8MP (`VxDelegate` with DMA-BUF zero-copy and
//! `CameraAdaptor`) and i.MX95 (Neutron NPU delegate).
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
//! yolov8 /opt/edgefirst/yolov8n_640x640.tflite image.jpg --delegate /usr/lib/libvx_delegate.so --save
//!
//! # i.MX95 with Neutron NPU
//! yolov8 /opt/edgefirst/yolov8n_640x640.imx95.tflite image.jpg --delegate /usr/lib/libneutron_delegate.so --save
//! ```

use std::path::PathBuf;
use std::time::Instant;

use edgefirst_hal::{
    decoder::{self, DetectBox, Nms},
    image::{
        Crop, Flip, ImageProcessor, ImageProcessorTrait as _, Rect, Rotation, TensorImage, RGB,
        RGBA,
    },
    tensor::{TensorMapTrait as _, TensorTrait as _},
};
use edgefirst_tflite::{Delegate, Interpreter, Library, Model, TensorType};
use ndarray::Array2;

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

// ── Output decoding ──────────────────────────────────────────────────────────

/// Dequantize a `TFLite` output tensor to f32.
#[allow(clippy::cast_precision_loss)]
fn dequantize_output(
    tensor: &edgefirst_tflite::Tensor<'_>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let params = tensor.quantization_params();
    match tensor.tensor_type() {
        TensorType::Float32 => Ok(tensor.as_slice::<f32>()?.to_vec()),
        TensorType::UInt8 => {
            let data = tensor.as_slice::<u8>()?;
            Ok(data
                .iter()
                .map(|&v| (f32::from(v) - params.zero_point as f32) * params.scale)
                .collect())
        }
        TensorType::Int8 => {
            let data = tensor.as_slice::<i8>()?;
            Ok(data
                .iter()
                .map(|&v| (f32::from(v) - params.zero_point as f32) * params.scale)
                .collect())
        }
        other => Err(format!("unsupported output tensor type: {other:?}").into()),
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    // ── 1. Load TFLite library (auto-discover) ──────────────────────────
    let lib = Library::new()?;

    // ── 2. Load model ───────────────────────────────────────────────────
    let model = Model::from_file(&lib, args.model.to_str().unwrap_or("model.tflite"))?;
    println!("Model: {}", args.model.display());

    // ── 3. Optionally load delegate and configure NPU features ──────────
    let mut use_camera_adaptor = false;
    let use_dmabuf;

    let delegate = if let Some(ref delegate_path) = args.delegate {
        let d = Delegate::load(delegate_path)?;
        println!("Delegate: {}", delegate_path.display());

        // Probe CameraAdaptor: set RGBA format before building interpreter.
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

    // ── 4. Build interpreter ────────────────────────────────────────────
    let mut builder = Interpreter::builder(&lib)?.num_threads(4);
    if let Some(d) = delegate {
        builder = builder.delegate(d);
    }
    let mut interpreter = builder.build(&model)?;

    println!("Inputs:  {}", interpreter.input_count());
    println!("Outputs: {}", interpreter.output_count());

    // ── 5. Inspect input tensor ─────────────────────────────────────────
    let (in_h, in_w, input_type, input_quant) = {
        let inputs = interpreter.inputs()?;
        let input = &inputs[0];
        let shape = input.shape()?;
        let tt = input.tensor_type();
        let qp = input.quantization_params();
        // Expect NHWC: [batch, height, width, channels]
        let h = shape[1];
        let w = shape[2];
        println!(
            "  input[0]: {} (scale={}, zero_point={})",
            input, qp.scale, qp.zero_point
        );
        (h, w, tt, qp)
    };

    // ── 6. Inspect output tensors ───────────────────────────────────────
    let output_count = interpreter.output_count();
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

    // ── 7. Load and preprocess image ────────────────────────────────────
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

    // ── 8. Write preprocessed pixels to input tensor / DMA-BUF ─────────
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

    // ── 10. Run inference ───────────────────────────────────────────────
    let t_inf = Instant::now();
    interpreter.invoke()?;
    let inference_time = t_inf.elapsed();

    // ── 11. Sync DMA-BUF back to CPU ────────────────────────────────────
    if let Some(handle) = dmabuf_handle {
        let delegate_ref = interpreter.delegate(0).expect("delegate not found");
        let dmabuf = delegate_ref.dmabuf().expect("DMA-BUF not available");
        dmabuf.sync_for_cpu(handle)?;
    }

    // ── 12. Read outputs and decode YOLO detections ─────────────────────
    let t_post = Instant::now();
    let mut detections: Vec<DetectBox> = Vec::with_capacity(100);

    let outputs = interpreter.outputs()?;

    #[allow(clippy::cast_precision_loss)]
    if output_count == 1 {
        // Single combined output: [1, 84, 8400] or [84, 8400]
        let tensor = &outputs[0];
        let shape = tensor.shape()?;
        let data = dequantize_output(tensor)?;

        // Strip batch dimension if present.
        let (rows, cols) = if shape.len() == 3 {
            (shape[1], shape[2])
        } else {
            (shape[0], shape[1])
        };

        // Auto-detect whether box values are in pixel space or already
        // normalized to [0,1]. Quantized models (e.g., i.MX95 Neutron) may
        // output already-normalized coordinates.
        let box_max = data[..4 * cols].iter().copied().fold(0.0f32, f32::max);
        let needs_norm = box_max > 2.0;

        let normalized = if needs_norm {
            let mut out = Vec::with_capacity(data.len());
            for row in 0..rows {
                for col in 0..cols {
                    let val = data[row * cols + col];
                    let norm = if row < 4 {
                        if row == 0 || row == 2 {
                            val / in_w as f32
                        } else {
                            val / in_h as f32
                        }
                    } else {
                        val
                    };
                    out.push(norm);
                }
            }
            out
        } else {
            data
        };

        let arr = Array2::from_shape_vec((rows, cols), normalized)?;
        decoder::yolo::decode_yolo_det_float(
            arr.view(),
            args.threshold,
            args.iou,
            Some(Nms::ClassAgnostic),
            &mut detections,
        );
    } else {
        // Split outputs: find boxes (shape[0]==4) and scores tensors
        let mut boxes_idx = None;
        let mut scores_idx = None;

        for (i, tensor) in outputs.iter().enumerate() {
            let shape = tensor.shape()?;
            if shape[0] == 4 {
                boxes_idx = Some(i);
            } else if scores_idx.is_none() {
                scores_idx = Some(i);
            }
        }

        let bi = boxes_idx.ok_or("cannot identify boxes output (shape[0]==4)")?;
        let si = scores_idx.ok_or("cannot identify scores output")?;

        let scores_shape = outputs[si].shape()?;
        let num_classes = scores_shape[0];
        let num_boxes = scores_shape[1];

        // Dequantize boxes and auto-detect pixel vs normalized coordinates.
        let boxes_raw = dequantize_output(&outputs[bi])?;
        let box_max = boxes_raw[..4 * num_boxes]
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let boxes_f32 = if box_max > 2.0 {
            // Pixel space: normalize per-axis to [0,1].
            let mut out = Vec::with_capacity(4 * num_boxes);
            for row in 0..4_usize {
                let div = if row == 0 || row == 2 {
                    in_w as f32
                } else {
                    in_h as f32
                };
                for col in 0..num_boxes {
                    out.push(boxes_raw[row * num_boxes + col] / div);
                }
            }
            out
        } else {
            boxes_raw[..4 * num_boxes].to_vec()
        };
        let boxes_arr = Array2::from_shape_vec((4, num_boxes), boxes_f32)?;

        let scores_f32 = dequantize_output(&outputs[si])?;
        let scores_arr = Array2::from_shape_vec(
            (num_classes, num_boxes),
            scores_f32[..num_classes * num_boxes].to_vec(),
        )?;

        decoder::yolo::decode_yolo_split_det_float(
            boxes_arr.view(),
            scores_arr.view(),
            args.threshold,
            args.iou,
            Some(Nms::ClassAgnostic),
            &mut detections,
        );
    }
    let postprocess_time = t_post.elapsed();

    // ── 13. Print results ───────────────────────────────────────────────
    println!("\n--- Timing ---");
    println!("  Preprocess:  {preprocess_time:?}");
    println!("  Inference:   {inference_time:?}");
    println!("  Postprocess: {postprocess_time:?}");
    println!(
        "  Total:       {:?}",
        preprocess_time + inference_time + postprocess_time
    );

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

    // ── 14. Save overlay ────────────────────────────────────────────────
    if args.save {
        let t_render = Instant::now();
        let mut overlay = TensorImage::load(&image_bytes, Some(RGBA), None)?;
        processor.render_to_image(&mut overlay, &detections, &[])?;

        let stem = args.image.file_stem().unwrap_or_default().to_string_lossy();
        let out_path = args.image.with_file_name(format!("{stem}_overlay.jpg"));
        overlay.save_jpeg(out_path.to_str().unwrap(), 95)?;

        println!("\n  Render:      {:?}", t_render.elapsed());
        println!("  Saved:       {}", out_path.display());
    }

    Ok(())
}
