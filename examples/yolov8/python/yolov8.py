#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

"""YOLOv8 Object Detection & Segmentation with edgefirst-tflite + edgefirst-hal.

End-to-end YOLOv8 inference using edgefirst-tflite for model execution and
edgefirst-hal for image preprocessing, YOLO decoding (via high-level Decoder
API), and overlay rendering.

Usage:
    python yolov8.py <model.tflite> <image.jpg> [options]

Examples:
    # CPU-only detection
    python yolov8.py yolov8n.tflite image.jpg

    # Benchmark with 5 warmup + 100 iterations
    python yolov8.py model.tflite image.jpg --delegate /usr/lib/libvx_delegate.so \
        --warmup 5 --iters 100 --save

Requirements:
    pip install edgefirst-tflite edgefirst-hal numpy
"""

from __future__ import annotations

import argparse
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
    """Classify TFLite output tensors into edgefirst-hal Output objects."""
    from edgefirst_hal import Output

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
        is_quantized = det.get("dtype", "float32") != "float32"

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


def compute_letterbox(src_w, src_h, dst_w, dst_h):
    """Compute letterbox destination rect preserving aspect ratio."""
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    left = (dst_w - new_w) // 2
    top = (dst_h - new_h) // 2
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

    if n == 1:
        print(f"--- {label} ---")
        print(f"  Infer:   {timings['infer'][0]:.1f}ms")
        print(f"  Decode:  {timings['decode'][0]:.1f}ms")
        if timings["render"]:
            print(f"  Render:  {timings['render'][0]:.1f}ms")
        print(f"  Total:   {timings['total'][0]:.1f}ms")
        return

    print(f"--- {label} ({n} iterations) ---")
    print("               min      max      avg      p95      p99")
    for name in ("infer", "decode", "render", "total"):
        vals = timings[name]
        if not vals:
            continue
        s = sorted(vals)
        mn, mx = s[0], s[-1]
        avg = sum(s) / len(s)
        p95 = percentile(s, 0.95)
        p99 = percentile(s, 0.99)
        lbl = name.capitalize()
        print(f"  {lbl:<8} {mn:>7.1f} {mx:>8.1f} {avg:>8.1f} {p95:>8.1f} {p99:>8.1f} ms")


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
    in_shape = tuple(inp["shape"])
    in_h, in_w = int(in_shape[1]), int(in_shape[2])
    in_dtype = inp["dtype"]
    print(f"  input[0]: shape={list(in_shape)} dtype={in_dtype}")

    for i, det in enumerate(output_details):
        print(f"  output[{i}]: shape={list(det['shape'])} dtype={det['dtype']}")

    # ── 3. Auto-detect model type and build Decoder ──────────────────
    from edgefirst_hal import Decoder, DecoderVersion, Nms

    hal_outputs = classify_outputs(output_details)

    is_segmentation = any(len(d["shape"]) == 4 for d in output_details)
    print(f"  Mode: {'segmentation' if is_segmentation else 'detection'}")

    decoder = Decoder.new_from_outputs(
        hal_outputs,
        score_threshold=args.threshold,
        iou_threshold=args.iou,
        nms=Nms.ClassAgnostic,
        decoder_version=DecoderVersion.Yolov8,
    )

    # ── 4. Preprocess image (once) ───────────────────────────────────
    from edgefirst_hal import (
        TensorImage, ImageProcessor, PixelFormat, Rotation, Flip, Rect,
    )

    t_pre = time.perf_counter()

    src = TensorImage.load(args.image, PixelFormat.Rgba)
    img_w, img_h = src.width, src.height
    print(f"Image: {img_w}x{img_h}")

    dst = TensorImage(in_w, in_h, PixelFormat.Rgb)
    processor = ImageProcessor()

    left, top, new_w, new_h = compute_letterbox(img_w, img_h, in_w, in_h)
    dst_crop = Rect(left, top, new_w, new_h)
    processor.convert(
        src, dst,
        rotation=Rotation.Rotate0,
        flip=Flip.NoFlip,
        dst_crop=dst_crop,
        dst_color=[114, 114, 114, 255],
    )

    # Write preprocessed pixels to input tensor (once).
    input_array = np.zeros((in_h, in_w, 3), dtype=np.uint8)
    dst.normalize_to_numpy(input_array)

    if in_dtype == "float32":
        input_data = input_array.astype(np.float32) / 255.0
    elif in_dtype == "int8":
        input_data = (input_array.astype(np.int16) - 128).astype(np.int8)
    else:
        input_data = input_array

    interpreter.set_tensor(0, input_data)
    preprocess_time = (time.perf_counter() - t_pre) * 1000

    # Pre-allocate overlay for rendering (reused across iterations).
    overlay = TensorImage.load(args.image, PixelFormat.Rgba) if args.save else None

    # ── 5. Iteration runner ──────────────────────────────────────────
    def run_iterations(n):
        timings = {"infer": [], "decode": [], "render": [], "total": []}
        boxes = scores = classes = masks = None

        for _ in range(n):
            t_total = time.perf_counter()

            # Infer
            t0 = time.perf_counter()
            interpreter.invoke()
            timings["infer"].append((time.perf_counter() - t0) * 1000)

            # Decode
            t0 = time.perf_counter()
            model_outputs = [
                interpreter.get_output_tensor(i)
                for i in range(len(output_details))
            ]
            boxes, scores, classes, masks = decoder.decode(model_outputs)

            # Normalize if in pixel space
            if len(scores) > 0 and np.max(boxes) > 2.0:
                boxes[:, [0, 2]] /= in_w
                boxes[:, [1, 3]] /= in_h

            timings["decode"].append((time.perf_counter() - t0) * 1000)

            # Render (if --save)
            if overlay is not None:
                t0 = time.perf_counter()
                processor.draw_masks(overlay, boxes, scores, classes, masks)
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
    boxes, scores, classes, masks, bench_timings = run_iterations(args.iters)

    # ── 8. Print detections (from last iteration) ────────────────────
    num_detections = len(scores) if scores is not None else 0
    print(f"\n--- Detections ({num_detections}) ---")
    for i in range(num_detections):
        label = int(classes[i])
        name = COCO[label] if label < len(COCO) else "?"
        score = float(scores[i]) * 100.0
        x1 = max(0.0, float(boxes[i, 0]) * img_w)
        y1 = max(0.0, float(boxes[i, 1]) * img_h)
        x2 = min(float(img_w), float(boxes[i, 2]) * img_w)
        y2 = min(float(img_h), float(boxes[i, 3]) * img_h)
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
    print(f"  Load:       {load_time:.1f}ms")
    print(f"  Preprocess: {preprocess_time:.1f}ms")
    print()
    print_stats(f"Benchmark (iters={args.iters})", bench_timings)


if __name__ == "__main__":
    main()
