# Testing Guide

This guide describes how to test `edgefirst-tflite` on NXP FRDM development
boards. Unit tests run on any host machine; integration tests and examples
require on-target hardware with a TensorFlow Lite runtime.

## Host Unit Tests

Unit tests exercise pure Rust logic (type conversions, error handling, metadata
parsing) without requiring a TFLite shared library.

```sh
cargo test --workspace --all-features
```

## Pre-Built Models

Pre-converted YOLOv8n models are available for download:

| Model | Target | URL |
|-------|--------|-----|
| YOLOv8n 640×640 (TFLite) | i.MX 8M Plus | <https://repo.edgefirst.ai/models/yolov8/yolov8n_640x640.tflite> |
| YOLOv8n 640×640 (Neutron) | i.MX 95 | <https://repo.edgefirst.ai/models/yolov8/yolov8n_640x640.imx95.tflite> |

Download both to `/opt/edgefirst/` on the target device (or any path you
prefer).

## Cross-Compilation

All workspace members, including examples, are cross-compiled for
`aarch64-unknown-linux-gnu` using `cargo-zigbuild`:

```sh
# Install zigbuild (once)
cargo install cargo-zigbuild

# Build everything
cargo zigbuild --workspace --all-features --target aarch64-unknown-linux-gnu

# Build test binaries
cargo zigbuild --workspace --all-features --target aarch64-unknown-linux-gnu --tests
```

Binaries are located in `target/aarch64-unknown-linux-gnu/debug/`.

## On-Target Testing: NXP FRDM i.MX 8M Plus

The i.MX 8M Plus uses the **VxDelegate** with the Vivante NPU. It supports
DMA-BUF zero-copy inference and CameraAdaptor RGBA→RGB preprocessing.

### Prerequisites

- FRDM i.MX 8M Plus board running an EdgeFirst or NXP BSP image
- `/usr/lib/libtensorflowlite_c.so` and `/usr/lib/libvx_delegate.so` installed
- A test image (JPEG or PNG) copied to the device

### Deploy and Run

```sh
# Copy binary and test image to the device
scp target/aarch64-unknown-linux-gnu/debug/yolov8 root@<imx8mp-ip>:/tmp/
scp test_image.jpg root@<imx8mp-ip>:/tmp/

# Run with VxDelegate (uses DMA-BUF + CameraAdaptor automatically)
ssh root@<imx8mp-ip> '/tmp/yolov8 \
    /opt/edgefirst/yolov8n_640x640.tflite \
    /tmp/test_image.jpg \
    --delegate /usr/lib/libvx_delegate.so \
    --save'
```

The `--save` flag writes a `test_image_overlay.jpg` with bounding box overlays.

### Integration Tests

```sh
# Copy test binary and testdata to the device
scp target/aarch64-unknown-linux-gnu/debug/deps/edgefirst_tflite-* root@<imx8mp-ip>:/tmp/
scp -r testdata root@<imx8mp-ip>:/tmp/

# Run integration tests
ssh root@<imx8mp-ip> 'cd /tmp && ./edgefirst_tflite-*'
```

Set `TFLITE_TEST_LIB=/path/to/libtensorflowlite_c.so` if the library is not in
the default search paths.

## On-Target Testing: NXP FRDM i.MX 95

The i.MX 95 uses the **Neutron NPU** delegate. It does not support DMA-BUF or
CameraAdaptor — the example falls back to standard tensor I/O automatically.

### Prerequisites

- FRDM i.MX 95 board running an EdgeFirst or NXP BSP image
- `/usr/lib/libtensorflowlite_c.so` and `/usr/lib/libneutron_delegate.so`
  installed
- A test image (JPEG or PNG) copied to the device

### Model Conversion for Neutron NPU

Standard TFLite models must be converted for the Neutron NPU using the
**eIQ Neutron Converter** tool from NXP. The converter is part of the
[eIQ Toolkit for End-to-End Model Development and Deployment][eiq-toolkit].

Pre-converted models are available at the URLs listed above. To convert your
own models, refer to the eIQ Toolkit documentation.

[eiq-toolkit]: https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT

### Deploy and Run

```sh
# Copy binary and test image to the device
scp target/aarch64-unknown-linux-gnu/debug/yolov8 root@<imx95-ip>:/tmp/
scp test_image.jpg root@<imx95-ip>:/tmp/

# Run with Neutron delegate
ssh root@<imx95-ip> '/tmp/yolov8 \
    /opt/edgefirst/yolov8n_640x640.imx95.tflite \
    /tmp/test_image.jpg \
    --delegate /usr/lib/libneutron_delegate.so \
    --save'
```

### Integration Tests

```sh
# Copy test binary and testdata to the device
scp target/aarch64-unknown-linux-gnu/debug/deps/edgefirst_tflite-* root@<imx95-ip>:/tmp/
scp -r testdata root@<imx95-ip>:/tmp/

# Run integration tests
ssh root@<imx95-ip> 'cd /tmp && ./edgefirst_tflite-*'
```

## Expected Output

A successful YOLOv8 run prints detected objects with class, confidence, and
bounding box coordinates:

```
Model: /opt/edgefirst/yolov8n_640x640.tflite
Input: 640x640, UInt8, 3 channels
Outputs: 1

Timing:
  Model load:      12.3 ms
  Preprocessing:    5.1 ms
  Inference:        8.7 ms
  Post-processing:  1.2 ms

Detections (threshold=0.25, IoU=0.45):
  dog (16): 77.2% [491, 448, 991, 1074]
  dog (16): 56.1% [959, 418, 1443, 1079]
```

Detection coordinates are in the original image pixel space (not the 640×640
model input space).
