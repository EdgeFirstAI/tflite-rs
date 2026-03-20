# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-03-20

### Changed

- Python wheels now use PyO3 `abi3-py38` stable ABI, producing a single
  portable wheel per platform that works on Python 3.8+ instead of
  requiring a separate wheel per Python version.

## [0.2.0] - 2026-03-20

### Added

- Python bindings (`edgefirst-tflite` on PyPI) via PyO3 with
  `tflite_runtime.interpreter.Interpreter`-compatible API.
- Python `load_delegate()` function for hardware accelerator delegates.
- Python `DmaBuf`, `CameraAdaptor`, and `Metadata` extension classes.
- Python zero-copy tensor views via `Interpreter.tensor()`.
- YOLOv8 detection and segmentation example (Rust + Python) using
  `edgefirst-hal` 0.9 high-level Decoder API.
- Auto-detection of detection vs segmentation from output tensor shapes.
- Support for split output models (Neutron/onnx2tf format).
- `--warmup` and `--iters` benchmarking with min/max/avg/p95/p99 stats.
- `vendored` feature on `edgefirst-tflite-sys` for downloading pre-built
  TFLite C API from GitHub Releases at build time.
- `TFLITE_LIBRARY_PATH` environment variable for explicit library override.
- Platform-conditional library discovery (macOS `.dylib`, Windows `.dll`).
- `tflite.yml` GitHub Actions workflow for building TFLite C API shared
  libraries for Linux (x86_64, aarch64), macOS (arm64), and Windows.
- `edgefirst-tflite-library` Python package structure for shipping
  pre-built TFLite shared libraries via PyPI.
- YOLOv8 int8 test models for i.MX8MP (VxDelegate) and i.MX95 (Neutron).
- README.md, TESTING.md, and ARCHITECTURE.md documentation.

### Changed

- Updated `edgefirst-hal` dependency from 0.8 to 0.9.
- YOLOv8 example refactored from manual YOLO decoding to high-level
  `DecoderBuilder` API with `add_output()` for all model formats.
- Overlay rendering uses `draw_masks()` (renamed from `render_to_image()`).
- Image tensor allocation uses `ImageProcessor::create_image()` for
  optimal memory backend selection (DMA-buf > PBO > system memory).
- GitHub Actions updated to Node.js 24 (all actions pinned to latest).

## [0.1.0] - 2026-03-03

### Added

- `edgefirst-tflite-sys`: Low-level FFI bindings for the TFLite C API with
  runtime symbol loading via `libloading` (164 functions).
- `edgefirst-tflite-sys`: Library version probing (`libtensorflow-lite.so.2.x.y`).
- `edgefirst-tflite-sys`: `VxDelegate` DMA-BUF and `CameraAdaptor` function
  pointer structs with runtime probing.
- `edgefirst-tflite`: `Library` for auto-discovering and loading the TFLite
  shared library.
- `edgefirst-tflite`: `Model` for loading models from files or byte buffers.
- `edgefirst-tflite`: `Interpreter` with builder pattern, thread configuration,
  and delegate support.
- `edgefirst-tflite`: Type-safe `Tensor` / `TensorMut` with `as_slice()` and
  `copy_from_slice()`.
- `edgefirst-tflite`: `Delegate` loading with key-value options and VxDelegate
  extension probing.
- `edgefirst-tflite`: `DmaBuf` API for zero-copy inference with DMA-BUF file
  descriptors (`dmabuf` feature).
- `edgefirst-tflite`: `CameraAdaptor` API for NPU-accelerated format conversion
  (`camera_adaptor` feature).
- `edgefirst-tflite`: `Metadata` extraction from TFLite model files
  (`metadata` feature).

[Unreleased]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/EdgeFirstAI/tflite-rs/releases/tag/v0.1.0
