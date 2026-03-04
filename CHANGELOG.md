# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/EdgeFirstAI/tflite-rs/releases/tag/v0.1.0
