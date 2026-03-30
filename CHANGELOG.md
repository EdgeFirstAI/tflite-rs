# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-03-30

### Added

- `Delegate::xnnpack(&Library, num_threads)` for CPU-accelerated inference
  via the built-in XNNPACK delegate.
- `xnnpack_delegate(num_threads)` Python function for XNNPACK delegate
  creation.
- `XnnPackFunctions` and `TfLiteXNNPackDelegateOptions` in
  `edgefirst-tflite-sys` for runtime XNNPACK symbol loading.
- `tensorflowlite_c::library()` accessor for the underlying
  `libloading::Library`.
- `discover_with_path()` in `edgefirst-tflite-sys` discovery module,
  returning the loaded library path alongside the function table.
- `Library::reopen()` (crate-internal) for built-in delegate lifetime
  management via OS refcount.

### Changed

- Library paths are now canonicalised when they refer to existing files,
  making `Library::reopen()` resilient to working-directory changes.

## [0.3.0] - 2026-03-27

### Added

- HAL Delegate DMA-BUF API: standard 7-function C ABI for querying
  DMA-BUF tensor info and cache synchronization, loaded at runtime via
  `dlsym` from any compliant delegate `.so` (EDGEAI-1190).
- `DmaBuf::tensor_info(tensor_index)` returning `TensorInfo` with
  `size`, `offset`, `shape`, `fd`, and `dtype` fields.
- `DmaBuf::sync_for_device(tensor_index: i32)` and
  `DmaBuf::sync_for_cpu(tensor_index: i32)` using tensor index instead
  of opaque buffer handle.
- `CameraAdaptor::is_format_supported(format: &str) -> bool` and
  `CameraAdaptor::format_info(format: &str) -> Result<FormatInfo>`
  via the new `hal_camera_adaptor_*` standard functions.
- `TensorInfo`, `DType`, and `FormatInfo` public types in the
  `edgefirst-tflite` crate.
- `hal_ffi` module in `edgefirst-tflite-sys` with FFI bindings for
  `HalDmaBufFunctions` (5 functions) and `HalCameraAdaptorFunctions`
  (2 functions).
- `hal_to_result()` error helper for errno-based HAL error conversion.
- Python bindings for all new HAL methods on `DmaBuf` and
  `CameraAdaptor`, with `@deprecated` type-checker annotations on all
  legacy VxDelegate-only methods in `edgefirst_tflite.pyi`.
- Zero-copy DMA-BUF pipeline example (`dmabuf_zero_copy`) using the
  new tensor-index-based API (EDGEAI-1146).

### Changed

- `Delegate` now calls `hal_dmabuf_get_instance()` immediately after
  probing HAL symbols to obtain the true inner delegate handle;
  this handle is passed to `DmaBuf` and `CameraAdaptor` so all HAL
  calls use the correct pointer even when TFLite wraps the delegate
  in an `ExternalDelegate` adapter.
- `DmaBuf` and `CameraAdaptor` now prefer the HAL backend when
  available, with VxDelegate functions used as a fallback.

### Deprecated

- `DmaBuf::register`, `unregister`, `request`, `release`,
  `bind_to_tensor`, `fd` (renamed `buffer_fd`), `begin_cpu_access`,
  `end_cpu_access`, `set_active`, `active_buffer`, `invalidate_graph`,
  `is_graph_compiled`, `sync_for_device_by_handle`,
  `sync_for_cpu_by_handle` — VxDelegate-specific methods replaced by
  the HAL-standard API.
- `CameraAdaptor::set_format`, `set_format_ex`, `set_formats`,
  `set_fourcc`, `format`, `is_supported`, `input_channels`,
  `output_channels`, `fourcc`, `from_fourcc` — replaced by
  `is_format_supported` and `format_info`.

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

[Unreleased]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/EdgeFirstAI/tflite-rs/releases/tag/v0.1.0
