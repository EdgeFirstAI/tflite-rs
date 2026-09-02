# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Models that store constant buffers outside the flatbuffer (the ai-edge /
  `LiteRT` "offset buffer" format, used by standard Ultralytics int8 `TFLite`
  exports) now load and run through the C API.** Such a model references its
  large weight/bias constants by `Buffer.offset` rather than storing them
  inline; the `TFLite` C API does not resolve those, so inference aborted with
  "Input tensor N lacks data" even though the same model runs through the C++
  interpreter. The model loader now detects offset-stored buffers and inlines
  them in memory before handing the model to the runtime — no file, temp, or
  cache is written, and a model that already stores every buffer inline is
  loaded unchanged with no added work.

## [0.10.0] - 2026-09-01

### Added

- **Raw-byte tensor accessors: `Tensor::as_bytes`, `TensorMut::as_bytes`,
  `TensorMut::as_bytes_mut`, and `TensorMut::copy_from_bytes`.** These view or
  fill a tensor's whole data buffer as `u8`, spanning the full `byte_size`
  independent of element type — the correct way to move a preprocessed input
  or read an output whose element type is not `u8`. The classic-interpreter
  counterpart to LiteRT `TensorBuffer::write_bytes`/`read_bytes_into`.
- `TensorType::byte_width`, returning the element size in bytes (or `None`
  for the variable-width and sub-byte types).

### Fixed

- **`as_slice`/`as_mut_slice` now reject a type-argument whose size does not
  match the tensor's element width, instead of silently returning a
  partial-length slice.** `as_slice::<u8>()` on a `Float32` tensor previously
  returned a slice of `volume` (element-count) bytes — a quarter of the
  buffer — which truncated raw-byte copies of float32-I/O models to the first
  25% of every input and output. It now errors, pointing callers at
  `as_bytes`. Calls whose type already matched the element width (the
  quantized-I/O path, `u8`/`i8` tensors) are unaffected.

## [0.9.0] - 2026-08-02

### Added

- **LiteRT Next soft-optional bindings and ergonomic API.**
  `edgefirst-tflite-sys` vendors LiteRT v2.1.6 C headers and generates
  `LiteRt*` bindings alongside the existing `tensorflowlite_c` table. Symbols
  are probed individually via `LiteRtFunctions::try_load`, which never panics
  when absent and reports the *name* of the first unresolved symbol.
- `edgefirst-tflite::litert` module with RAII wrappers: `Environment`, `Model`,
  `Options` / `HwAccelerators`, `CompiledModel` (sync `run` +
  `is_fully_accelerated`), `TensorBuffer` / `BufferRequirements`, and
  accelerator enumeration (`accelerators`).
- `Library::litert()`, `Library::has_litert()`, and
  `Library::litert_missing_symbol()`, plus `Error::is_litert_unavailable()`,
  `Error::litert_status_code()`, and `Error::litert_missing_symbol()` for
  dual-runtime hosts. The named symbol distinguishes a classic TensorFlow Lite
  library from a partial or version-skewed LiteRT build.
- Compile-time ABI cross-check: the hand-written `LiteRtFunctions` signatures
  are asserted identical to the bindgen-generated table, so a header re-vendor
  that changes a signature becomes a build error rather than a silent ABI bug.
- `HwAccelerators` gains `Display`, `contains`, `is_empty`, and `BitOrAssign`;
  unrecognised accelerator bits round-trip and render as `unknown(0x…)`.
- `TensorBuffer::tensor_type()` and `TensorBuffer::read_bytes_into()`.
- `crates/tflite-sys/litert/vendor.sh` and `litert/patches/` make re-vendoring
  reproducible; local modifications to upstream headers are recorded as patches
  instead of being applied in place.
- `litert-compiled-model` example and LiteRT-gated integration tests.
- Classic TensorFlow Lite `Interpreter` / `Delegate` paths remain unchanged
  on libraries that do not export `LiteRt*`.
- **Zero-copy model input via TFLite custom allocations.**
  `Interpreter::set_custom_allocation_for_input` binds a caller-owned buffer as
  an input tensor's storage, so a GPU-resident buffer can be read by the
  runtime directly instead of copied into the arena every inference. The call
  is `unsafe` — the runtime keeps the raw pointer for its lifetime — but
  validates what it can first: input range, `bytes >= tensor.byte_size()`, and
  64-byte alignment (`kDefaultTensorAlignment`). The skip-alignment flag is
  deliberately not exposed; upstream documents it as a crash risk in
  `Invoke()`.
- `edgefirst_tflite_sys::experimental_ffi` binds the two `c_api_experimental.h`
  symbols this needs as a soft-optional table, resolved individually so a
  runtime without them still loads. `Library::has_custom_allocation` reports
  availability, and `Error::is_unsupported` / `Error::unsupported_api` name the
  missing entry point so callers can fall back instead of failing.

### Fixed

- `CompiledModel` now borrows the `Model` it was compiled from.
  `LiteRtCreateCompiledModel` does not take ownership and the runtime reads the
  model's flatbuffer on every inference, so dropping the model first was a
  use-after-free that segfaulted in practice. It is now a compile error.
- `BufferRequirements` now borrows its `CompiledModel`. The C API documents the
  returned requirements as owned by the compiled model and valid only during
  its lifetime; the previous `Copy` value could outlive it.
- `TensorBuffer` host mappings are released by an RAII guard, so an error or a
  panic between lock and unlock can no longer leave a buffer mapped.
- `TensorBuffer::size()` reports the size the runtime actually allocated
  (`LiteRtGetTensorBufferSize`) rather than the size requested, which correctly
  bounds every host mapping when a backend pads for alignment.
- `litert::accelerators()` validates the count reported by the C API before
  pre-allocating, and rejects a null accelerator handle.

### Changed

- `Options::get_hardware_accelerators` renamed to
  `Options::hardware_accelerator_set`.
- `TensorBuffer::read_bytes` takes `&mut self`, reflecting that host mapping
  mutates buffer state.
- `CompiledModel::create_input_buffer` / `create_output_buffer` no longer take a
  `&Model` argument — they use the model the compiled model already holds,
  removing any possibility of passing a mismatched one.
- `CompiledModel::run` reuses internal handle arrays, so a warm inference loop
  performs no allocation.
- `NOTICE` attributes the vendored LiteRT headers (Apache-2.0, Google LLC).
- `docs/superpowers/` is now gitignored: it holds local agent working notes,
  not project documentation.
- **`yolov8` example: `edgefirst-hal` 0.25 → 0.27.1.** `create_image` and
  `TensorDyn::image` now require a `CpuAccess` declaration; each buffer in the
  example declares what it actually does (`ReadWrite` for the decode target,
  `None` for the GPU-only working image, `Read` for buffers the host maps).
  Mis-declaring is not an error, only a silent slow path, so the choices are
  documented at each allocation. 0.27.1 includes the EGL dynamic-loader /
  iOS `libEGL` resolution fix and corrected `hal_import_image` C docs.
- The `yolov8` example now requests `TensorMemory::Dma` explicitly for its
  pipeline buffers — the HAL's portable name for a platform-native zero-copy
  GPU buffer (DMA-BUF on Linux, IOSurface on macOS/iOS, `AHardwareBuffer` on
  Android) — falling back to auto-selection, and prints the backend it actually
  obtained rather than assuming.
- The `yolov8` example binds its input buffer through the new custom-allocation
  API on Apple platforms, removing the per-frame arena copy. This is
  Apple-only for now: an `IOSurface` base address outlives the map guard, while
  the HAL's DMA-BUF `map()` is a per-map `mmap` whose address does not.
  Tracked upstream as [EdgeFirstAI/hal#134](https://github.com/EdgeFirstAI/hal/issues/134).
- The `yolov8` example builds on non-Linux hosts again. `ImageProcessor::import_image`
  is `#[cfg(target_os = "linux")]` in every `edgefirst-hal` release — importing
  a delegate-owned buffer by file descriptor is a DMA-BUF concept — so the two
  import sites go through a wrapper with a non-Linux stub, and the delegate
  probe no longer offers the import path off Linux.
- `LiteRtFunctions::try_load` no longer requires
  `LiteRtCreateTensorBufferFromHostMemory` (unused by the safe API); partial
  LiteRT builds that omit it are no longer reported as unavailable.
- `require_litert!()` skips without panicking if library discovery fails between
  the availability check and the LiteRT probe.
- Root README, `ARCHITECTURE.md`, and crate descriptions updated for the dual-
  runtime surface (LiteRT Next + custom allocations).
- Replaced the YOLOv8 test fixtures with models whose embedded `edgefirst.json`
  matches the current decoder schema, covering all three output layouts
  (`combined`, `logical`, `smart`) for both detection and segmentation, with
  i.MX 95 Neutron variants alongside the portable ones. These are now stored in
  Git LFS; `testdata/minimal.tflite` deliberately is not, since it is
  `include_bytes!`-ed into the test binaries.

## [0.8.0] - 2026-06-23

### Changed

- **Raised the minimum supported Rust version (MSRV) from 1.75 to 1.88.**
  Required by the `libloading` 0.9 and `zip` 8 upgrades, which both declare
  a 1.88 MSRV.
- Updated all workspace dependencies to their latest releases:
  - `pyo3` 0.24 → 0.29 and `numpy` 0.24 → 0.29 (Python bindings).
  - `libloading` 0.8 → 0.9 (runtime symbol loading in
    `edgefirst-tflite-sys` and `edgefirst-tflite`).
  - `zip` 2 → 8 (vendored TFLite extraction and the `archive` feature).
  - `ureq` 2 → 3 (vendored TFLite download in the `edgefirst-tflite-sys`
    build script).
  - `flatbuffers` 25.2 → 25.12 (the `metadata` feature).
  - `edgefirst-hal` 0.23.0 → 0.25.2 (the `yolov8` example).
- `edgefirst-tflite-sys`: regenerated FFI loader bound for `libloading` 0.9.
  `tensorflowlite_c::new` now requires `libloading::AsFilename` instead of
  `AsRef<OsStr>`; `update.sh` rewrites the bindgen output accordingly.
- `edgefirst-tflite-sys`: migrated the vendored TFLite downloader to the
  `ureq` 3 response/body API and switched the build-script TLS feature from
  the removed `tls` flag to `rustls`.
- Python bindings: replaced the deprecated `PyObject` alias with `Py<PyAny>`
  and opted `OpEvent` into the explicit `#[pyclass(from_py_object)]` derive
  to preserve its `FromPyObject` behaviour under `pyo3` 0.29.
- `yolov8` example: migrated to the `edgefirst-hal` 0.25 image-loading and
  letterbox API. Decoding no longer takes a `DecodeOptions` (images decode to
  their native pixel format), `ImageProcessor::import_image` takes an
  `Option<Colorimetry>` argument, and the letterbox `Crop` is expressed via
  the new `source`/`fit` fields instead of `dst_rect`.

### Fixed

- Resolved new Clippy lints surfaced by the Rust 1.96 toolchain
  (`borrow_as_ptr`, `ref_as_ptr`, `manual_c_str_literals`): raw-pointer FFI
  arguments now use `&raw const`/`&raw mut` and `std::ptr::from_ref`, and test
  C strings use `c"..."` literals.

## [0.7.0] - 2026-05-18

### Added

- `Send + Sync` trait implementations on `Library`, `Model`, and `Delegate`;
  `Send` (without `Sync`) on `Interpreter`. Enables multi-interpreter
  concurrent inference patterns with `std::thread::scope`.
- Compile-time `Send`/`Sync` assertions in `lib.rs` to prevent accidental
  regression of thread-safety contracts.
- `async_pipeline` example demonstrating a ring-buffer submit/wait pattern
  with fill → infer → read stages overlapping across threads.
- Integration tests for multi-interpreter correctness (4 threads × 50
  iterations).
- On-device tests verifying VxDelegate pipeline pattern (single interpreter
  moved between threads) and NeutronDelegate multi-interpreter concurrency.

### Changed

- `Delegate` `Send + Sync` implementation is no longer gated behind the
  `dmabuf` feature; it is now unconditional.
- Updated `edgefirst-hal` dependency from 0.21.0 to 0.23.0 in the `yolov8`
  example.

## [0.6.0] - 2026-05-08

### Added

- `archive` feature on `edgefirst-tflite` exposing `archive::ModelArchive`
  for reading the ZIP archive that the EdgeFirst tflite-converter appends
  to the FlatBuffer payload (`edgefirst.json`, `labels.txt`,
  `metadata.json`). Convenience helpers `archive::edgefirst_json()`,
  `archive::labels()`, and `archive::has_archive()` cover the common
  one-shot reads.
- Schema v2 fixtures in `testdata/`: `yolov8n-seg-combined-int8.tflite`,
  `yolov8n-seg-logical-int8.tflite`, `yolov8n-seg-smart-int8.tflite`
  exercising the fused, logical-split, and per-scale FPN-split decoder
  layouts respectively.
- Archive unit tests round-tripping all three schema v2 layouts and
  asserting layout-signature contract pins.
- Python bindings for the new `archive` API: `ModelArchive` class,
  `has_archive()` module helper, and `Interpreter.get_archive()`.
- Python bindings for the existing `Profiler` API (gap closure from
  0.5.0): `Profiler` and `OpEvent` classes, plus a `profiler=`
  keyword argument on `Interpreter`.

### Changed

- `yolov8` example builds the HAL `Decoder` from the model's embedded
  `edgefirst.json` (via `SchemaV2::parse_json` + `DecoderBuilder::with_schema`)
  instead of the prior shape-based heuristic. The same example now
  drives all three converter output layouts (combined / logical / smart)
  uniformly. Class labels are read from the embedded `labels.txt` rather
  than a hardcoded COCO list.
- `yolov8` example propagates each TFLite output tensor's
  `(scale, zero_point)` onto the corresponding HAL `TensorDyn` output
  buffer. The HAL per-scale decoder reads quantization from the tensor
  itself, so this attachment is required for the per-scale FPN-split
  ("smart") path.
- Updated `edgefirst-hal` dependency from 0.20 to 0.21.0 in the `yolov8`
  example.

## [0.5.1] - 2026-05-07

### Changed

- Updated `edgefirst-hal` dependency from 0.18 to 0.20 in the `yolov8`
  example. The bump is source-compatible: no API changes were required
  in example code.

## [0.5.0] - 2026-04-26

### Added

- `Profiler` for per-op telemetry timing via the TFLite
  `TfLiteInterpreterOptionsSetTelemetryProfiler` C API.
- `OpEvent` type capturing operator name, index, subgraph index, and
  duration in microseconds.
- `InterpreterBuilder::profiler(&Profiler)` to attach a profiler before
  building the interpreter.
- `Profiler::events()`, `drain_events()`, `clear()`, and `event_count()`
  for reading and managing collected profiling data.

### Changed

- Updated `edgefirst-hal` dependency from 0.14 to 0.18.

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

[Unreleased]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/EdgeFirstAI/tflite-rs/releases/tag/v0.1.0
