# Architecture

## Crate Structure

```
edgefirst-tflite          (safe API)
  └─ edgefirst-tflite-sys (FFI bindings)
       └─ libloading      (runtime symbol loading)
```

### `edgefirst-tflite-sys`

Low-level FFI plumbing. No safe wrappers.

- **`ffi.rs`** -- `bindgen`-generated struct with 164 function pointers,
  loaded at runtime via `libloading` (`--dynamic-loading`).
- **`discovery.rs`** -- Version-probing library search
  (`libtensorflow-lite.so.2.{49..1}.{9..0}` then unversioned fallbacks).
- **`vx_ffi.rs`** -- Function pointer structs for the `VxDelegate` DMA-BUF
  and `CameraAdaptor` C APIs. These are probed separately from the delegate
  `.so` because they are extension APIs not part of the standard TFLite C API.

### `edgefirst-tflite`

Ergonomic, safe Rust API. This is the primary user-facing crate.

- **`library.rs`** -- `Library` wraps the sys-level FFI handle.
- **`model.rs`** -- `Model` loads TFLite models from files or byte buffers.
- **`interpreter.rs`** -- `InterpreterBuilder` (builder pattern) and
  `Interpreter` for running inference.
- **`tensor.rs`** -- `Tensor` / `TensorMut` for type-safe tensor access.
- **`delegate.rs`** -- `Delegate` loads hardware acceleration delegates and
  probes for `VxDelegate` extensions.
- **`dmabuf.rs`** -- `DmaBuf` for zero-copy DMA-BUF operations (feature-gated).
- **`camera_adaptor.rs`** -- `CameraAdaptor` for NPU preprocessing (feature-gated).
- **`metadata.rs`** -- `Metadata` extraction from model files (feature-gated).

## FFI Binding Strategy

Symbols are resolved **at runtime**, not link-time. `bindgen` generates a
struct where each TFLite C API function is a field containing a function
pointer loaded via `libloading::Library::get()`. This allows the same binary
to work across different TFLite library versions and platforms.

## VxDelegate Extension Probing

When a `Delegate` is loaded, the crate probes the delegate `.so` for optional
VxDelegate symbols (e.g., `VxDelegateRegisterDmaBuf`,
`VxCameraAdaptorSetFormat`). If found, the function pointers are stored
alongside the delegate and exposed through `delegate.dmabuf()` and
`delegate.camera_adaptor()`.

## DMA-BUF Zero-Copy Data Flow

```
Camera (V4L2)                    NPU (TIM-VX)
  │                                  ▲
  │  DMA-BUF fd                      │  DMA-BUF fd
  ▼                                  │
┌──────────────────────────────────────┐
│             DMA-BUF                  │
│         (shared memory)              │
└──────────────────────────────────────┘
  │                                  ▲
  │  register + bind_to_tensor       │  sync_for_device
  ▼                                  │
┌──────────────────────────────────────┐
│         edgefirst-tflite             │
│    DmaBuf::register(fd, size, sync) │
│    DmaBuf::bind_to_tensor(h, idx)   │
│    Interpreter::invoke()             │
└──────────────────────────────────────┘
```

1. Camera produces frames into DMA-BUF buffers.
2. Application registers the buffer fd with `DmaBuf::register()`.
3. Buffer is bound to an input tensor with `DmaBuf::bind_to_tensor()`.
4. `sync_for_device()` ensures cache coherency before NPU access.
5. `Interpreter::invoke()` runs inference using the bound buffer directly.
6. `sync_for_cpu()` ensures output data is visible to the CPU.

No `memcpy` occurs between camera capture and NPU inference.

## CameraAdaptor NPU Preprocessing

The `CameraAdaptor` API injects preprocessing nodes into the TIM-VX graph:

```
Camera (RGBA) ──► CameraAdaptor (NPU) ──► Model Input (RGB 224x224)
                    │
                    ├── Format conversion (RGBA → RGB)
                    ├── Resize (optional)
                    └── Letterbox (optional)
```

These operations run on the NPU as part of the inference graph, avoiding CPU
preprocessing overhead.

## Error Handling

The crate uses a single `Error` type wrapping a private `ErrorKind` enum.
Callers inspect errors through methods rather than matching on variants:

- `is_library_error()` -- library loading or symbol resolution failed
- `is_delegate_error()` -- delegate returned an error status
- `is_null_pointer()` -- a C API call returned null
- `status_code()` -- returns the TFLite `StatusCode` if applicable

```
Error::status(code)        →  ErrorKind::Status(StatusCode)
Error::null_pointer(msg)   →  ErrorKind::NullPointer + context
Error::from(libloading)    →  ErrorKind::Library(libloading::Error)
Error::invalid_argument()  →  ErrorKind::InvalidArgument(String)
```

The `std::error::Error::source()` chain is preserved for library errors,
enabling upstream error inspection.

## Metadata Extraction

The `metadata` feature extracts human-readable metadata from TFLite model
files using embedded FlatBuffer structures:

```
model.tflite bytes
  │
  ├── root_as_model() ──► Model schema
  │     ├── description
  │     └── metadata[] ──► find "TFLITE_METADATA" buffer
  │           └── buffer_index
  │
  └── buffers[buffer_index]
        └── root_as_model_metadata() ──► ModelMetadata schema
              ├── name
              ├── version
              ├── description (merged with model description)
              ├── author
              ├── license
              └── min_parser_version
```

Models without a `TFLITE_METADATA` buffer return a `Metadata` struct with
all fields set to `None`.
