# Architecture

## Crate Dependency Graph

```mermaid
graph TD
    examples["examples/*<br/>(basic_inference, dmabuf_zero_copy, ...)"]
    python["edgefirst-tflite<br/>(Python / PyO3)<br/>crates/python/"]
    tflite["edgefirst-tflite<br/>(safe Rust API)<br/>crates/tflite/"]
    sys["edgefirst-tflite-sys<br/>(FFI bindings)<br/>crates/tflite-sys/"]
    libloading["libloading"]
    pyo3["PyO3"]

    examples --> tflite
    python --> tflite
    python --> pyo3
    tflite --> sys
    sys --> libloading
```

## Crate Structure

```
edgefirst-tflite          (safe API)
  └─ edgefirst-tflite-sys (FFI bindings)
       └─ libloading      (runtime symbol loading)

edgefirst-tflite          (Python bindings, crates/python/)
  └─ edgefirst-tflite     (safe API, re-used)
  └─ pyo3
```

### `edgefirst-tflite-sys`

Low-level FFI plumbing. No safe wrappers.

- **`ffi.rs`** -- `bindgen`-generated struct with 164 function pointers,
  loaded at runtime via `libloading` (`--dynamic-loading`).
- **`discovery.rs`** -- Library discovery with 4-step priority chain (see below).
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

## Runtime Symbol Loading

Symbols are resolved **at runtime**, not link-time. `bindgen --dynamic-loading`
generates a `tensorflowlite_c` struct where each TFLite C API function is a
field containing a function pointer. The struct is instantiated with
`tensorflowlite_c::new(path)` (unsafe), which resolves all required symbols
at once via `libloading`.

This design means:
- No link-time dependency on a TFLite shared library — the binary compiles
  without TFLite installed.
- The same binary works across TFLite versions and platforms without
  recompilation.
- Missing or incompatible libraries produce clear runtime errors instead of
  silent linker failures.

`Library` in `crates/tflite/src/library.rs` wraps this struct and is the only
entry point. All call sites access TFLite symbols through `Library::as_sys()`.

## Library Discovery Mechanism

`discovery::discover()` in `crates/tflite-sys/src/discovery.rs` searches for
the TFLite shared library in 4 steps:

```mermaid
flowchart TD
    start([discover called])
    env{TFLITE_LIBRARY_PATH\nset?}
    vendored{vendored feature\nenabled?}
    versioned[Try versioned paths\nlibtensorflow-lite.so.2.N.P\nN=49..1, P=9..0]
    unversioned[Try libtensorflowlite_c.so]
    fallback[Try libtensorflow-lite.so]
    found([Return Ok - library loaded])
    err([Return Err - not found])

    start --> env
    env -- yes --> found
    env -- no --> vendored
    vendored -- yes, file exists --> found
    vendored -- no / file missing --> versioned
    versioned -- match --> found
    versioned -- no match --> unversioned
    unversioned -- match --> found
    unversioned -- no match --> fallback
    fallback -- match --> found
    fallback -- no match --> err
```

**Step 1 — `TFLITE_LIBRARY_PATH` env var:** If set, the library is loaded
from that exact path and no further probing occurs. Useful for CI, containers,
and testing alternative library versions.

**Step 2 — Vendored path:** When built with `--features vendored`, `build.rs`
sets the compile-time env var `EDGEFIRST_TFLITE_VENDORED_DIR` pointing to the
`OUT_DIR` where the downloaded library was extracted. `try_vendored()` reads
this via `option_env!()` and attempts to load it. If the file has been deleted
or the build is stale, this step silently skips to step 3.

**Step 3 — Versioned system paths:** Probes up to ~500 paths of the form
`libtensorflow-lite.so.2.N.P` (N=49..1, P=9..0) in descending version order.
Picks up the newest installed version. On an i.MX 8M Plus EVK this loop
completes in ~25 ms regardless of outcome.

**Step 4 — Unversioned fallbacks:** Tries `libtensorflowlite_c.so` then
`libtensorflow-lite.so` using the system's dynamic linker search path
(`LD_LIBRARY_PATH`, `/etc/ld.so.conf`, etc.).

## Vendored Feature Design

The `vendored` feature on `edgefirst-tflite-sys` enables automatic download of
a pre-built `libtensorflowlite_c` during `cargo build`. This follows the same
pattern as `openssl-sys`'s `vendored` feature.

### Build-time flow

```mermaid
sequenceDiagram
    participant cargo as cargo build
    participant build_rs as build.rs (vendored mod)
    participant github as GitHub Releases
    participant out_dir as OUT_DIR

    cargo->>build_rs: Invoke (feature=vendored)
    build_rs->>build_rs: Read TFLITE_VERSION env var<br/>(default: 2.19.0)
    build_rs->>build_rs: Determine platform artifact name
    alt Library already cached
        build_rs->>out_dir: Check if .so/.dylib/.dll exists
        build_rs->>cargo: Emit EDGEFIRST_TFLITE_VENDORED_DIR
    else First build / cache miss
        build_rs->>github: GET tflite-vX.Y.Z/libtensorflowlite_c-{platform}.tar.gz
        github-->>build_rs: Archive bytes
        build_rs->>out_dir: Extract library
        build_rs->>cargo: Emit EDGEFIRST_TFLITE_VENDORED_DIR
        build_rs->>cargo: cargo:warning= with library path
    end
```

### Runtime discovery (vendored)

`option_env!("EDGEFIRST_TFLITE_VENDORED_DIR")` is baked into the binary at
compile time. At runtime, `try_vendored()` constructs the platform-specific
library name, checks the path, and calls `load()`.

### Deployment note

The vendored library lives in Cargo's `OUT_DIR` (a build artifact directory).
`cargo run` and `cargo test` work transparently. For deployed binaries, copy
the library alongside the executable. The `cargo:warning=` message from
`build.rs` shows the exact source path.

### File layout

| File | Role |
|------|------|
| `crates/tflite-sys/build.rs` | Entry point; delegates to `vendored` mod when feature enabled |
| `crates/tflite-sys/vendored.rs` | Download + extract logic (`ureq`, `flate2`, `tar`, `zip`) |
| `crates/tflite-sys/src/discovery.rs` | `try_vendored()` + 4-step discovery chain |
| `.github/workflows/tflite.yml` | On-demand workflow that builds and publishes platform archives |

## Lifetime Model

Rust lifetimes enforce the correct teardown order without runtime reference
counting:

```
Library                 — owns the TFLite function table (loaded .so)
  └─ Model<'lib>        — borrows &Library; cannot outlive Library
  └─ Interpreter<'lib>  — borrows &Library; cannot outlive Library
       └─ Tensor<'interp>     — borrows &Interpreter; read-only view
       └─ TensorMut<'interp>  — borrows &mut Interpreter; mutable view
```

Each C object has a corresponding RAII wrapper with `impl Drop`:

| C object | Rust wrapper | Drop call |
|----------|-------------|-----------|
| `TfLiteModel*` | `Model` | `TfLiteModelDelete` |
| `TfLiteInterpreterOptions*` | `InterpreterBuilder` | `TfLiteInterpreterOptionsDelete` |
| `TfLiteInterpreter*` | `Interpreter` | `TfLiteInterpreterDelete` |
| `TfLiteDelegate*` | `Delegate` | `tflite_plugin_destroy_delegate` |

`Delegate` also owns the `libloading::Library` for the delegate `.so` as a
private field (`_lib`), keeping it resident for the delegate's full lifetime.

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

## Python Bindings Architecture

The Python package (`crates/python/`) wraps the Rust `edgefirst-tflite` API
using [PyO3](https://pyo3.rs) and is built with [maturin](https://maturin.rs).

### Lifetime Erasure in PyInterpreter

PyO3 `#[pyclass]` structs must be `'static` — they cannot hold Rust lifetime
parameters. The `Library → Model → Interpreter` ownership chain normally uses
lifetime parameters to enforce teardown order. The Python bindings erase these
lifetimes using a heap-pinned `InterpreterOwned` struct:

```
Pin<Box<Library>>          — heap-stable address, cannot be moved
  └─ Model<'static>        — 'static lifetime; safe because Library is pinned
  └─ Interpreter<'static>  — 'static lifetime; safe because Library is pinned
```

`InterpreterOwned::drop` manually drops `Interpreter` then `Model` before
`Library` (which is the last declared field and drops automatically). Field
declaration order in `InterpreterOwned` is load-bearing — do not reorder.

### Zero-Copy Tensor Views

`Interpreter.tensor(index)` returns a `_TensorAccessor` callable. Each call
to the accessor returns a NumPy array that shares memory directly with the
TFLite C-allocated tensor buffer (no copy). The accessor holds an
`allocation_generation` counter snapshot; if `allocate_tensors()` or
`resize_tensor_input()` is called, the generation increments and the accessor
raises an error on the next `__call__`, preventing stale pointer access.

### Module Layout

| File | Exported Python name |
|------|---------------------|
| `interpreter.rs` | `Interpreter`, `_TensorAccessor` |
| `delegate.rs` | `Delegate`, `load_delegate()` |
| `dmabuf.rs` | `DmaBuf` (returned by `Interpreter.dmabuf()`) |
| `camera_adaptor.rs` | `CameraAdaptor` (returned by `Delegate.camera_adaptor`) |
| `metadata.rs` | `ModelMetadata` (returned by `Interpreter.get_metadata()`) |
| `error.rs` | `TfLiteError`, `LibraryError`, `DelegateError`, `InvalidArgumentError` |

### Build Toolchain

```
maturin develop          — install editable into current venv
maturin build --release  — produce a .whl for distribution
```

The maturin feature `pyo3/extension-module` is set in `crates/python/pyproject.toml`.
The Rust module name exposed to Python is `edgefirst_tflite` (set via
`tool.maturin.module-name`).

## edgefirst-tflite-library Python Package

`crates/tflite-library/` is a **pure-Python** package (no Rust/C build step)
that ships a pre-built `libtensorflowlite_c` shared library as package data
inside a platform-tagged wheel.

```
edgefirst_tflite_library/
  __init__.py              — exposes library_path() -> str
  libtensorflowlite_c.so   — (Linux) placed here during wheel build
  libtensorflowlite_c.dylib — (macOS)
  tensorflowlite_c.dll      — (Windows)
```

The wheel is built by a separate CI workflow that:
1. Downloads the platform archive from the `tflite-vX.Y.Z` GitHub Release
   (produced by `.github/workflows/tflite.yml`).
2. Places the shared library inside the `edgefirst_tflite_library/` directory.
3. Builds and publishes a platform-tagged wheel to PyPI.

The package version matches the TFLite version it ships
(`edgefirst-tflite-library==2.19.0` ships TFLite 2.19.0).

`library_path()` returns the absolute filesystem path to the bundled library,
which users pass to `Interpreter(library_path=...)`. This is intentionally
explicit — automatic injection into `Interpreter.__init__` is a planned
follow-up.
