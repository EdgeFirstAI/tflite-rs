# AI Assistant Instructions — edgefirst-tflite

This file provides project-level guidance for AI assistants (GitHub Copilot,
Claude Code, etc.) working on this repository. Follow these instructions in
addition to any global or user-level configuration.

---

## Project Overview

**edgefirst-tflite** provides ergonomic Rust bindings for the TensorFlow Lite
C API (and soft-optional LiteRT Next), designed for edge AI inference on NXP
i.MX platforms and Android LiteRT hosts.

Key characteristics:

- **Runtime symbol loading** — TFLite is loaded at runtime via `libloading`.
  There is no link-time dependency on a TFLite shared library.
- **Dual runtime** — classic `Interpreter` always works on TFLite-only `.so`
  files; `litert::CompiledModel` engages when `LiteRt*` symbols resolve.
  `Library::litert_missing_symbol()` names the first unresolved symbol, which
  separates "classic TFLite" from "partial LiteRT build".
- **DMA-BUF zero-copy inference** — the `dmabuf` feature enables zero-copy
  tensor hand-off from camera or DMA buffers directly to the NPU.
- **NPU-accelerated preprocessing** — the `camera_adaptor` feature exposes
  VxDelegate's `CameraAdaptor` API for format conversion on the NPU.
- **Target platforms** — `aarch64-unknown-linux-gnu`, specifically i.MX8M Plus,
  i.MX93, and i.MX95.

---

## Repository Layout

```
edgefirst-tflite/
├── Cargo.toml                  # Workspace manifest (resolver v2, edition 2021)
├── rustfmt.toml                # Workspace-wide rustfmt config (edition = "2021")
├── crates/
│   ├── tflite/                 # edgefirst-tflite  — safe, idiomatic Rust API
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── library.rs      # Library: wraps tflite-sys, probes LiteRT
│   │       ├── model.rs        # Model: loads a .tflite file or buffer
│   │       ├── interpreter.rs  # Interpreter + InterpreterBuilder (builder pattern)
│   │       ├── litert/         # LiteRT Next CompiledModel path (soft-optional)
│   │       ├── delegate.rs     # Delegate + DelegateOptions (external delegate)
│   │       ├── tensor.rs       # Tensor / TensorMut / TensorType / QuantizationParams
│   │       ├── error.rs        # Error / StatusCode / LiteRtStatusCode / Result
│   │       ├── dmabuf.rs       # [feature = "dmabuf"]  DMA-BUF zero-copy API
│   │       ├── camera_adaptor.rs  # [feature = "camera_adaptor"]  NPU preprocessing
│   │       └── metadata.rs     # [feature = "metadata"]  Model metadata extraction
│   └── tflite-sys/             # edgefirst-tflite-sys  — raw FFI bindings
│       └── src/
│           ├── lib.rs
│           ├── ffi.rs          # bindgen-generated TFLite C API (included via include!)
│           ├── litert_ffi.rs   # bindgen-generated LiteRT C API (types + ABI check)
│           ├── litert.rs       # LiteRtFunctions::try_load + ABI cross-check
│           ├── litert/         # vendored LiteRT headers, patches, vendor.sh
│           ├── vx_ffi.rs       # VxDelegate DMA-BUF and CameraAdaptor symbols
│           └── discovery.rs    # discover() / load() — library search and probing
├── examples/                   # Standalone example binaries
│   ├── basic_inference/        # cargo run -p basic-inference -- model.tflite
│   ├── litert_compiled_model/  # cargo run -p litert-compiled-model -- model.tflite
│   └── dmabuf_zero_copy/       # cargo run -p dmabuf-zero-copy -- model.tflite
└── testdata/
    └── minimal.tflite          # Minimal valid TFLite flatbuffer for integration tests
```

---

## Workspace Configuration

| Setting | Value |
|---------|-------|
| Rust edition | 2021 |
| MSRV | 1.88 |
| Resolver | v2 |
| License | Apache-2.0 |
| Authors | Au-Zone Technologies `<support@au-zone.com>` |
| Repository | `https://github.com/EdgeFirstAI/tflitec-sys` |

---

## Feature Flags

| Feature | Description | Modules enabled |
|---------|-------------|-----------------|
| `dmabuf` | DMA-BUF zero-copy inference via VxDelegate | `dmabuf.rs` |
| `camera_adaptor` | NPU-accelerated format conversion | `camera_adaptor.rs` |
| `metadata` | TFLite model metadata extraction | `metadata.rs` |
| `full` | Enables all optional features | all of the above |

Default features: none (base crate has no optional features enabled).

The `metadata` feature pulls in `flatbuffers`, `zip`, and `yaml-rust2`.

---

## Code Style

### File Headers

Every source file — Rust (`.rs`) and TOML (`.toml`) — **must** begin with
SPDX license headers. No exceptions.

Rust files:

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.
```

TOML files:

```toml
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.
```

### Formatting

- `rustfmt` with the workspace `rustfmt.toml` (`edition = "2021"`).
- Run `cargo fmt --all` before committing.
- Format check: `cargo fmt --all -- --check`.

### Lints

Workspace lints are defined in the root `Cargo.toml`:

```toml
[workspace.lints.rust]
missing_debug_implementations = "warn"
unsafe_op_in_unsafe_fn = "warn"

[workspace.lints.clippy]
cargo    = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
missing_errors_doc = "allow"
module_name_repetitions = "allow"
```

All crates inherit these via `[lints] workspace = true`.

Generated FFI code in `ffi.rs` suppresses lints with a blanket `#[allow(...)]`
block — do not apply workspace lints inside the generated file.

### Unsafe Code

- All `unsafe` blocks must have a `// SAFETY:` comment explaining the invariant
  being upheld.
- Unsafe operations inside `unsafe fn` must still use `unsafe {}` blocks; this
  is enforced by the `unsafe_op_in_unsafe_fn` lint.

---

## Architecture Patterns

### Runtime Symbol Loading

`edgefirst-tflite-sys` uses `bindgen --dynamic-loading` to generate a
`tensorflowlite_c` struct whose fields are function pointers. The struct is
loaded with `tensorflowlite_c::new(path)` (unsafe), which resolves all required
symbols at once.

`Library` in `crates/tflite/src/library.rs` wraps this struct and exposes it
via `Library::as_sys()`. All call sites go through `Library::as_sys()` — never
use the sys crate's structs directly in the high-level crate.

### Optional Symbol Tables (LiteRT, XNNPACK, experimental C API)

Symbols that may or may not exist in the loaded library are **not** resolved by
a bindgen `--dynamic-loading` struct, which fails wholesale if any one symbol is
missing. They use a hand-written function-pointer struct with a `try_load`
constructor instead — `XnnPackFunctions`, `LiteRtFunctions`, and
`ExperimentalFunctions` all follow this pattern.

This is also why `c_api_experimental.h` stays out of `wrapper.h`: including it
would put every experimental symbol in the eagerly-resolved bindgen table, so a
runtime missing any one of them would fail to load at all. Bind what you need in
`crates/tflite-sys/src/experimental_ffi.rs` and let the safe layer name the
unavailable API (`Error::is_unsupported` / `Error::unsupported_api`) so callers
can fall back instead of failing.

For LiteRT the table is declared by the `litert_functions!` macro in
`crates/tflite-sys/src/litert.rs`. One entry declares the field name, the C
symbol name, and the signature; the macro derives the struct, `try_load`, and a
compile-time cross-check against the bindgen-generated table. Two rules follow:

- **Add symbols to the macro, never to the struct directly.** Anything else
  bypasses the ABI cross-check and the missing-symbol reporting.
- **Never collapse a resolution failure into a bare `Option`.** `try_load`
  returns the name of the first unresolved symbol so callers can tell "classic
  TFLite" from "partial LiteRT build". Preserve that through to the `Error`.

### Vendored Headers

`crates/tflite-sys/litert/` holds upstream LiteRT headers (Apache-2.0, Google
LLC), pinned by `litert/VERSION` and attributed in the top-level `NOTICE`.

**Never edit the vendored headers in place.** Add a patch under
`litert/patches/` and re-run `litert/vendor.sh`, which reapplies patches after
fetching upstream. An in-place edit is silently lost at the next re-vendor.

### Borrowed C Handles

Several LiteRT C objects are views into a parent, valid only for the parent's
lifetime — a compiled model's buffer requirements, a model's signatures, an
accelerator's name. When wrapping one, give the wrapper a lifetime parameter
naming the parent (using `PhantomData` if there is no field to carry it) rather
than storing a bare pointer. Both currently wrapped cases are guarded by
`compile_fail` doctests; keep that pattern for new ones.

### Builder Pattern

`Interpreter` is created through `InterpreterBuilder`:

```rust
let mut interpreter = Interpreter::builder(&lib)?
    .num_threads(4)
    .delegate(delegate)
    .build(&model)?;
```

`InterpreterBuilder::build` also calls `TfLiteInterpreterAllocateTensors`
before returning. The caller receives a fully allocated interpreter.

### Lifetime Parameters

Lifetime parameters tie resources to their owner:

- `Model<'lib>` — borrows `&Library`; cannot outlive the library.
- `Interpreter<'lib>` — borrows `&Library`; cannot outlive the library.
- `Tensor<'interp>` / `TensorMut<'interp>` — borrow the interpreter; cannot
  outlive the interpreter.

Do not introduce owned copies of C pointers to circumvent these lifetimes.

### RAII for C Resources

Every C object has a corresponding Rust wrapper with `impl Drop`:

| C object | Rust wrapper | Drop call |
|----------|-------------|-----------|
| `TfLiteModel*` | `Model` | `TfLiteModelDelete` |
| `TfLiteInterpreterOptions*` | `InterpreterBuilder` | `TfLiteInterpreterOptionsDelete` |
| `TfLiteInterpreter*` | `Interpreter` | `TfLiteInterpreterDelete` |
| `TfLiteDelegate*` | `Delegate` | `tflite_plugin_destroy_delegate` |

`Delegate` also owns the `libloading::Library` that the delegate `.so` was
loaded from (`_lib`), keeping the library resident for the delegate's lifetime.

### Error Handling

```rust
// Public type — always use this in function signatures.
pub struct Error { kind: ErrorKind, context: Option<String> }

// Internal enum — never expose variants to callers.
enum ErrorKind { Status(StatusCode), NullPointer, Library(libloading::Error), InvalidArgument(String) }
```

Callers classify errors via inspection methods, not variant matching:

```rust
err.is_library_error()    // library loading / symbol resolution failed
err.is_delegate_error()   // TFLite returned a delegate-related status code
err.is_null_pointer()     // a C API call returned null
err.status_code()         // -> Option<StatusCode>
```

Always attach context with `.with_context("...")` at the call site:

```rust
error::status_to_result(status)
    .map_err(|e| e.with_context("TfLiteInterpreterAllocateTensors"))?;
```

### VxDelegate Probing

When `Delegate::load_with_options` loads a delegate `.so`, it immediately
probes for optional VxDelegate extension symbols using `try_load` on
`VxDmaBufFunctions` and `VxCameraAdaptorFunctions`. These are stored as
`Option<T>` fields (feature-gated). If found, they are exposed via:

- `delegate.dmabuf()` — returns `Option<DmaBuf<'_>>` (feature `dmabuf`)
- `delegate.camera_adaptor()` — returns `Option<CameraAdaptor<'_>>` (feature `camera_adaptor`)

### Thread Safety

The TFLite C API has no thread affinity. Core types implement `Send`/`Sync`:

| Type | `Send` | `Sync` | Notes |
|------|--------|--------|-------|
| `Library` | ✅ | ✅ | Immutable function table |
| `Model<'lib>` | ✅ | ✅ | Read-only after creation |
| `Interpreter<'lib>` | ✅ | ❌ | Mutable state; exclusive ownership only |
| `Delegate` | ✅ | ✅ | Opaque handle |

Multiple interpreters can be created from one model for concurrent inference.
Each interpreter has independent tensor state. Use `std::thread::scope` with
the `'lib` lifetime. See `examples/async_pipeline/` for a pipelined
submit/wait pattern.

---

## Build and Test Commands

### Local (x86_64 host)

```sh
# Format check
cargo fmt --all -- --check

# Lint (base features)
cargo clippy --workspace -- -D warnings

# Lint (all optional features)
cargo clippy --workspace --features full -- -D warnings

# Documentation
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

# Unit tests (pure Rust, no FFI — run anywhere)
cargo test --workspace --all-features
```

### Cross-compile for i.MX (aarch64)

```sh
# Build all workspace members
cargo zigbuild --workspace --all-features --target aarch64-unknown-linux-gnu

# Build test binaries
cargo zigbuild --workspace --all-features --target aarch64-unknown-linux-gnu --tests
```

### On-device Integration Testing

1. Cross-compile test binaries:
   ```sh
   cargo zigbuild --workspace --all-features --target aarch64-unknown-linux-gnu --tests
   ```
2. Locate test binaries in `target/aarch64-unknown-linux-gnu/debug/deps/`.
3. SCP test binaries and the `testdata/` directory to the device.
4. SSH into the device and run the test binary directly.
5. Set `TFLITE_TEST_LIB=/path/to/libtensorflowlite_c.so` if the library is not
   in the default search paths.

Integration tests auto-discover TFLite using `discovery::discover()`. DMA-BUF
and `CameraAdaptor` tests only pass on devices with VxDelegate and NPU
hardware.

---

## How to Add Examples

1. Create `examples/<name>/Cargo.toml`:
   ```toml
   # SPDX-License-Identifier: Apache-2.0
   # Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

   [package]
   name = "<name>"
   version.workspace = true
   edition.workspace = true
   rust-version.workspace = true
   license.workspace = true
   publish = false

   [dependencies]
   edgefirst-tflite = { path = "../../crates/tflite", features = [...] }
   ```

2. Create `examples/<name>/src/main.rs` with:
   - SPDX header comment block
   - Module doc comment including the `cargo run -p <name>` invocation
   - `fn main() -> Result<(), Box<dyn std::error::Error>>`

3. The example is auto-included via `members = ["examples/*"]` in the workspace
   `Cargo.toml`. No changes to the workspace manifest are needed.

---

## How to Add Feature-Gated Modules

1. Add the feature to `crates/tflite/Cargo.toml` `[features]`:
   ```toml
   my_feature = ["dep:some-crate"]
   full = ["metadata", "dmabuf", "camera_adaptor", "my_feature"]
   ```

2. Create `crates/tflite/src/my_feature.rs` with SPDX header and module doc.

3. Add to `crates/tflite/src/lib.rs`:
   ```rust
   #[cfg(feature = "my_feature")]
   pub mod my_feature;
   ```

4. Add re-exports to `lib.rs` if the feature exposes public types.

5. Update the feature flags table in the module-level doc comment in `lib.rs`
   and in this file.

---

## Release Checklist

When preparing a release, the following files must be updated manually:

1. **`Cargo.toml`** (workspace root):
   - Bump `[workspace.package] version`
   - Bump `[workspace.dependencies]` versions for `edgefirst-tflite-sys` and
     `edgefirst-tflite` to match

2. **`CHANGELOG.md`**:
   - Move items from `[Unreleased]` into a new `[X.Y.Z] - YYYY-MM-DD` section
   - Add a fresh empty `[Unreleased]` heading above it
   - Add a comparison link at the bottom:
     `[X.Y.Z]: https://github.com/EdgeFirstAI/tflite-rs/compare/vPREV...vX.Y.Z`
   - Update the `[Unreleased]` comparison link to point from the new tag

3. **Commit and tag**:
   ```sh
   git commit -s -m "Release vX.Y.Z"
   git tag -s vX.Y.Z -m "vX.Y.Z"
   git push && git push --tags
   ```

4. The `release.yml` GitHub Actions workflow handles crates.io publishing
   automatically when a `v*` tag is pushed.

---

## Commit and Branch Conventions

Branch naming: `(feature|bugfix|chore)/EFT-###-short-description`

Commit format: `EFT-###: Brief imperative description`

Example: `EFT-42: Add metadata feature for FlatBuffers model info extraction`

All commits must be signed (`git commit -s`).

---

## License Compatibility

This project is licensed under Apache-2.0. All dependencies must use
permissive licenses (Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC).

Do not introduce dependencies with GPL, LGPL, AGPL, SSPL, or other
copyleft licenses.

---

## Contact and Support

- Support email: support@au-zone.com
- Homepage: https://au-zone.com
- EdgeFirst ecosystem: https://edgefirst.ai
