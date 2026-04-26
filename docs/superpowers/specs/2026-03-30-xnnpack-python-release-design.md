# XNNPACK Python Binding & v0.4.0 Release

**Date:** 2026-03-30
**Status:** Draft
**Version:** 0.3.0 -> 0.4.0

## Context

The Rust API gained `Delegate::xnnpack(&lib, num_threads)` for CPU-accelerated
inference via the XNNPACK built-in delegate. The Python bindings do not expose
this yet. This spec covers the Python binding, documentation updates, and the
0.4.0 release.

## 1. Python Binding

### New function: `xnnpack_delegate(num_threads=0)`

Module-level function in `edgefirst_tflite`, alongside `load_delegate()`.

**Signature:**

```python
def xnnpack_delegate(num_threads: int = 0) -> Delegate:
    """Create an XNNPACK delegate for CPU-accelerated inference.

    XNNPACK optimises floating-point and quantised operations on ARM and
    x86 CPUs using SIMD instructions.

    Args:
        num_threads: XNNPACK threadpool size. 0 lets XNNPACK choose.

    Returns:
        A Delegate to pass to Interpreter(experimental_delegates=[...]).

    Raises:
        InvalidArgumentError: If the TFLite library lacks XNNPACK support.
    """
```

**Usage:**

```python
from edgefirst_tflite import Interpreter, xnnpack_delegate

delegate = xnnpack_delegate(num_threads=4)
interp = Interpreter(
    model_path="model.tflite",
    experimental_delegates=[delegate],
)
```

**Implementation (Rust):**

- `Library::new()` to auto-discover TFLite
- `Delegate::xnnpack(&lib, num_threads)` to create the delegate
- Wrap in `PyDelegate { inner: Some(delegate) }`
- Errors via `error::to_py_err` (maps to `InvalidArgumentError` for missing
  symbols, `LibraryError` for discovery failure)

### Files modified

| File | Change |
|------|--------|
| `crates/python/src/delegate.rs` | Add `xnnpack_delegate()` pyfunction |
| `crates/python/src/lib.rs` | Register `xnnpack_delegate` in module |
| `crates/python/edgefirst_tflite.pyi` | Add type stub |
| `crates/python/README.md` | Add XNNPACK usage section |

## 2. Rust Documentation

### README.md

- Add XNNPACK to the delegate overview section (alongside VxDelegate, Ethos-U,
  Neutron)
- Add a Rust usage example showing `Delegate::xnnpack(&lib, 4)`
- Note that XNNPACK requires the TFLite library compiled with
  `-DTFLITE_ENABLE_XNNPACK=ON`

### ARCHITECTURE.md

- Add a "Built-in Delegates" subsection alongside the existing "Delegate
  Extension Probing" section
- Explain the difference: external delegates load from separate `.so` via
  `tflite_plugin_*`; built-in delegates (XNNPACK) use symbols from the main
  TFLite library resolved via `XnnPackFunctions::try_load()`
- Note the `Library::reopen()` mechanism for lifetime management

## 3. Release v0.4.0

### Version bump

- `Cargo.toml` workspace.package.version: `"0.4.0"`
- `Cargo.toml` workspace.dependencies: `edgefirst-tflite-sys = "0.4.0"`,
  `edgefirst-tflite = "0.4.0"`

### CHANGELOG.md

Move [Unreleased] into [0.4.0] with entries:

**Added:**
- `Delegate::xnnpack(&Library, num_threads)` for CPU-accelerated inference
  via the built-in XNNPACK delegate
- `xnnpack_delegate(num_threads)` Python function for XNNPACK delegate
  creation
- `XnnPackFunctions` and `TfLiteXNNPackDelegateOptions` in
  `edgefirst-tflite-sys` for runtime XNNPACK symbol loading
- `discover_with_path()` in discovery module, returning the loaded library
  path
- `Library::reopen()` for built-in delegate lifetime management
- `tensorflowlite_c::library()` accessor for the underlying
  `libloading::Library`

**Changed:**
- Library paths are now canonicalised when they refer to existing files,
  making `reopen()` resilient to working-directory changes

### Branch and PR

- Branch: `release/0.4.0` from `main`
- PR title: `Release v0.4.0`
- After merge: tag `v0.4.0`, push tag (CI handles crates.io publish)

## Verification

1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace -- -D warnings`
3. `cargo clippy --workspace --features full -- -D warnings`
4. `cargo test --workspace --all-features`
5. `RUSTDOCFLAGS="-D warnings" cargo doc -p edgefirst-tflite -p edgefirst-tflite-sys --all-features --no-deps`
6. Python: `maturin develop` + `python -c "from edgefirst_tflite import xnnpack_delegate; print(xnnpack_delegate)"`
