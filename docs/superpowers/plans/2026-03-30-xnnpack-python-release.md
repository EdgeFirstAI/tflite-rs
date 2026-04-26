# XNNPACK Python Binding & v0.4.0 Release — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose XNNPACK delegate support in the Python API and release v0.4.0.

**Architecture:** Add `xnnpack_delegate(num_threads)` as a module-level Python function that auto-discovers the TFLite library and creates an XNNPACK delegate. Update Rust and Python documentation. Bump version and prepare the release branch.

**Tech Stack:** Rust, PyO3, maturin, cargo

---

### Task 1: Add `xnnpack_delegate()` Python function

**Files:**
- Modify: `crates/python/src/delegate.rs` (append after `load_delegate` at line 183)
- Modify: `crates/python/src/lib.rs:39` (add function registration)

- [ ] **Step 1: Add the Rust implementation**

Append to `crates/python/src/delegate.rs` after the `load_delegate` function:

```rust
// ---------------------------------------------------------------------------
// xnnpack_delegate() — module-level function
// ---------------------------------------------------------------------------

/// Create an XNNPACK delegate for CPU-accelerated inference.
///
/// XNNPACK optimises floating-point and quantised operations on ARM and
/// x86 CPUs using SIMD instructions.
///
/// Args:
///     num_threads: XNNPACK threadpool size. 0 lets XNNPACK choose.
///
/// Returns:
///     A `Delegate` to pass to `Interpreter(experimental_delegates=[...])`.
#[pyfunction]
#[pyo3(signature = (num_threads=0))]
pub fn xnnpack_delegate(num_threads: i32) -> PyResult<PyDelegate> {
    let lib = edgefirst_tflite::Library::new().map_err(error::to_py_err)?;
    let delegate =
        edgefirst_tflite::Delegate::xnnpack(&lib, num_threads).map_err(error::to_py_err)?;
    Ok(PyDelegate {
        inner: Some(delegate),
    })
}
```

- [ ] **Step 2: Register in the Python module**

In `crates/python/src/lib.rs`, add after line 39 (`load_delegate` registration):

```rust
    m.add_function(wrap_pyfunction!(delegate::xnnpack_delegate, m)?)?;
```

- [ ] **Step 3: Build and verify**

Run:
```bash
cargo clippy --workspace -- -D warnings
cargo clippy --workspace --features full -- -D warnings
cargo test --workspace --all-features
```
Expected: all pass, no warnings.

- [ ] **Step 4: Commit**

```bash
git add crates/python/src/delegate.rs crates/python/src/lib.rs
git commit -s -m "Add xnnpack_delegate() Python function"
```

---

### Task 2: Update Python type stub

**Files:**
- Modify: `crates/python/edgefirst_tflite.pyi` (insert after `load_delegate` at line 330)

- [ ] **Step 1: Add the type stub**

Insert after the `load_delegate` function definition (after line 330, before the `# DMA-BUF` section comment):

```python
def xnnpack_delegate(num_threads: int = 0) -> Delegate:
    """Create an XNNPACK delegate for CPU-accelerated inference.

    XNNPACK optimises floating-point and quantised operations on ARM and
    x86 CPUs using SIMD instructions (NEON on ARM, AVX/SSE on x86).

    Args:
        num_threads: XNNPACK threadpool size. Use 1 for single-threaded,
            higher values for parallelism, or 0 to let XNNPACK choose.

    Returns:
        A ``Delegate`` to pass to ``Interpreter(experimental_delegates=[...])``.

    Raises:
        InvalidArgumentError: If the TFLite library was not compiled with
            XNNPACK support (``-DTFLITE_ENABLE_XNNPACK=ON``).
        LibraryError: If no TFLite shared library can be found.

    Example::

        delegate = xnnpack_delegate(num_threads=4)
        interp = Interpreter(
            model_path="model.tflite",
            experimental_delegates=[delegate],
        )
    """
    ...

```

- [ ] **Step 2: Commit**

```bash
git add crates/python/edgefirst_tflite.pyi
git commit -s -m "Add xnnpack_delegate type stub"
```

---

### Task 3: Update Python README

**Files:**
- Modify: `crates/python/README.md` (insert XNNPACK section after the VxDelegate delegate example around line 92)

- [ ] **Step 1: Add XNNPACK section**

Insert after the closing ` ``` ` of the VxDelegate delegate example (after line 92, before `## EdgeFirst Extensions`):

```markdown

### XNNPACK (CPU Acceleration)

XNNPACK accelerates floating-point and quantised models on ARM and x86 CPUs
using SIMD instructions. No external delegate library is needed — XNNPACK is
built into the TFLite library when compiled with `-DTFLITE_ENABLE_XNNPACK=ON`.

```python
from edgefirst_tflite import Interpreter, xnnpack_delegate

delegate = xnnpack_delegate(num_threads=4)

interp = Interpreter(
    model_path="model.tflite",
    experimental_delegates=[delegate],
)
interp.invoke()
```

```

- [ ] **Step 2: Commit**

```bash
git add crates/python/README.md
git commit -s -m "Document XNNPACK delegate in Python README"
```

---

### Task 4: Update Rust README

**Files:**
- Modify: `README.md` (two locations)

- [ ] **Step 1: Add XNNPACK delegate section**

After the `## VxDelegate Extensions` section (after line 176, before `## Examples`), add:

```markdown

## XNNPACK (CPU Acceleration)

XNNPACK accelerates floating-point and quantised models on ARM and x86 CPUs
using SIMD instructions. Unlike external delegates, XNNPACK is built into
the TFLite library (when compiled with `-DTFLITE_ENABLE_XNNPACK=ON`).

```rust,no_run
use edgefirst_tflite::{Delegate, Interpreter, Library, Model};

let lib = Library::new()?;
let model = Model::from_file(&lib, "model.tflite")?;

let delegate = Delegate::xnnpack(&lib, 4)?;

let mut interpreter = Interpreter::builder(&lib)?
    .delegate(delegate)
    .num_threads(4)
    .build(&model)?;

interpreter.invoke()?;
# Ok::<(), edgefirst_tflite::Error>(())
```

```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -s -m "Document XNNPACK delegate in Rust README"
```

---

### Task 5: Update ARCHITECTURE.md

**Files:**
- Modify: `ARCHITECTURE.md` (insert after the "Delegate Extension Probing" section, around line 255)

- [ ] **Step 1: Add built-in delegates section**

Insert after line 255 (after the HAL/VxDelegate probing section, before `## DMA-BUF Zero-Copy Data Flow`):

```markdown

## Built-in Delegates (XNNPACK)

Unlike external delegates that are loaded from separate `.so` files via the
`tflite_plugin_create_delegate` plugin ABI, built-in delegates have their
symbols compiled into the main TFLite library. XNNPACK is the primary
example.

`Delegate::xnnpack(&Library, num_threads)` works as follows:

1. `XnnPackFunctions::try_load(lib.as_sys().library())` — resolves
   `TfLiteXNNPackDelegateOptionsDefault`, `TfLiteXNNPackDelegateCreate`,
   and `TfLiteXNNPackDelegateDelete` from the main TFLite library.
   Returns `None` when the library was compiled without XNNPACK.
2. Calls `TfLiteXNNPackDelegateOptionsDefault()` to get safe defaults,
   overrides `num_threads`.
3. Calls `TfLiteXNNPackDelegateCreate(&opts)` to obtain a
   `*mut TfLiteDelegate`.
4. `Library::reopen()` opens a second OS handle to the same `.so`,
   incrementing the `dlopen` refcount. This handle is stored in
   `Delegate._lib` so the XNNPACK `delete` function pointer remains
   valid even if the original `Library` is dropped first.

The `xnnpack_ffi` module in `edgefirst-tflite-sys` follows the same
`try_load` pattern as `vx_ffi` and `hal_ffi`, with function pointers
stored in an `XnnPackFunctions` struct.

```

- [ ] **Step 2: Commit**

```bash
git add ARCHITECTURE.md
git commit -s -m "docs: Add built-in delegates section to ARCHITECTURE.md"
```

---

### Task 6: Version bump and CHANGELOG

**Files:**
- Modify: `Cargo.toml` (lines 9, 33, 34)
- Modify: `CHANGELOG.md` (lines 8-9, 128)

- [ ] **Step 1: Bump version in Cargo.toml**

In root `Cargo.toml`, change:

```toml
# line 9
version = "0.4.0"

# line 33
edgefirst-tflite-sys = { version = "0.4.0", path = "crates/tflite-sys" }
# line 34
edgefirst-tflite = { version = "0.4.0", path = "crates/tflite" }
```

- [ ] **Step 2: Update CHANGELOG.md**

Replace the `## [Unreleased]` section (lines 8-9) with:

```markdown
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
```

Update the comparison links at the bottom of the file. Replace line 128:

```markdown
[Unreleased]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/EdgeFirstAI/tflite-rs/compare/v0.2.1...v0.3.0
```

- [ ] **Step 3: Verify build with new version**

Run:
```bash
cargo clippy --workspace -- -D warnings
cargo clippy --workspace --features full -- -D warnings
cargo test --workspace --all-features
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml CHANGELOG.md
git commit -s -m "Release v0.4.0"
```

---

### Task 7: Create release branch and PR

- [ ] **Step 1: Create release branch and push**

```bash
git checkout -b release/0.4.0
git push -u origin release/0.4.0
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --title "Release v0.4.0" --body "$(cat <<'EOF'
## Summary

- Add `Delegate::xnnpack(&lib, num_threads)` for CPU-accelerated inference via built-in XNNPACK delegate
- Add `xnnpack_delegate(num_threads)` Python function
- Update Rust and Python documentation with XNNPACK usage examples
- Update ARCHITECTURE.md with built-in delegate section

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full v0.4.0 entry.

## Test plan

- [x] `cargo fmt --all -- --check`
- [x] `cargo clippy --workspace -- -D warnings`
- [x] `cargo clippy --workspace --features full -- -D warnings`
- [x] `cargo test --workspace --all-features`
- [ ] CI passes on all platforms
EOF
)"
```
