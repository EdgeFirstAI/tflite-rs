// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Safe wrapper around the `TFLite` shared-library handle.

use std::fmt;
use std::path::{Path, PathBuf};

use edgefirst_tflite_sys::experimental_ffi::ExperimentalFunctions;
use edgefirst_tflite_sys::litert::{LiteRtFunctions, MissingSymbol};

use crate::error::{Error, Result};

/// Handle to a loaded `TFLite` shared library.
///
/// `Library` wraps the FFI function table produced by `libloading` and
/// `bindgen`, providing safe construction via auto-discovery or an explicit
/// filesystem path. Everything else in this crate borrows from a `Library`, so
/// it must outlive every [`Model`](crate::Model),
/// [`Interpreter`](crate::Interpreter), and [`litert`](crate::litert) handle
/// created from it — the borrow checker enforces this.
///
/// # `LiteRT` probing
///
/// Hosts that ship `LiteRT` Next export both `TfLite*` and `LiteRt*` from the
/// *same* shared object (Android's `libLiteRt.so` is one such library). At load
/// time `Library` probes for the `LiteRt*` surface on the already-open handle:
///
/// - [`Library::has_litert`] — is the `LiteRT` path available?
/// - [`Library::litert`] — the resolved function table, if so.
/// - [`Library::litert_missing_symbol`] — *why not*, if not.
///
/// Probing never fails the load: a classic TensorFlow Lite library yields a
/// fully working [`Interpreter`](crate::Interpreter) and a `LiteRT` path that
/// reports [`Error::is_litert_unavailable`] on first use.
///
/// # Examples
///
/// ```no_run
/// use edgefirst_tflite::Library;
///
/// // Auto-discover TFLite library
/// let lib = Library::new()?;
///
/// // Or load from a specific path
/// let lib = Library::from_path("/usr/lib/libtensorflowlite_c.so")?;
///
/// if lib.has_litert() {
///     println!("LiteRT Next available");
/// } else if let Some(sym) = lib.litert_missing_symbol() {
///     println!("no LiteRT: {sym} unresolved");
/// }
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
pub struct Library {
    // Declared before `inner` so it is dropped first. `LiteRtFunctions` holds
    // no destructor today, but its pointers are only meaningful while `inner`
    // keeps the shared object mapped; this ordering keeps that true by
    // construction if the type ever gains a `Drop`.
    litert: std::result::Result<LiteRtFunctions, MissingSymbol>,
    // Same ordering rationale as `litert`: resolved once at load, valid only
    // while `inner` keeps the shared object mapped.
    experimental: Option<ExperimentalFunctions>,
    inner: edgefirst_tflite_sys::tensorflowlite_c,
    path: Option<PathBuf>,
}

impl Library {
    /// Discover and load the `TFLite` shared library automatically.
    ///
    /// This probes well-known versioned and unversioned library paths using
    /// the [`edgefirst_tflite_sys::discovery`] module.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if no compatible `TFLite` library can be found.
    pub fn new() -> Result<Self> {
        let (inner, path) =
            edgefirst_tflite_sys::discovery::discover_with_path().map_err(Error::from)?;
        Ok(Self::from_inner(inner, Some(path)))
    }

    /// Load the `TFLite` shared library from a specific `path`.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the library cannot be loaded from `path` or
    /// required symbols are missing.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let raw = path.as_ref();
        let inner = edgefirst_tflite_sys::discovery::load(raw).map_err(Error::from)?;
        // Canonicalise so reopen() works even if the working directory
        // changes later. Fall back to the original for soname-only paths.
        let resolved = if raw.is_file() {
            std::fs::canonicalize(raw).unwrap_or_else(|_| raw.to_path_buf())
        } else {
            raw.to_path_buf()
        };
        Ok(Self::from_inner(inner, Some(resolved)))
    }

    fn from_inner(inner: edgefirst_tflite_sys::tensorflowlite_c, path: Option<PathBuf>) -> Self {
        // SAFETY: `inner.library()` is the libloading handle that owns the
        // mapped object containing any LiteRt* symbols, and that handle lives
        // in `Self` alongside the resolved table, so the pointers cannot
        // outlive the mapping. `try_load` reports the first unresolved symbol
        // rather than failing, which is the expected path on classic TFLite.
        let litert = unsafe { LiteRtFunctions::try_load(inner.library()) };
        // SAFETY: same handle, same lifetime argument as above. Absent on a
        // runtime built without the experimental C API, which is not an error.
        let experimental = unsafe { ExperimentalFunctions::try_load(inner.library()) };
        Self {
            litert,
            experimental,
            inner,
            path,
        }
    }

    /// Returns a reference to the underlying FFI function table.
    ///
    /// This is an escape hatch for advanced users who need direct access to
    /// the raw `tensorflowlite_c` bindings.
    #[must_use]
    pub fn as_sys(&self) -> &edgefirst_tflite_sys::tensorflowlite_c {
        &self.inner
    }

    /// Returns the resolved `LiteRT` function table, or `None` on a library
    /// that does not export the full `LiteRt*` surface.
    ///
    /// Most callers do not need this: constructing a
    /// [`litert::Environment`](crate::litert::Environment) returns a
    /// descriptive error when `LiteRT` is unavailable, which is usually better
    /// than an `Option` you have to interpret. Reach for this when you need the
    /// raw function pointers.
    #[must_use]
    pub fn litert(&self) -> Option<&LiteRtFunctions> {
        self.litert.as_ref().ok()
    }

    /// Returns `true` when the full `LiteRT` symbol set resolved at load time.
    ///
    /// Use this to branch between the [`Interpreter`](crate::Interpreter) and
    /// [`litert::CompiledModel`](crate::litert::CompiledModel) paths.
    #[must_use]
    pub fn has_litert(&self) -> bool {
        self.litert.is_ok()
    }

    /// Returns the name of the first `LiteRt*` symbol that failed to resolve,
    /// or `None` when the full set is present.
    ///
    /// This is the diagnostic companion to [`Library::has_litert`]. Symbols are
    /// probed in a fixed order beginning with `LiteRtCreateEnvironment`, so:
    ///
    /// - `Some("LiteRtCreateEnvironment")` — the library exports no `LiteRT`
    ///   surface at all; it is a classic TensorFlow Lite build. Expected, not a
    ///   fault.
    /// - `Some(other)` — the library exports *some* `LiteRt*` symbols but not
    ///   the required set. This usually means a version skew between the
    ///   vendored headers and the deployed runtime, and is worth surfacing.
    /// - `None` — the full set resolved.
    ///
    /// The complete list of probed symbols is
    /// [`LiteRtFunctions::required_symbols`].
    #[must_use]
    pub fn litert_missing_symbol(&self) -> Option<&'static str> {
        self.litert.as_ref().err().map(|e| e.name())
    }

    /// The `LiteRT` probe result, preserving which symbol was missing.
    ///
    /// Used by [`crate::litert`] constructors to build a descriptive error.
    pub(crate) fn litert_result(&self) -> std::result::Result<&LiteRtFunctions, MissingSymbol> {
        self.litert.as_ref().map_err(|e| *e)
    }

    /// Returns `true` when the runtime exports the experimental C API backing
    /// [`Interpreter::set_custom_allocation_for_input`](crate::Interpreter::set_custom_allocation_for_input).
    ///
    /// Check this before designing a pipeline around custom allocations: the
    /// symbols come from `c_api_experimental.h` and a runtime built without
    /// them still works, just with an arena copy on every inference.
    #[must_use]
    pub fn has_custom_allocation(&self) -> bool {
        self.experimental.is_some()
    }

    /// The resolved experimental function table, if the runtime exports it.
    pub(crate) fn experimental(&self) -> Option<&ExperimentalFunctions> {
        self.experimental.as_ref()
    }

    /// Re-open the underlying shared library, incrementing the OS refcount.
    ///
    /// This is used internally to keep the main `TFLite` library alive for
    /// built-in delegates (e.g., XNNPACK) whose function pointers live in
    /// the main library rather than a separate delegate `.so`.
    pub(crate) fn reopen(&self) -> Result<libloading::Library> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| Error::invalid_argument("library path not available for reopen"))?;
        // SAFETY: Re-opening the same shared library increments the OS
        // refcount. The path is known-valid because it was successfully
        // loaded during construction.
        unsafe { libloading::Library::new(path.as_os_str()) }.map_err(Error::from)
    }
}

// SAFETY: `Library` holds a `tensorflowlite_c` struct whose fields are
// function pointers (all `Send + Sync`) and a `libloading::Library` (which
// is `Send + Sync`). The TFLite C API has no thread affinity — function
// pointers resolved from a loaded library are safe to call from any thread.
unsafe impl Send for Library {}

// SAFETY: All access through `&Library` is via immutable function-pointer
// calls (`as_sys()` returns `&tensorflowlite_c`). No interior mutability.
unsafe impl Sync for Library {}

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Library")
            .field("inner", &"tensorflowlite_c { .. }")
            .field("path", &self.path)
            // Summarised rather than dumped: the resolved table is 30+ function
            // pointers and adds nothing to a debug line.
            .field(
                "litert",
                &match &self.litert {
                    Ok(_) => "available".to_string(),
                    Err(missing) => format!("unavailable ({})", missing.name()),
                },
            )
            .field(
                "experimental",
                &if self.experimental.is_some() {
                    "available"
                } else {
                    "unavailable"
                },
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Name of a shared library that is always present and never exports
    /// `LiteRt*`, used as a negative control for the probe.
    fn system_c_library() -> &'static str {
        #[cfg(target_os = "macos")]
        {
            "libSystem.B.dylib"
        }
        #[cfg(target_os = "linux")]
        {
            "libc.so.6"
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            "kernel32.dll"
        }
    }

    #[test]
    fn litert_probe_fails_on_library_without_symbols() {
        // SAFETY: loading the system C library is safe; we only probe symbols.
        let lib = unsafe { libloading::Library::new(system_c_library()) }.expect("system libc");
        let litert = unsafe { LiteRtFunctions::try_load(&lib) };
        let missing = litert.expect_err("system C library must not export LiteRt* symbols");
        // The failure must name the *first* probed symbol, which is what lets
        // callers distinguish "no LiteRT" from "partial LiteRT".
        assert_eq!(missing.name(), "LiteRtCreateEnvironment");
    }

    #[test]
    fn litert_unavailable_error_names_the_missing_symbol() {
        // SAFETY: as above.
        let lib = unsafe { libloading::Library::new(system_c_library()) }.expect("system libc");
        let missing = unsafe { LiteRtFunctions::try_load(&lib) }.unwrap_err();
        let err = Error::litert_unavailable(missing);
        assert!(err.is_litert_unavailable());
        assert_eq!(err.litert_missing_symbol(), Some("LiteRtCreateEnvironment"));
    }

    #[test]
    fn required_symbol_list_is_non_empty_and_prefixed() {
        let symbols = LiteRtFunctions::required_symbols();
        assert!(!symbols.is_empty());
        assert!(symbols.iter().all(|s| s.starts_with("LiteRt")));
    }
}
