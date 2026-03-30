// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Safe wrapper around the `TFLite` shared-library handle.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Handle to a loaded `TFLite` shared library.
///
/// `Library` wraps the FFI function table produced by `libloading` and
/// `bindgen`, providing safe construction via auto-discovery or an explicit
/// filesystem path.
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
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
pub struct Library {
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
        Ok(Self {
            inner,
            path: Some(path),
        })
    }

    /// Load the `TFLite` shared library from a specific `path`.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the library cannot be loaded from `path` or
    /// required symbols are missing.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref().to_path_buf();
        let inner = edgefirst_tflite_sys::discovery::load(&p).map_err(Error::from)?;
        Ok(Self {
            inner,
            path: Some(p),
        })
    }

    /// Returns a reference to the underlying FFI function table.
    ///
    /// This is an escape hatch for advanced users who need direct access to
    /// the raw `tensorflowlite_c` bindings.
    #[must_use]
    pub fn as_sys(&self) -> &edgefirst_tflite_sys::tensorflowlite_c {
        &self.inner
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

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Library")
            .field("inner", &"tensorflowlite_c { .. }")
            .finish()
    }
}
