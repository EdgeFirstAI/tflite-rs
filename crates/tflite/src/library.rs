// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Safe wrapper around the `TFLite` shared-library handle.

use std::fmt;
use std::path::Path;

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
}

impl Library {
    /// Discover and load the `TFLite` shared library automatically.
    ///
    /// This probes well-known versioned and unversioned library paths using
    /// [`edgefirst_tflite_sys::discovery::discover`].
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if no compatible `TFLite` library can be found.
    pub fn new() -> Result<Self> {
        let inner = edgefirst_tflite_sys::discovery::discover().map_err(Error::from)?;
        Ok(Self { inner })
    }

    /// Load the `TFLite` shared library from a specific `path`.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the library cannot be loaded from `path` or
    /// required symbols are missing.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let inner = edgefirst_tflite_sys::discovery::load(path).map_err(Error::from)?;
        Ok(Self { inner })
    }

    /// Returns a reference to the underlying FFI function table.
    ///
    /// This is an escape hatch for advanced users who need direct access to
    /// the raw `tensorflowlite_c` bindings.
    #[must_use]
    pub fn as_sys(&self) -> &edgefirst_tflite_sys::tensorflowlite_c {
        &self.inner
    }
}

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Library")
            .field("inner", &"tensorflowlite_c { .. }")
            .finish()
    }
}
