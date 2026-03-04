// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Delegate loading with configuration options.
//!
//! Delegates provide hardware acceleration for `TFLite` inference. The most
//! common delegate for i.MX platforms is the `VxDelegate`, which offloads
//! operations to the NPU.

use std::ffi::CString;
use std::path::Path;
use std::ptr::{self, NonNull};

use edgefirst_tflite_sys::TfLiteDelegate;

use crate::error::{Error, Result};

#[cfg(feature = "dmabuf")]
use edgefirst_tflite_sys::vx_ffi::VxDmaBufFunctions;

#[cfg(feature = "camera_adaptor")]
use edgefirst_tflite_sys::vx_ffi::VxCameraAdaptorFunctions;

// ---------------------------------------------------------------------------
// DelegateOptions
// ---------------------------------------------------------------------------

/// Key-value options for configuring an external delegate.
///
/// # Examples
///
/// ```no_run
/// use edgefirst_tflite::DelegateOptions;
///
/// let opts = DelegateOptions::new()
///     .option("cache_file_path", "/tmp/vx_cache")
///     .option("device_id", "0");
/// ```
#[derive(Debug, Default, Clone)]
pub struct DelegateOptions {
    options: Vec<(String, String)>,
}

impl DelegateOptions {
    /// Create an empty set of delegate options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a key-value option pair.
    #[must_use]
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.push((key.into(), value.into()));
        self
    }
}

// ---------------------------------------------------------------------------
// Delegate
// ---------------------------------------------------------------------------

/// An external `TFLite` delegate for hardware acceleration.
///
/// Delegates are loaded from shared libraries that export the standard
/// `tflite_plugin_create_delegate` / `tflite_plugin_destroy_delegate`
/// entry points.
///
/// # Examples
///
/// ```no_run
/// use edgefirst_tflite::{Delegate, DelegateOptions};
///
/// // Load delegate with default options
/// let delegate = Delegate::load("libvx_delegate.so")?;
///
/// // Load delegate with options
/// let delegate = Delegate::load_with_options(
///     "libvx_delegate.so",
///     &DelegateOptions::new()
///         .option("cache_file_path", "/tmp/vx_cache")
///         .option("device_id", "0"),
/// )?;
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
#[allow(clippy::struct_field_names)]
pub struct Delegate {
    delegate: NonNull<TfLiteDelegate>,
    free: unsafe extern "C" fn(*mut TfLiteDelegate),
    // Keeps the delegate .so loaded for the delegate's lifetime.
    _lib: libloading::Library,

    #[cfg(feature = "dmabuf")]
    dmabuf_fns: Option<VxDmaBufFunctions>,

    #[cfg(feature = "camera_adaptor")]
    camera_adaptor_fns: Option<VxCameraAdaptorFunctions>,
}

impl Delegate {
    /// Load an external delegate from a shared library with default options.
    ///
    /// # Errors
    ///
    /// Returns an error if the library cannot be loaded, required symbols
    /// are missing, or the delegate returns a null pointer.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(path, &DelegateOptions::default())
    }

    /// Load an external delegate with configuration options.
    ///
    /// # Errors
    ///
    /// Returns an error if the library cannot be loaded, required symbols
    /// are missing, the delegate returns a null pointer, or any option key
    /// or value contains an interior NUL byte.
    pub fn load_with_options(path: impl AsRef<Path>, options: &DelegateOptions) -> Result<Self> {
        // SAFETY: Loading the shared library via `libloading`. The library is
        // kept alive in `_lib` for the lifetime of the `Delegate`.
        let lib =
            unsafe { libloading::Library::new(path.as_ref().as_os_str()) }.map_err(Error::from)?;

        // SAFETY: Resolving the `tflite_plugin_create_delegate` symbol from
        // the loaded library. The library is valid and loaded above.
        let create_fn = unsafe {
            lib.get::<unsafe extern "C" fn(
                *const *const std::os::raw::c_char,
                *const *const std::os::raw::c_char,
                usize,
                Option<unsafe extern "C" fn(*const std::os::raw::c_char)>,
            ) -> *mut TfLiteDelegate>(b"tflite_plugin_create_delegate")
        }
        .map_err(Error::from)?;

        // SAFETY: Resolving the `tflite_plugin_destroy_delegate` symbol from
        // the same loaded library.
        let destroy_fn = unsafe {
            lib.get::<unsafe extern "C" fn(*mut TfLiteDelegate)>(b"tflite_plugin_destroy_delegate")
        }
        .map_err(Error::from)?;

        // Convert options to C string arrays.
        let (keys_c, values_c): (Vec<CString>, Vec<CString>) = options
            .options
            .iter()
            .map(|(k, v)| {
                Ok((
                    CString::new(k.as_str()).map_err(|_| {
                        Error::invalid_argument(format!(
                            "option key \"{k}\" contains interior NUL byte"
                        ))
                    })?,
                    CString::new(v.as_str()).map_err(|_| {
                        Error::invalid_argument(format!(
                            "option value \"{v}\" contains interior NUL byte"
                        ))
                    })?,
                ))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
        let keys_ptrs: Vec<*const std::os::raw::c_char> =
            keys_c.iter().map(|c| c.as_ptr()).collect();
        let values_ptrs: Vec<*const std::os::raw::c_char> =
            values_c.iter().map(|c| c.as_ptr()).collect();

        // SAFETY: `create_fn` is a valid symbol resolved above. `keys_ptrs`
        // and `values_ptrs` point to valid NUL-terminated C strings (from
        // `CString::new`), or null when empty. `keys_c` and `values_c` are
        // alive for this call, keeping the pointers valid.
        let raw = unsafe {
            create_fn(
                if keys_ptrs.is_empty() {
                    ptr::null()
                } else {
                    keys_ptrs.as_ptr()
                },
                if values_ptrs.is_empty() {
                    ptr::null()
                } else {
                    values_ptrs.as_ptr()
                },
                options.options.len(),
                None,
            )
        };

        let delegate = NonNull::new(raw)
            .ok_or_else(|| Error::null_pointer("tflite_plugin_create_delegate returned null"))?;

        // Copy the destroy function pointer before lib is stored.
        let free = *destroy_fn;

        // Probe for VxDelegate extensions.
        #[cfg(feature = "dmabuf")]
        // SAFETY: `lib` is a valid loaded library. `try_load` resolves
        // optional symbols; missing symbols return `None`, not UB.
        let dmabuf_fns = unsafe { VxDmaBufFunctions::try_load(&lib) };

        #[cfg(feature = "camera_adaptor")]
        // SAFETY: Same as `VxDmaBufFunctions::try_load` above — resolves
        // optional CameraAdaptor symbols from the loaded library.
        let camera_adaptor_fns = unsafe { VxCameraAdaptorFunctions::try_load(&lib) };

        Ok(Self {
            delegate,
            free,
            _lib: lib,
            #[cfg(feature = "dmabuf")]
            dmabuf_fns,
            #[cfg(feature = "camera_adaptor")]
            camera_adaptor_fns,
        })
    }

    /// Returns the raw delegate pointer.
    ///
    /// This is an escape hatch for advanced use cases that need direct
    /// FFI access to the delegate.
    #[must_use]
    pub fn as_ptr(&self) -> *mut TfLiteDelegate {
        self.delegate.as_ptr()
    }

    /// Access DMA-BUF extensions if available on this delegate.
    #[cfg(feature = "dmabuf")]
    #[must_use]
    pub fn dmabuf(&self) -> Option<crate::dmabuf::DmaBuf<'_>> {
        self.dmabuf_fns
            .as_ref()
            .map(|fns| crate::dmabuf::DmaBuf::new(self.delegate, fns))
    }

    /// Returns `true` if this delegate supports DMA-BUF zero-copy.
    #[cfg(feature = "dmabuf")]
    #[must_use]
    pub fn has_dmabuf(&self) -> bool {
        self.dmabuf_fns.is_some()
    }

    /// Access `CameraAdaptor` extensions if available on this delegate.
    #[cfg(feature = "camera_adaptor")]
    #[must_use]
    pub fn camera_adaptor(&self) -> Option<crate::camera_adaptor::CameraAdaptor<'_>> {
        self.camera_adaptor_fns
            .as_ref()
            .map(|fns| crate::camera_adaptor::CameraAdaptor::new(self.delegate, fns))
    }

    /// Returns `true` if this delegate supports `CameraAdaptor`.
    #[cfg(feature = "camera_adaptor")]
    #[must_use]
    pub fn has_camera_adaptor(&self) -> bool {
        self.camera_adaptor_fns.is_some()
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Delegate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("Delegate");
        d.field("ptr", &self.delegate);

        #[cfg(feature = "dmabuf")]
        d.field("has_dmabuf", &self.dmabuf_fns.is_some());

        #[cfg(feature = "camera_adaptor")]
        d.field("has_camera_adaptor", &self.camera_adaptor_fns.is_some());

        d.finish_non_exhaustive()
    }
}

impl Drop for Delegate {
    fn drop(&mut self) {
        // SAFETY: The delegate pointer was created by `tflite_plugin_create_delegate`
        // and `free` is the matching `tflite_plugin_destroy_delegate` from the same
        // library, which is still loaded (held by `_lib`).
        unsafe { (self.free)(self.delegate.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_options() {
        let opts = DelegateOptions::new();
        let debug = format!("{opts:?}");
        assert_eq!(debug, "DelegateOptions { options: [] }");
    }

    #[test]
    fn builder_chaining() {
        let opts = DelegateOptions::new().option("a", "1").option("b", "2");
        assert_eq!(opts.options.len(), 2);
    }

    #[test]
    fn default_matches_new() {
        let from_new = format!("{:?}", DelegateOptions::new());
        let from_default = format!("{:?}", DelegateOptions::default());
        assert_eq!(from_new, from_default);
    }

    #[test]
    fn clone_produces_equal_values() {
        let opts = DelegateOptions::new().option("key", "value");
        let cloned = opts.clone();
        assert_eq!(format!("{opts:?}"), format!("{cloned:?}"));
    }

    #[test]
    fn debug_formatting_not_empty() {
        let opts = DelegateOptions::new().option("cache", "/tmp");
        let debug = format!("{opts:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("DelegateOptions"));
        assert!(debug.contains("cache"));
        assert!(debug.contains("/tmp"));
    }
}
