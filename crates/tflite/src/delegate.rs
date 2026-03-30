// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Delegate loading with configuration options.
//!
//! Delegates provide hardware acceleration for `TFLite` inference. External
//! delegates (e.g., `VxDelegate` for NPU) are loaded from shared libraries.
//! Built-in delegates (e.g., XNNPACK for CPU SIMD) use symbols from the
//! main `TFLite` library.

use std::ffi::{c_void, CString};
use std::path::Path;
use std::ptr::{self, NonNull};

use edgefirst_tflite_sys::xnnpack_ffi::XnnPackFunctions;
use edgefirst_tflite_sys::TfLiteDelegate;

use crate::error::{Error, Result};

#[cfg(feature = "dmabuf")]
use edgefirst_tflite_sys::hal_ffi::HalDmaBufFunctions;

#[cfg(feature = "dmabuf")]
use edgefirst_tflite_sys::vx_ffi::VxDmaBufFunctions;

#[cfg(feature = "camera_adaptor")]
use edgefirst_tflite_sys::hal_ffi::HalCameraAdaptorFunctions;

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

/// A `TFLite` delegate for hardware acceleration.
///
/// Delegates come in two flavours:
///
/// - **External** — loaded from a separate `.so` via
///   [`Delegate::load`] / [`Delegate::load_with_options`], using the
///   `tflite_plugin_create_delegate` / `tflite_plugin_destroy_delegate`
///   plugin entry points.
/// - **Built-in** — created from symbols inside the main `TFLite`
///   library (e.g., [`Delegate::xnnpack`]).
///
/// In both cases the `Delegate` owns the resources needed to keep the
/// delegate alive and will call the matching destroy function on drop.
///
/// # Examples
///
/// ```no_run
/// use edgefirst_tflite::{Delegate, DelegateOptions, Library};
///
/// let lib = Library::new()?;
///
/// // External delegate with default options
/// let delegate = Delegate::load("libvx_delegate.so")?;
///
/// // External delegate with options
/// let delegate = Delegate::load_with_options(
///     "libvx_delegate.so",
///     &DelegateOptions::new()
///         .option("cache_file_path", "/tmp/vx_cache")
///         .option("device_id", "0"),
/// )?;
///
/// // Built-in XNNPACK delegate
/// let delegate = Delegate::xnnpack(&lib, 4)?;
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
#[allow(clippy::struct_field_names)]
pub struct Delegate {
    delegate: NonNull<TfLiteDelegate>,
    free: unsafe extern "C" fn(*mut TfLiteDelegate),
    // Keeps the delegate .so loaded for the delegate's lifetime.
    _lib: libloading::Library,

    #[cfg(feature = "dmabuf")]
    hal_dmabuf_fns: Option<HalDmaBufFunctions>,

    /// Inner delegate handle returned by `hal_dmabuf_get_instance()`.
    ///
    /// This is the opaque `hal_delegate_t` (`*mut c_void`) that HAL API
    /// functions expect as their first argument. It is distinct from the
    /// `TfLiteDelegate*` outer pointer and must be used for all HAL calls.
    /// Both `DmaBuf` and `CameraAdaptor` share this same handle.
    #[cfg(feature = "dmabuf")]
    hal_delegate_handle: Option<*mut c_void>,

    #[cfg(feature = "dmabuf")]
    dmabuf_fns: Option<VxDmaBufFunctions>,

    #[cfg(feature = "camera_adaptor")]
    hal_camera_fns: Option<HalCameraAdaptorFunctions>,

    #[cfg(feature = "camera_adaptor")]
    camera_adaptor_fns: Option<VxCameraAdaptorFunctions>,
}

// SAFETY: `hal_delegate_handle` is a raw pointer obtained from
// `hal_dmabuf_get_instance()`. The HAL contract guarantees this pointer
// is valid and stable for the lifetime of the loaded delegate library.
// `Delegate` is the sole owner and never shares the handle concurrently.
#[cfg(feature = "dmabuf")]
// SAFETY: See above — the handle is stable and not concurrently accessed.
unsafe impl Send for Delegate {}
#[cfg(feature = "dmabuf")]
// SAFETY: All HAL API methods take `&self` and the underlying C functions
// are thread-safe per the HAL contract.
unsafe impl Sync for Delegate {}

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

        // Probe for delegate DMA-BUF extensions.
        #[cfg(feature = "dmabuf")]
        // SAFETY: `lib` is a valid loaded library. `try_load` resolves
        // optional symbols; missing symbols return `None`, not UB.
        let hal_dmabuf_fns = unsafe { HalDmaBufFunctions::try_load(&lib) };

        // Call `hal_dmabuf_get_instance()` to obtain the inner delegate handle.
        // This is the opaque `hal_delegate_t` pointer that all HAL API calls
        // expect. A null result means HAL is not available on this device.
        #[cfg(feature = "dmabuf")]
        let hal_delegate_handle: Option<*mut c_void> = hal_dmabuf_fns.as_ref().and_then(|fns| {
            // SAFETY: `get_instance` is a valid function pointer loaded from
            // the delegate library. It takes no arguments and returns an opaque
            // handle that is valid for the lifetime of the library.
            let ptr = unsafe { (fns.get_instance)() };
            if ptr.is_null() {
                None
            } else {
                Some(ptr)
            }
        });

        #[cfg(feature = "dmabuf")]
        // SAFETY: Same as above — resolves optional VxDelegate DMA-BUF
        // symbols as a fallback for delegates that haven't adopted the
        // HAL DMA-BUF API yet.
        let dmabuf_fns = unsafe { VxDmaBufFunctions::try_load(&lib) };

        #[cfg(feature = "camera_adaptor")]
        // SAFETY: Same as above — resolves optional HAL Camera Adaptor
        // symbols from the loaded library.
        let hal_camera_fns = unsafe { HalCameraAdaptorFunctions::try_load(&lib) };

        #[cfg(feature = "camera_adaptor")]
        // SAFETY: Same as `VxDmaBufFunctions::try_load` above — resolves
        // optional CameraAdaptor symbols from the loaded library.
        let camera_adaptor_fns = unsafe { VxCameraAdaptorFunctions::try_load(&lib) };

        Ok(Self {
            delegate,
            free,
            _lib: lib,
            #[cfg(feature = "dmabuf")]
            hal_dmabuf_fns,
            #[cfg(feature = "dmabuf")]
            hal_delegate_handle,
            #[cfg(feature = "dmabuf")]
            dmabuf_fns,
            #[cfg(feature = "camera_adaptor")]
            hal_camera_fns,
            #[cfg(feature = "camera_adaptor")]
            camera_adaptor_fns,
        })
    }

    /// Create an XNNPACK delegate for CPU-accelerated inference.
    ///
    /// XNNPACK is a built-in `TFLite` delegate that optimises floating-point
    /// and quantised operations on ARM and x86 CPUs using SIMD instructions.
    ///
    /// The `num_threads` parameter controls the XNNPACK threadpool size.
    /// Use 1 for single-threaded execution, or a higher value for
    /// multi-threaded parallelism. A value of 0 lets XNNPACK choose.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The `TFLite` library was not compiled with XNNPACK support
    /// - The delegate creation returns a null pointer
    /// - The library cannot be re-opened (internal lifetime management)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use edgefirst_tflite::{Delegate, Interpreter, Library, Model};
    ///
    /// let lib = Library::new()?;
    /// let model = Model::from_file(&lib, "model.tflite")?;
    ///
    /// let delegate = Delegate::xnnpack(&lib, 4)?;
    ///
    /// let mut interpreter = Interpreter::builder(&lib)?
    ///     .delegate(delegate)
    ///     .num_threads(4)
    ///     .build(&model)?;
    /// # Ok::<(), edgefirst_tflite::Error>(())
    /// ```
    pub fn xnnpack(lib: &crate::Library, num_threads: i32) -> Result<Self> {
        // SAFETY: `lib.as_sys().library()` returns a reference to the loaded
        // TFLite library. `try_load` resolves optional symbols; missing
        // symbols return `None`.
        let fns =
            unsafe { XnnPackFunctions::try_load(lib.as_sys().library()) }.ok_or_else(|| {
                Error::invalid_argument(
                    "XNNPACK delegate symbols not found — \
                     the TFLite library may not have been compiled with XNNPACK support",
                )
            })?;

        // Get default options, then override num_threads.
        // SAFETY: `options_default` is a valid function pointer resolved above.
        let mut opts = unsafe { (fns.options_default)() };
        opts.num_threads = num_threads;

        // SAFETY: `create` is a valid function pointer. `opts` is properly
        // initialised from `options_default` with `num_threads` overridden.
        let raw = unsafe { (fns.create)(&opts) };
        let delegate = NonNull::new(raw)
            .ok_or_else(|| Error::null_pointer("TfLiteXNNPackDelegateCreate returned null"))?;

        let free = fns.delete;

        // Re-open the main TFLite library to keep it alive for the
        // delegate's lifetime. This increments the OS refcount at
        // near-zero cost.
        let tflite_lib = lib.reopen()?;

        Ok(Self {
            delegate,
            free,
            _lib: tflite_lib,
            #[cfg(feature = "dmabuf")]
            hal_dmabuf_fns: None,
            #[cfg(feature = "dmabuf")]
            hal_delegate_handle: None,
            #[cfg(feature = "dmabuf")]
            dmabuf_fns: None,
            #[cfg(feature = "camera_adaptor")]
            hal_camera_fns: None,
            #[cfg(feature = "camera_adaptor")]
            camera_adaptor_fns: None,
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
    ///
    /// Returns `Some` if the delegate exports either the HAL Delegate
    /// DMA-BUF API (`hal_dmabuf_*`) or the legacy `VxDelegate` DMA-BUF API.
    /// The HAL API is preferred when both are available.
    #[cfg(feature = "dmabuf")]
    #[must_use]
    pub fn dmabuf(&self) -> Option<crate::dmabuf::DmaBuf<'_>> {
        if self.hal_dmabuf_fns.is_some() || self.dmabuf_fns.is_some() {
            Some(crate::dmabuf::DmaBuf::new(
                self.delegate,
                self.hal_delegate_handle,
                self.hal_dmabuf_fns.as_ref(),
                self.dmabuf_fns.as_ref(),
            ))
        } else {
            None
        }
    }

    /// Returns `true` if this delegate supports DMA-BUF zero-copy.
    #[cfg(feature = "dmabuf")]
    #[must_use]
    pub fn has_dmabuf(&self) -> bool {
        self.hal_dmabuf_fns.is_some() || self.dmabuf_fns.is_some()
    }

    /// Access `CameraAdaptor` extensions if available on this delegate.
    ///
    /// Returns `Some` if the delegate exports either the HAL Delegate
    /// Camera Adaptor API (`hal_camera_adaptor_*`) or the legacy
    /// `VxDelegate` `CameraAdaptor` API. The HAL API is preferred when
    /// both are available.
    #[cfg(feature = "camera_adaptor")]
    #[must_use]
    pub fn camera_adaptor(&self) -> Option<crate::camera_adaptor::CameraAdaptor<'_>> {
        if self.hal_camera_fns.is_some() || self.camera_adaptor_fns.is_some() {
            // The CameraAdaptor HAL API reuses the same inner delegate handle
            // as the DMA-BUF HAL API — there is no separate
            // `hal_camera_adaptor_get_instance`.
            #[cfg(feature = "dmabuf")]
            let hal_handle = self.hal_delegate_handle;
            #[cfg(not(feature = "dmabuf"))]
            let hal_handle: Option<*mut std::ffi::c_void> = None;

            Some(crate::camera_adaptor::CameraAdaptor::new(
                self.delegate,
                hal_handle,
                self.hal_camera_fns.as_ref(),
                self.camera_adaptor_fns.as_ref(),
            ))
        } else {
            None
        }
    }

    /// Returns `true` if this delegate supports `CameraAdaptor`.
    #[cfg(feature = "camera_adaptor")]
    #[must_use]
    pub fn has_camera_adaptor(&self) -> bool {
        self.hal_camera_fns.is_some() || self.camera_adaptor_fns.is_some()
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Delegate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("Delegate");
        d.field("ptr", &self.delegate);

        #[cfg(feature = "dmabuf")]
        d.field("has_hal_dmabuf", &self.hal_dmabuf_fns.is_some());

        #[cfg(feature = "dmabuf")]
        d.field(
            "has_hal_delegate_handle",
            &self.hal_delegate_handle.is_some(),
        );

        #[cfg(feature = "dmabuf")]
        d.field("has_vx_dmabuf", &self.dmabuf_fns.is_some());

        #[cfg(feature = "camera_adaptor")]
        d.field("has_hal_camera_adaptor", &self.hal_camera_fns.is_some());

        #[cfg(feature = "camera_adaptor")]
        d.field("has_vx_camera_adaptor", &self.camera_adaptor_fns.is_some());

        d.finish_non_exhaustive()
    }
}

impl Drop for Delegate {
    fn drop(&mut self) {
        // SAFETY: The delegate pointer was created by the matching create
        // function (`tflite_plugin_create_delegate` for external delegates,
        // `TfLiteXNNPackDelegateCreate` for XNNPACK) and `free` is the
        // corresponding destroy function from the same library, which is
        // still loaded (held by `_lib`).
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
