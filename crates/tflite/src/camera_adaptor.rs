// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Safe `CameraAdaptor` API for NPU-accelerated preprocessing.
//!
//! This module provides [`CameraAdaptor`], a safe wrapper around the HAL
//! Delegate Camera Adaptor API (`hal_camera_adaptor_*`) and the legacy
//! `VxDelegate` `CameraAdaptor` C API. The HAL API is the primary interface;
//! legacy `VxDelegate`-specific methods are deprecated and will be removed
//! in a future release.
//!
//! # Primary API (HAL Delegate)
//!
//! ```no_run
//! use edgefirst_tflite::Delegate;
//!
//! let delegate = Delegate::load("libvx_delegate.so")?;
//! let adaptor = delegate.camera_adaptor().expect("CameraAdaptor not supported");
//!
//! if adaptor.is_format_supported("rgba") {
//!     let info = adaptor.format_info("rgba")?;
//!     println!("in={}, out={}, fourcc={}", info.input_channels, info.output_channels, info.fourcc);
//! }
//! # Ok::<(), edgefirst_tflite::Error>(())
//! ```
//!
//! # Legacy API (`VxDelegate`)
//!
//! ```no_run
//! use edgefirst_tflite::Delegate;
//!
//! let delegate = Delegate::load("libvx_delegate.so")?;
//! let adaptor = delegate.camera_adaptor().expect("CameraAdaptor not supported");
//!
//! // Configure RGBA -> RGB conversion on the NPU
//! adaptor.set_format(0, "rgba")?;
//! # Ok::<(), edgefirst_tflite::Error>(())
//! ```

use std::ffi::{c_void, CStr, CString};
use std::ptr::NonNull;

use edgefirst_tflite_sys::hal_ffi::{HalCameraAdaptorFormatInfo, HalCameraAdaptorFunctions};
use edgefirst_tflite_sys::vx_ffi::VxCameraAdaptorFunctions;
use edgefirst_tflite_sys::TfLiteDelegate;

use crate::error::{self, Error, Result};

// ---------------------------------------------------------------------------
// FormatInfo
// ---------------------------------------------------------------------------

/// Camera adaptor format information returned by [`CameraAdaptor::format_info`].
///
/// Describes the channel layout and V4L2 `FourCC` code for a camera format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatInfo {
    /// Number of input channels for this format.
    pub input_channels: i32,
    /// Number of output channels for this format.
    pub output_channels: i32,
    /// V4L2 `FourCC` code string.
    pub fourcc: String,
}

// ---------------------------------------------------------------------------
// CameraAdaptor
// ---------------------------------------------------------------------------

/// Safe interface for delegate camera format conversion on NPU.
///
/// Obtained from [`Delegate::camera_adaptor()`](crate::Delegate::camera_adaptor)
/// when the loaded delegate supports `CameraAdaptor`. Uses the HAL Delegate
/// Camera Adaptor API (`hal_camera_adaptor_*`) as the primary backend, with
/// legacy `VxDelegate` symbols as a fallback.
#[derive(Debug)]
pub struct CameraAdaptor<'a> {
    delegate: NonNull<TfLiteDelegate>,
    /// Inner delegate handle from `hal_dmabuf_get_instance()`.
    ///
    /// The Camera Adaptor HAL API reuses the same opaque `hal_delegate_t`
    /// handle as the DMA-BUF HAL API — there is no separate
    /// `hal_camera_adaptor_get_instance`. It is `None` when HAL symbols are
    /// not present or `get_instance()` returned null.
    hal_handle: Option<*mut c_void>,
    hal_fns: Option<&'a HalCameraAdaptorFunctions>,
    vx_fns: Option<&'a VxCameraAdaptorFunctions>,
}

impl<'a> CameraAdaptor<'a> {
    /// Create a new `CameraAdaptor` wrapper with HAL and/or `VxDelegate` backends.
    pub(crate) fn new(
        delegate: NonNull<TfLiteDelegate>,
        hal_handle: Option<*mut c_void>,
        hal_fns: Option<&'a HalCameraAdaptorFunctions>,
        vx_fns: Option<&'a VxCameraAdaptorFunctions>,
    ) -> Self {
        Self {
            delegate,
            hal_handle,
            hal_fns,
            vx_fns,
        }
    }

    /// Returns the inner `hal_delegate_t` handle for HAL API calls.
    ///
    /// Returns the handle from `hal_dmabuf_get_instance()` when available,
    /// falling back to casting the outer `TfLiteDelegate*` pointer. The
    /// fallback is kept for delegates that expose HAL symbols but return a
    /// null instance handle (should not happen in practice).
    fn hal_delegate_ptr(&self) -> *mut c_void {
        self.hal_handle
            .unwrap_or_else(|| self.delegate.as_ptr().cast::<c_void>())
    }

    // =======================================================================
    // Primary API (HAL Delegate Camera Adaptor)
    // =======================================================================

    /// Check if a format string is supported by this delegate.
    ///
    /// Uses the HAL API when available, falling back to `VxDelegate`.
    #[must_use]
    pub fn is_format_supported(&self, format: &str) -> bool {
        let Ok(c_format) = CString::new(format) else {
            return false;
        };
        if let Some(hal) = self.hal_fns {
            // SAFETY: delegate pointer is valid and cast to the opaque
            // hal_delegate_t (void*) that the HAL API expects. `c_format`
            // is a valid NUL-terminated C string.
            unsafe { (hal.is_supported)(self.hal_delegate_ptr(), c_format.as_ptr()) == 1 }
        } else if let Some(vx) = self.vx_fns {
            // SAFETY: `c_format` is a valid NUL-terminated C string.
            unsafe { (vx.is_supported)(c_format.as_ptr()) }
        } else {
            false
        }
    }

    /// Get format information for a camera format string.
    ///
    /// Returns the input/output channel counts and V4L2 `FourCC` code.
    ///
    /// This method requires the HAL Delegate Camera Adaptor API. When
    /// only the legacy `VxDelegate` backend is available, the information
    /// is assembled from the individual `VxDelegate` query functions.
    ///
    /// # Errors
    ///
    /// Returns an error if the format is not supported or the underlying
    /// API call fails.
    pub fn format_info(&self, format: &str) -> Result<FormatInfo> {
        let c_format = CString::new(format)
            .map_err(|_| Error::invalid_argument("format contains interior NUL byte"))?;

        if let Some(hal) = self.hal_fns {
            let mut info = HalCameraAdaptorFormatInfo::default();
            // SAFETY: delegate pointer is valid. `info` is a valid mutable
            // reference that the C function will populate. We pass the struct
            // size for ABI versioning compatibility.
            let ret = unsafe {
                (hal.get_format_info)(
                    self.hal_delegate_ptr(),
                    c_format.as_ptr(),
                    &mut info,
                    std::mem::size_of::<HalCameraAdaptorFormatInfo>(),
                )
            };
            error::hal_to_result(ret, "hal_camera_adaptor_get_format_info")?;

            // Extract fourcc as a string, trimming NUL padding.
            let fourcc_len = info
                .fourcc
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(info.fourcc.len());
            let fourcc = String::from_utf8_lossy(&info.fourcc[..fourcc_len]).into_owned();

            Ok(FormatInfo {
                input_channels: info.input_channels,
                output_channels: info.output_channels,
                fourcc,
            })
        } else if let Some(vx) = self.vx_fns {
            // Fallback: assemble from individual VxDelegate queries.
            // SAFETY: `c_format` is a valid NUL-terminated C string.
            let input_channels = unsafe { (vx.get_input_channels)(c_format.as_ptr()) };
            let output_channels = unsafe { (vx.get_output_channels)(c_format.as_ptr()) };

            // SAFETY: `c_format` is a valid NUL-terminated C string.
            let fourcc_ptr = unsafe { (vx.get_fourcc)(c_format.as_ptr()) };
            let fourcc = if fourcc_ptr.is_null() {
                String::new()
            } else {
                // SAFETY: `fourcc_ptr` is non-null and points to a static
                // NUL-terminated string.
                unsafe { CStr::from_ptr(fourcc_ptr) }
                    .to_str()
                    .unwrap_or("")
                    .to_owned()
            };

            Ok(FormatInfo {
                input_channels,
                output_channels,
                fourcc,
            })
        } else {
            Err(Error::invalid_argument(
                "no CameraAdaptor backend available",
            ))
        }
    }

    // =======================================================================
    // Legacy `VxDelegate` API (deprecated)
    // =======================================================================

    /// Helper: get the `VxDelegate` function pointers or return an error.
    fn vx(&self) -> Result<&VxCameraAdaptorFunctions> {
        self.vx_fns.ok_or_else(|| {
            Error::invalid_argument(
                "this method requires the `VxDelegate` CameraAdaptor API, which is not available",
            )
        })
    }

    /// Set the camera format for an input tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    pub fn set_format(&self, tensor_index: i32, format: &str) -> Result<()> {
        let vx = self.vx()?;
        let c_format = CString::new(format)
            .map_err(|_| Error::invalid_argument("format contains interior NUL byte"))?;
        // SAFETY: `self.delegate` is a valid delegate pointer; `c_format` is a
        // valid NUL-terminated C string (from `CString::new` above). Function
        // pointers were loaded from the same library that created the delegate.
        let status =
            unsafe { (vx.set_format)(self.delegate.as_ptr(), tensor_index, c_format.as_ptr()) };
        error::status_to_result(status)
    }

    /// Set camera format with resize and letterbox options.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[allow(clippy::too_many_arguments)]
    pub fn set_format_ex(
        &self,
        tensor_index: i32,
        format: &str,
        width: u32,
        height: u32,
        letterbox: bool,
        letterbox_color: u32,
    ) -> Result<()> {
        let vx = self.vx()?;
        let c_format = CString::new(format)
            .map_err(|_| Error::invalid_argument("format contains interior NUL byte"))?;
        // SAFETY: delegate pointer is valid; `c_format` is a valid C string.
        let status = unsafe {
            (vx.set_format_ex)(
                self.delegate.as_ptr(),
                tensor_index,
                c_format.as_ptr(),
                width,
                height,
                letterbox,
                letterbox_color,
            )
        };
        error::status_to_result(status)
    }

    /// Set explicit camera and model formats.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    pub fn set_formats(
        &self,
        tensor_index: i32,
        camera_format: &str,
        model_format: &str,
    ) -> Result<()> {
        let vx = self.vx()?;
        let c_camera = CString::new(camera_format)
            .map_err(|_| Error::invalid_argument("camera_format contains interior NUL byte"))?;
        let c_model = CString::new(model_format)
            .map_err(|_| Error::invalid_argument("model_format contains interior NUL byte"))?;
        // SAFETY: delegate pointer is valid; both format strings are valid C strings.
        let status = unsafe {
            (vx.set_formats)(
                self.delegate.as_ptr(),
                tensor_index,
                c_camera.as_ptr(),
                c_model.as_ptr(),
            )
        };
        error::status_to_result(status)
    }

    /// Set camera format using a V4L2 `FourCC` code.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    pub fn set_fourcc(&self, tensor_index: i32, fourcc: u32) -> Result<()> {
        let vx = self.vx()?;
        // SAFETY: delegate pointer is valid.
        let status = unsafe { (vx.set_fourcc)(self.delegate.as_ptr(), tensor_index, fourcc) };
        error::status_to_result(status)
    }

    /// Get the current format for an input tensor.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[must_use]
    pub fn format(&self, tensor_index: i32) -> Option<String> {
        let vx = self.vx_fns?;
        // SAFETY: delegate pointer is valid.
        let ptr = unsafe { (vx.get_format)(self.delegate.as_ptr(), tensor_index) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and points to a NUL-terminated string
        // owned by the delegate (valid for the delegate's lifetime).
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str().ok().map(String::from)
    }

    /// Check if a format string is supported.
    #[deprecated(
        note = "`VxDelegate`-specific, use `is_format_supported()` instead; will be removed in a future release"
    )]
    #[must_use]
    pub fn is_supported(&self, format: &str) -> bool {
        let Ok(c_format) = CString::new(format) else {
            return false;
        };
        if let Some(vx) = self.vx_fns {
            // SAFETY: `c_format` is a valid NUL-terminated C string.
            unsafe { (vx.is_supported)(c_format.as_ptr()) }
        } else {
            false
        }
    }

    /// Get the number of input channels for a format.
    #[deprecated(
        note = "`VxDelegate`-specific, use `format_info()` instead; will be removed in a future release"
    )]
    #[must_use]
    pub fn input_channels(&self, format: &str) -> i32 {
        let Ok(c_format) = CString::new(format) else {
            return 0;
        };
        self.vx_fns.map_or(0, |vx| {
            // SAFETY: `c_format` is a valid NUL-terminated C string.
            unsafe { (vx.get_input_channels)(c_format.as_ptr()) }
        })
    }

    /// Get the number of output channels for a format.
    #[deprecated(
        note = "`VxDelegate`-specific, use `format_info()` instead; will be removed in a future release"
    )]
    #[must_use]
    pub fn output_channels(&self, format: &str) -> i32 {
        let Ok(c_format) = CString::new(format) else {
            return 0;
        };
        self.vx_fns.map_or(0, |vx| {
            // SAFETY: `c_format` is a valid NUL-terminated C string.
            unsafe { (vx.get_output_channels)(c_format.as_ptr()) }
        })
    }

    /// Get the `FourCC` code string for a format.
    #[deprecated(
        note = "`VxDelegate`-specific, use `format_info()` instead; will be removed in a future release"
    )]
    #[must_use]
    pub fn fourcc(&self, format: &str) -> Option<String> {
        let vx = self.vx_fns?;
        let c_format = CString::new(format).ok()?;
        // SAFETY: `c_format` is a valid NUL-terminated C string.
        let ptr = unsafe { (vx.get_fourcc)(c_format.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and points to a static NUL-terminated string.
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str().ok().map(String::from)
    }

    /// Convert a `FourCC` code to a format string.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[must_use]
    pub fn from_fourcc(&self, fourcc: &str) -> Option<String> {
        let vx = self.vx_fns?;
        let c_fourcc = CString::new(fourcc).ok()?;
        // SAFETY: `c_fourcc` is a valid NUL-terminated C string.
        let ptr = unsafe { (vx.from_fourcc)(c_fourcc.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and points to a static NUL-terminated string.
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str().ok().map(String::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_info_debug() {
        let info = FormatInfo {
            input_channels: 4,
            output_channels: 3,
            fourcc: "RGBA".to_owned(),
        };
        let debug = format!("{info:?}");
        assert!(debug.contains("FormatInfo"));
        assert!(debug.contains("input_channels"));
        assert!(debug.contains("RGBA"));
    }

    #[test]
    fn format_info_clone_eq() {
        let info = FormatInfo {
            input_channels: 4,
            output_channels: 3,
            fourcc: "RGBA".to_owned(),
        };
        let cloned = info.clone();
        assert_eq!(info, cloned);
    }
}
