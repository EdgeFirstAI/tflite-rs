// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Safe `CameraAdaptor` API for NPU-accelerated preprocessing.
//!
//! This module provides [`CameraAdaptor`], a safe wrapper around the
//! `VxDelegate` `CameraAdaptor` C API. It configures the delegate to inject
//! format conversion operations (e.g., RGBA to RGB) into the TIM-VX graph,
//! running them on the NPU instead of the CPU.
//!
//! # Example
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

use std::ffi::{CStr, CString};
use std::ptr::NonNull;

use edgefirst_tflite_sys::vx_ffi::VxCameraAdaptorFunctions;
use edgefirst_tflite_sys::TfLiteDelegate;

use crate::error::{self, Error, Result};

// ---------------------------------------------------------------------------
// CameraAdaptor
// ---------------------------------------------------------------------------

/// Safe interface for `VxDelegate` `CameraAdaptor` format conversion on NPU.
///
/// Obtained from [`Delegate::camera_adaptor()`](crate::Delegate::camera_adaptor)
/// when the loaded delegate supports `CameraAdaptor`.
#[derive(Debug)]
pub struct CameraAdaptor<'a> {
    delegate: NonNull<TfLiteDelegate>,
    fns: &'a VxCameraAdaptorFunctions,
}

impl<'a> CameraAdaptor<'a> {
    /// Create a new `CameraAdaptor` wrapper.
    pub(crate) fn new(
        delegate: NonNull<TfLiteDelegate>,
        fns: &'a VxCameraAdaptorFunctions,
    ) -> Self {
        Self { delegate, fns }
    }

    /// Set the camera format for an input tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    pub fn set_format(&self, tensor_index: i32, format: &str) -> Result<()> {
        let c_format = CString::new(format)
            .map_err(|_| Error::invalid_argument("format contains interior NUL byte"))?;
        // SAFETY: `self.delegate` is a valid delegate pointer; `c_format` is a
        // valid NUL-terminated C string (from `CString::new` above). Function
        // pointers were loaded from the same library that created the delegate.
        let status = unsafe {
            (self.fns.set_format)(self.delegate.as_ptr(), tensor_index, c_format.as_ptr())
        };
        error::status_to_result(status)
    }

    /// Set camera format with resize and letterbox options.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
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
        let c_format = CString::new(format)
            .map_err(|_| Error::invalid_argument("format contains interior NUL byte"))?;
        // SAFETY: delegate pointer is valid; `c_format` is a valid C string.
        let status = unsafe {
            (self.fns.set_format_ex)(
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
    pub fn set_formats(
        &self,
        tensor_index: i32,
        camera_format: &str,
        model_format: &str,
    ) -> Result<()> {
        let c_camera = CString::new(camera_format)
            .map_err(|_| Error::invalid_argument("camera_format contains interior NUL byte"))?;
        let c_model = CString::new(model_format)
            .map_err(|_| Error::invalid_argument("model_format contains interior NUL byte"))?;
        // SAFETY: delegate pointer is valid; both format strings are valid C strings.
        let status = unsafe {
            (self.fns.set_formats)(
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
    pub fn set_fourcc(&self, tensor_index: i32, fourcc: u32) -> Result<()> {
        // SAFETY: delegate pointer is valid.
        let status = unsafe { (self.fns.set_fourcc)(self.delegate.as_ptr(), tensor_index, fourcc) };
        error::status_to_result(status)
    }

    /// Get the current format for an input tensor.
    #[must_use]
    pub fn format(&self, tensor_index: i32) -> Option<String> {
        // SAFETY: delegate pointer is valid.
        let ptr = unsafe { (self.fns.get_format)(self.delegate.as_ptr(), tensor_index) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and points to a NUL-terminated string
        // owned by the delegate (valid for the delegate's lifetime).
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str().ok().map(String::from)
    }

    /// Check if a format string is supported.
    #[must_use]
    pub fn is_supported(&self, format: &str) -> bool {
        let Ok(c_format) = CString::new(format) else {
            return false;
        };
        // SAFETY: `c_format` is a valid NUL-terminated C string.
        unsafe { (self.fns.is_supported)(c_format.as_ptr()) }
    }

    /// Get the number of input channels for a format.
    #[must_use]
    pub fn input_channels(&self, format: &str) -> i32 {
        let Ok(c_format) = CString::new(format) else {
            return 0;
        };
        // SAFETY: `c_format` is a valid NUL-terminated C string.
        unsafe { (self.fns.get_input_channels)(c_format.as_ptr()) }
    }

    /// Get the number of output channels for a format.
    #[must_use]
    pub fn output_channels(&self, format: &str) -> i32 {
        let Ok(c_format) = CString::new(format) else {
            return 0;
        };
        // SAFETY: `c_format` is a valid NUL-terminated C string.
        unsafe { (self.fns.get_output_channels)(c_format.as_ptr()) }
    }

    /// Get the `FourCC` code string for a format.
    #[must_use]
    pub fn fourcc(&self, format: &str) -> Option<String> {
        let c_format = CString::new(format).ok()?;
        // SAFETY: `c_format` is a valid NUL-terminated C string.
        let ptr = unsafe { (self.fns.get_fourcc)(c_format.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and points to a static NUL-terminated string.
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str().ok().map(String::from)
    }

    /// Convert a `FourCC` code to a format string.
    #[must_use]
    pub fn from_fourcc(&self, fourcc: &str) -> Option<String> {
        let c_fourcc = CString::new(fourcc).ok()?;
        // SAFETY: `c_fourcc` is a valid NUL-terminated C string.
        let ptr = unsafe { (self.fns.from_fourcc)(c_fourcc.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and points to a static NUL-terminated string.
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str().ok().map(String::from)
    }
}
