// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Python `CameraAdaptor` class for NPU-accelerated preprocessing.

use pyo3::prelude::*;

use crate::error::{self, TfLiteError};
use crate::interpreter::PyInterpreter;

/// `CameraAdaptor` interface for `VxDelegate` NPU format conversion.
///
/// Obtained via `interp.delegate(0)` and accessing `CameraAdaptor` methods.
/// The interpreter must remain alive while this object is in use.
#[pyclass(name = "CameraAdaptor", unsendable)]
pub struct PyCameraAdaptor {
    pub(crate) interp: Py<PyInterpreter>,
    pub(crate) delegate_index: usize,
}

impl std::fmt::Debug for PyCameraAdaptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CameraAdaptor")
            .field("delegate_index", &self.delegate_index)
            .finish_non_exhaustive()
    }
}

impl PyCameraAdaptor {
    /// Helper: borrow interpreter → delegate → `camera_adaptor`, call `f`.
    fn with_adaptor<F, R>(&self, py: Python<'_>, f: F) -> PyResult<R>
    where
        F: FnOnce(&edgefirst_tflite::camera_adaptor::CameraAdaptor<'_>) -> PyResult<R>,
    {
        let interp = self.interp.bind(py).borrow();
        interp.with_delegate(self.delegate_index, |delegate| {
            let adaptor = delegate.camera_adaptor().ok_or_else(|| {
                TfLiteError::new_err("CameraAdaptor not available on this delegate")
            })?;
            f(&adaptor)
        })
    }
}

#[pymethods]
impl PyCameraAdaptor {
    /// Set the camera format for an input tensor.
    fn set_format(&self, py: Python<'_>, tensor_index: i32, format: &str) -> PyResult<()> {
        self.with_adaptor(py, |a| {
            a.set_format(tensor_index, format).map_err(error::to_py_err)
        })
    }

    /// Set camera format with resize and letterbox options.
    #[pyo3(signature = (tensor_index, format, width, height, letterbox=false, letterbox_color=0))]
    #[allow(clippy::too_many_arguments)]
    fn set_format_ex(
        &self,
        py: Python<'_>,
        tensor_index: i32,
        format: &str,
        width: u32,
        height: u32,
        letterbox: bool,
        letterbox_color: u32,
    ) -> PyResult<()> {
        self.with_adaptor(py, |a| {
            a.set_format_ex(
                tensor_index,
                format,
                width,
                height,
                letterbox,
                letterbox_color,
            )
            .map_err(error::to_py_err)
        })
    }

    /// Set explicit camera and model formats.
    fn set_formats(
        &self,
        py: Python<'_>,
        tensor_index: i32,
        camera_format: &str,
        model_format: &str,
    ) -> PyResult<()> {
        self.with_adaptor(py, |a| {
            a.set_formats(tensor_index, camera_format, model_format)
                .map_err(error::to_py_err)
        })
    }

    /// Set camera format using a V4L2 `FourCC` code.
    fn set_fourcc(&self, py: Python<'_>, tensor_index: i32, fourcc: u32) -> PyResult<()> {
        self.with_adaptor(py, |a| {
            a.set_fourcc(tensor_index, fourcc).map_err(error::to_py_err)
        })
    }

    /// Get the current format for an input tensor.
    fn format(&self, py: Python<'_>, tensor_index: i32) -> PyResult<Option<String>> {
        self.with_adaptor(py, |a| Ok(a.format(tensor_index)))
    }

    /// Check if a format string is supported.
    fn is_supported(&self, py: Python<'_>, format: &str) -> PyResult<bool> {
        self.with_adaptor(py, |a| Ok(a.is_supported(format)))
    }

    /// Get the number of input channels for a format.
    fn input_channels(&self, py: Python<'_>, format: &str) -> PyResult<i32> {
        self.with_adaptor(py, |a| Ok(a.input_channels(format)))
    }

    /// Get the number of output channels for a format.
    fn output_channels(&self, py: Python<'_>, format: &str) -> PyResult<i32> {
        self.with_adaptor(py, |a| Ok(a.output_channels(format)))
    }

    /// Get the `FourCC` code string for a format.
    fn fourcc(&self, py: Python<'_>, format: &str) -> PyResult<Option<String>> {
        self.with_adaptor(py, |a| Ok(a.fourcc(format)))
    }

    /// Convert a `FourCC` code to a format string.
    #[allow(clippy::wrong_self_convention)]
    fn from_fourcc(&self, py: Python<'_>, fourcc: &str) -> PyResult<Option<String>> {
        self.with_adaptor(py, |a| Ok(a.from_fourcc(fourcc)))
    }
}
