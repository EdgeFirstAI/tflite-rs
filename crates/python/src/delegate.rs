// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Python `Delegate` and `DelegateRef` classes, plus `load_delegate()`.

use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::prelude::*;

use crate::error::{self, InvalidArgumentError};

// ---------------------------------------------------------------------------
// Delegate — owned, passed to Interpreter constructor
// ---------------------------------------------------------------------------

/// An external `TFLite` delegate for hardware acceleration.
///
/// Created via `load_delegate()`. Passed to the `Interpreter` constructor
/// via `experimental_delegates`. Consumed on use (cannot be reused).
#[pyclass(name = "Delegate", unsendable)]
pub struct PyDelegate {
    inner: Option<edgefirst_tflite::Delegate>,
}

impl std::fmt::Debug for PyDelegate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Delegate")
            .field("active", &self.inner.is_some())
            .finish()
    }
}

impl PyDelegate {
    /// Take the inner delegate, transferring ownership to the caller.
    pub fn take_inner(&mut self) -> Option<edgefirst_tflite::Delegate> {
        self.inner.take()
    }

    /// Borrow the inner delegate, or return an error if already consumed.
    fn inner_ref(&self) -> PyResult<&edgefirst_tflite::Delegate> {
        self.inner
            .as_ref()
            .ok_or_else(|| InvalidArgumentError::new_err("delegate already consumed"))
    }
}

#[pymethods]
impl PyDelegate {
    /// Whether this delegate supports DMA-BUF zero-copy.
    #[getter]
    fn has_dmabuf(&self) -> PyResult<bool> {
        Ok(self.inner_ref()?.has_dmabuf())
    }

    /// Whether this delegate supports `CameraAdaptor`.
    #[getter]
    fn has_camera_adaptor(&self) -> PyResult<bool> {
        Ok(self.inner_ref()?.has_camera_adaptor())
    }

    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "Delegate(active)".to_string()
        } else {
            "Delegate(consumed)".to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// DelegateRef — borrowed reference through Interpreter
// ---------------------------------------------------------------------------

/// Borrowed reference to a delegate owned by an Interpreter.
///
/// Access via `Interpreter.delegate(index)`. Holds a reference to the
/// interpreter to prevent garbage collection and to access delegate
/// capabilities and extensions.
#[pyclass(name = "DelegateRef", unsendable)]
pub struct PyDelegateRef {
    pub(crate) interp: Py<crate::interpreter::PyInterpreter>,
    pub(crate) delegate_index: usize,
}

impl std::fmt::Debug for PyDelegateRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DelegateRef")
            .field("delegate_index", &self.delegate_index)
            .finish_non_exhaustive()
    }
}

#[pymethods]
impl PyDelegateRef {
    /// The delegate index within the interpreter.
    #[getter]
    fn index(&self) -> usize {
        self.delegate_index
    }

    /// Whether this delegate supports DMA-BUF zero-copy.
    #[getter]
    fn has_dmabuf(&self, py: Python<'_>) -> PyResult<bool> {
        let interp = self.interp.bind(py).borrow();
        interp.with_delegate(self.delegate_index, |d| Ok(d.has_dmabuf()))
    }

    /// Whether this delegate supports `CameraAdaptor`.
    #[getter]
    fn has_camera_adaptor(&self, py: Python<'_>) -> PyResult<bool> {
        let interp = self.interp.bind(py).borrow();
        interp.with_delegate(self.delegate_index, |d| Ok(d.has_camera_adaptor()))
    }

    /// Get a `DmaBuf` interface for this delegate's DMA-BUF extensions.
    ///
    /// Returns `None` if the delegate does not support DMA-BUF.
    fn dmabuf(&self, py: Python<'_>) -> PyResult<Option<crate::dmabuf::PyDmaBuf>> {
        let interp = self.interp.bind(py).borrow();
        interp.with_delegate(self.delegate_index, |d| {
            Ok(d.has_dmabuf().then(|| crate::dmabuf::PyDmaBuf {
                interp: self.interp.clone_ref(py),
                delegate_index: self.delegate_index,
            }))
        })
    }

    /// Get a `CameraAdaptor` interface for this delegate's NPU preprocessing.
    ///
    /// Returns `None` if the delegate does not support `CameraAdaptor`.
    fn camera_adaptor(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<crate::camera_adaptor::PyCameraAdaptor>> {
        let interp = self.interp.bind(py).borrow();
        interp.with_delegate(self.delegate_index, |d| {
            Ok(d.has_camera_adaptor()
                .then(|| crate::camera_adaptor::PyCameraAdaptor {
                    interp: self.interp.clone_ref(py),
                    delegate_index: self.delegate_index,
                }))
        })
    }

    fn __repr__(&self) -> String {
        format!("DelegateRef(index={})", self.delegate_index)
    }
}

// ---------------------------------------------------------------------------
// load_delegate() — module-level function
// ---------------------------------------------------------------------------

/// Load an external delegate from a shared library.
///
/// Args:
///     library: Path to the delegate shared library (e.g., `libvx_delegate.so`).
///     options: Optional dict of key-value configuration options.
///
/// Returns:
///     A `Delegate` object to pass to `Interpreter(experimental_delegates=[...])`.
#[pyfunction]
#[pyo3(signature = (library, options=None))]
#[allow(clippy::needless_pass_by_value, clippy::implicit_hasher)]
pub fn load_delegate(
    library: PathBuf,
    options: Option<HashMap<String, String>>,
) -> PyResult<PyDelegate> {
    let delegate = if let Some(opts) = options {
        let mut delegate_options = edgefirst_tflite::DelegateOptions::new();
        for (k, v) in opts {
            delegate_options = delegate_options.option(k, v);
        }
        edgefirst_tflite::Delegate::load_with_options(&library, &delegate_options)
    } else {
        edgefirst_tflite::Delegate::load(&library)
    };
    let delegate = delegate.map_err(error::to_py_err)?;
    Ok(PyDelegate {
        inner: Some(delegate),
    })
}
