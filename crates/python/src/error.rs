// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Python exception hierarchy for `edgefirst-tflite`.

use pyo3::prelude::*;

pyo3::create_exception!(
    edgefirst_tflite,
    TfLiteError,
    pyo3::exceptions::PyRuntimeError
);
pyo3::create_exception!(edgefirst_tflite, LibraryError, TfLiteError);
pyo3::create_exception!(edgefirst_tflite, DelegateError, TfLiteError);
pyo3::create_exception!(edgefirst_tflite, InvalidArgumentError, TfLiteError);

/// Convert an [`edgefirst_tflite::Error`] to the appropriate Python exception.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn to_py_err(err: edgefirst_tflite::Error) -> PyErr {
    let msg = err.to_string();
    if err.is_library_error() {
        LibraryError::new_err(msg)
    } else if err.is_delegate_error() {
        DelegateError::new_err(msg)
    } else if err.is_invalid_argument() {
        InvalidArgumentError::new_err(msg)
    } else {
        TfLiteError::new_err(msg)
    }
}

/// Register exception types on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("TfLiteError", m.py().get_type::<TfLiteError>())?;
    m.add("LibraryError", m.py().get_type::<LibraryError>())?;
    m.add("DelegateError", m.py().get_type::<DelegateError>())?;
    m.add(
        "InvalidArgumentError",
        m.py().get_type::<InvalidArgumentError>(),
    )?;
    Ok(())
}
