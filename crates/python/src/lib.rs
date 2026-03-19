// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Python bindings for `edgefirst-tflite`.
//!
//! This crate provides a Python package (`edgefirst_tflite`) that is
//! API-compatible with `tflite_runtime.interpreter.Interpreter` for the
//! core inference path, while exposing `EdgeFirst` extensions (DMA-BUF,
//! `CameraAdaptor`) and model metadata.

mod camera_adaptor;
mod delegate;
mod dmabuf;
mod error;
mod interpreter;
mod metadata;
mod tensor_utils;

use pyo3::prelude::*;

/// The `edgefirst_tflite` Python module.
#[pymodule]
fn edgefirst_tflite(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version from Cargo.toml
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Exception hierarchy
    error::register(m)?;

    // Classes
    m.add_class::<interpreter::PyInterpreter>()?;
    m.add_class::<delegate::PyDelegate>()?;
    m.add_class::<delegate::PyDelegateRef>()?;
    m.add_class::<dmabuf::PyDmaBuf>()?;
    m.add_class::<camera_adaptor::PyCameraAdaptor>()?;
    m.add_class::<metadata::PyMetadata>()?;

    // Module-level functions
    m.add_function(wrap_pyfunction!(delegate::load_delegate, m)?)?;

    Ok(())
}
