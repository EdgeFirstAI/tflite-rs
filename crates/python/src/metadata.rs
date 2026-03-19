// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Python `Metadata` class for `TFLite` model metadata extraction.

use pyo3::prelude::*;

/// Extracted metadata from a `TFLite` model file.
#[derive(Debug)]
#[pyclass(name = "Metadata")]
pub struct PyMetadata {
    pub(crate) inner: edgefirst_tflite::metadata::Metadata,
}

#[pymethods]
impl PyMetadata {
    /// Model name.
    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }

    /// Model version.
    #[getter]
    fn version(&self) -> Option<&str> {
        self.inner.version.as_deref()
    }

    /// Model description.
    #[getter]
    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }

    /// Model author.
    #[getter]
    fn author(&self) -> Option<&str> {
        self.inner.author.as_deref()
    }

    /// Model license.
    #[getter]
    fn license(&self) -> Option<&str> {
        self.inner.license.as_deref()
    }

    /// Minimum parser version required.
    #[getter]
    fn min_parser_version(&self) -> Option<&str> {
        self.inner.min_parser_version.as_deref()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}
