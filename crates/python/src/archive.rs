// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Python `ModelArchive` class for the ZIP-trailer metadata.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::error::to_py_err;

/// Python wrapper around `edgefirst_tflite::archive::ModelArchive`.
///
/// Owns the model bytes so the embedded `zip::ZipArchive<Cursor<&[u8]>>`
/// has a stable backing buffer for the wrapper's lifetime.
#[pyclass(name = "ModelArchive", unsendable)]
#[derive(Debug)]
pub struct PyModelArchive {
    // The bytes outlive the archive (it is `'static` over `bytes`).
    bytes: Box<[u8]>,
    // SAFETY: archive borrows `bytes` for the lifetime of `self`.
    // pinning is handled by `Box<[u8]>` (heap address never moves) and
    // `unsendable` prevents the wrapper from being moved across threads
    // in a way that could invalidate the borrow.
    archive: edgefirst_tflite::archive::ModelArchive<'static>,
}

impl PyModelArchive {
    /// Build a wrapper from owned bytes.
    pub(crate) fn from_owned_bytes(bytes: Vec<u8>) -> PyResult<Self> {
        let bytes: Box<[u8]> = bytes.into_boxed_slice();
        // SAFETY: `bytes` is stored in `Self.bytes` (Box never reallocates),
        // so the slice reference handed to ModelArchive remains valid for
        // as long as `Self` lives. We extend the lifetime to 'static to
        // satisfy pyclass which forbids non-'static borrows; the actual
        // borrow is constrained by the wrapper's lifetime.
        let slice: &'static [u8] = unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) };
        let archive = edgefirst_tflite::archive::ModelArchive::new(slice).map_err(to_py_err)?;
        Ok(Self { bytes, archive })
    }
}

#[pymethods]
impl PyModelArchive {
    /// Open the embedded ZIP archive of a `.tflite` model.
    ///
    /// Pass either ``path`` (filesystem path) or ``content`` (raw bytes).
    #[new]
    #[pyo3(signature = (path=None, *, content=None))]
    fn new(path: Option<PathBuf>, content: Option<Vec<u8>>) -> PyResult<Self> {
        let bytes = match (path, content) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "specify either `path` or `content`, not both",
                ));
            }
            (Some(p), None) => std::fs::read(&p)
                .map_err(|e| PyValueError::new_err(format!("read {}: {e}", p.display())))?,
            (None, Some(c)) => c,
            (None, None) => {
                return Err(PyValueError::new_err(
                    "ModelArchive requires `path` or `content`",
                ));
            }
        };
        Self::from_owned_bytes(bytes)
    }

    /// Number of entries in the archive.
    fn __len__(&self) -> usize {
        self.archive.len()
    }

    /// True if the archive contains an entry with this name.
    fn __contains__(&self, name: &str) -> bool {
        self.archive.contains(name)
    }

    /// List the names of all entries in the archive.
    fn entry_names(&self) -> Vec<String> {
        self.archive.entry_names().map(str::to_owned).collect()
    }

    /// Read an entry by name as bytes.
    fn read<'py>(&mut self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let buf = self.archive.read(name).map_err(to_py_err)?;
        Ok(PyBytes::new(py, &buf))
    }

    /// Read an entry by name as a UTF-8 string.
    fn read_to_string(&mut self, name: &str) -> PyResult<String> {
        self.archive.read_to_string(name).map_err(to_py_err)
    }

    /// Read ``edgefirst.json`` as a UTF-8 string.
    fn edgefirst_json(&mut self) -> PyResult<String> {
        self.archive.edgefirst_json().map_err(to_py_err)
    }

    /// Read ``labels.txt`` and split into one label per line.
    fn labels(&mut self) -> PyResult<Vec<String>> {
        self.archive.labels().map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelArchive(entries={}, bytes={})",
            self.archive.len(),
            self.bytes.len()
        )
    }
}

/// Module-level helper: probe whether a byte buffer ends with a valid
/// ZIP archive (the signature the EdgeFirst converter appends).
#[pyfunction]
#[pyo3(signature = (content))]
pub fn has_archive(content: &[u8]) -> bool {
    edgefirst_tflite::archive::has_archive(content)
}
