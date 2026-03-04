// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Model loading for `TFLite` inference.

use std::os::raw::c_void;
use std::path::Path;
use std::ptr::NonNull;

use edgefirst_tflite_sys::TfLiteModel;

use crate::error::{Error, Result};
use crate::Library;

/// A loaded `TFLite` model.
///
/// Models can be created from in-memory bytes or from a file. The model
/// data is kept alive for the lifetime of the `Model`.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
pub struct Model<'lib> {
    ptr: NonNull<TfLiteModel>,
    model_mem: Vec<u8>,
    lib: &'lib Library,
}

impl<'lib> Model<'lib> {
    /// Create a `Model` from raw bytes.
    ///
    /// Takes ownership of the provided byte buffer and passes a pointer to
    /// the underlying `TFLite` C API. The data is kept alive for the
    /// lifetime of the returned `Model`.
    pub fn from_bytes(lib: &'lib Library, data: impl Into<Vec<u8>>) -> Result<Self> {
        let model_mem: Vec<u8> = data.into();
        // SAFETY: We pass a valid pointer and length from the owned Vec.
        // The Vec is stored in `model_mem` and lives as long as the Model,
        // satisfying TFLite's requirement that the buffer outlives the model.
        let raw = unsafe {
            lib.as_sys()
                .TfLiteModelCreate(model_mem.as_ptr().cast::<c_void>(), model_mem.len())
        };
        let ptr = NonNull::new(raw)
            .ok_or_else(|| Error::null_pointer("TfLiteModelCreate returned null"))?;
        Ok(Self {
            ptr,
            model_mem,
            lib,
        })
    }

    /// Create a `Model` by reading a file from disk.
    ///
    /// Reads the entire file into memory, then delegates to
    /// [`Model::from_bytes`].
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read (I/O error) or if the
    /// `TFLite` C API fails to parse the model bytes (returns null).
    pub fn from_file(lib: &'lib Library, path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref())
            .map_err(|e| Error::invalid_argument(format!("{}: {e}", path.as_ref().display())))?;
        Self::from_bytes(lib, data)
    }

    /// Returns the raw model data bytes.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.model_mem
    }

    /// Returns the raw `TfLiteModel` pointer for use by the interpreter.
    pub(crate) fn as_ptr(&self) -> *mut TfLiteModel {
        self.ptr.as_ptr()
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was created by `TfLiteModelCreate` and has not
        // been deleted yet. The matching `TfLiteModelDelete` releases the
        // model resources allocated by the C library.
        unsafe {
            self.lib.as_sys().TfLiteModelDelete(self.ptr.as_ptr());
        }
    }
}
