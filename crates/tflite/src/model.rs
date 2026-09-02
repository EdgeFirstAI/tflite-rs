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
    /// Buffer handed to `TfLiteModelCreate`; the model's constant tensors
    /// point into it, so it must outlive the model. This is the offset-buffer
    /// inlined rewrite when the source used offset buffers, otherwise the
    /// source bytes themselves.
    runtime_mem: Vec<u8>,
    /// The original source bytes, returned by [`Model::data`]. Preserved
    /// separately only when inlining produced a different `runtime_mem`, so a
    /// metadata trailer the exporter appended after the flatbuffer (which the
    /// runtime buffer drops) stays readable. `None` means `data()` returns
    /// `runtime_mem` directly (no inlining happened — the common case).
    source_mem: Option<Vec<u8>>,
    lib: &'lib Library,
}

impl<'lib> Model<'lib> {
    /// Create a `Model` from raw bytes.
    ///
    /// Takes ownership of the provided byte buffer and passes a pointer to
    /// the underlying `TFLite` C API. The data is kept alive for the
    /// lifetime of the returned `Model`.
    pub fn from_bytes(lib: &'lib Library, data: impl Into<Vec<u8>>) -> Result<Self> {
        let source_mem: Vec<u8> = data.into();
        // ai-edge / LiteRT exporters (e.g. Ultralytics int8 TFLite) may store
        // large weight/bias buffers *outside* the flatbuffer, referenced by
        // `Buffer.offset`. The TFLite C API does not resolve those, so the
        // interpreter would abort with "Input tensor N lacks data". Rewrite
        // the model in memory so every buffer is inline and hand *that* to the
        // runtime; a model that already stores all buffers inline is used
        // as-is (no rewrite, no extra copy). The original bytes are kept for
        // `data()` so an appended metadata trailer survives — see `source_mem`.
        let (runtime_mem, source_mem) = match crate::inline::inline_offset_buffers(&source_mem) {
            Some(inlined) => (inlined, Some(source_mem)),
            None => (source_mem, None),
        };
        // SAFETY: We pass a valid pointer and length from the owned Vec.
        // The Vec is stored in `runtime_mem` and lives as long as the Model,
        // satisfying TFLite's requirement that the buffer outlives the model.
        let raw = unsafe {
            lib.as_sys()
                .TfLiteModelCreate(runtime_mem.as_ptr().cast::<c_void>(), runtime_mem.len())
        };
        let ptr = NonNull::new(raw)
            .ok_or_else(|| Error::null_pointer("TfLiteModelCreate returned null"))?;
        Ok(Self {
            ptr,
            runtime_mem,
            source_mem,
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

    /// Returns the original model bytes as provided to the loader.
    ///
    /// This is the source model, not the offset-buffer-inlined rewrite handed
    /// to the runtime, so any trailing metadata the exporter appended after
    /// the flatbuffer is preserved for metadata readers.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.source_mem.as_deref().unwrap_or(&self.runtime_mem)
    }

    /// Returns the raw `TfLiteModel` pointer for use by the interpreter.
    pub(crate) fn as_ptr(&self) -> *mut TfLiteModel {
        self.ptr.as_ptr()
    }
}

// SAFETY: `Model` owns a `NonNull<TfLiteModel>` (an opaque C handle with
// no thread affinity), a `Vec<u8>` backing buffer (`Send + Sync`), and a
// `&Library` reference (`Sync`). The `TfLiteModel*` is immutable after
// creation — multiple interpreters can be created from it concurrently.
unsafe impl Send for Model<'_> {}

// SAFETY: The `TfLiteModel*` is internally reference-counted and read-only
// after construction. Sharing `&Model` across threads is safe because all
// access is read-only (creating interpreters takes `&Model`, not `&mut`).
unsafe impl Sync for Model<'_> {}

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
