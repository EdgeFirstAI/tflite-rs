// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! `LiteRT` model loading and signature introspection.

use std::ffi::CString;
use std::os::raw::c_void;
use std::path::Path;

use edgefirst_tflite_sys::litert::{
    LiteRtFunctions, Model as RawModel, ParamIndex, RankedTensorType, Signature, Tensor,
};

use crate::error::{litert_status_to_result, Error, Result};
use crate::litert::Environment;

/// A `LiteRT` model — a parsed `.tflite` flatbuffer, not yet compiled.
///
/// A model describes *what* to run; [`CompiledModel`](crate::litert::CompiledModel)
/// decides *where* it runs. Loading is cheap relative to compilation, so it is
/// reasonable to load a model, inspect its signatures, and only then choose
/// accelerators.
///
/// # Signatures
///
/// `LiteRT` models expose one or more *signatures*, each a named entry point
/// with its own inputs and outputs. Models exported by most toolchains have
/// exactly one, at index `0` — that is what
/// [`CompiledModel::run_default`](crate::litert::CompiledModel::run_default) and
/// the `signature_index: 0` arguments throughout this module refer to. Call
/// [`Model::num_signatures`] if you need to handle multi-signature models.
///
/// # Examples
///
/// ```no_run
/// use edgefirst_tflite::{litert, Library};
///
/// let lib = Library::new()?;
/// let env = litert::Environment::new(&lib)?;
/// let model = litert::Model::from_file(&env, "model.tflite")?;
///
/// println!("signatures: {}", model.num_signatures()?);
/// println!("inputs:     {}", model.num_inputs(0)?);
/// println!("outputs:    {}", model.num_outputs(0)?);
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
///
/// # Lifetime
///
/// `Model<'env>` borrows the [`Environment`] it was loaded into. It must also
/// outlive any [`CompiledModel`](crate::litert::CompiledModel) built from it —
/// the compiled model reads this model's flatbuffer during every inference.
/// That requirement is enforced by the borrow checker, not left to the caller.
///
/// # Thread safety
///
/// `Model` is [`Send`] and [`Sync`]. All methods are const reads of the parsed
/// flatbuffer: they resolve a signature and read tensor metadata, and mutate
/// nothing.
#[derive(Debug)]
pub struct Model<'env> {
    ptr: RawModel,
    /// Retained for buffer-backed models: `LiteRtCreateModelFromBuffer`
    /// documents that "the caller must ensure that the buffer remains valid for
    /// the lifetime of the model", so the model owns the bytes it was parsed
    /// from. Never read directly — its presence *is* the invariant.
    _buffer: Option<Vec<u8>>,
    fns: &'env LiteRtFunctions,
}

impl<'env> Model<'env> {
    /// Load a model from a filesystem path.
    ///
    /// The file is read by the runtime; nothing is retained on the Rust side.
    ///
    /// # Errors
    ///
    /// - [`Error::is_invalid_argument`](crate::Error::is_invalid_argument) if
    ///   `path` is not valid UTF-8 or contains an interior NUL byte.
    /// - A `LiteRT` status error if the file cannot be read
    ///   (`ERROR_FILE_IO`) or is not a valid flatbuffer
    ///   (`ERROR_INVALID_FLATBUFFER`).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use edgefirst_tflite::{litert, Library};
    /// # let lib = Library::new()?;
    /// # let env = litert::Environment::new(&lib)?;
    /// let model = litert::Model::from_file(&env, "yolov8n.tflite")?;
    /// # Ok::<(), edgefirst_tflite::Error>(())
    /// ```
    pub fn from_file(env: &'env Environment<'_>, path: impl AsRef<Path>) -> Result<Self> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            Error::invalid_argument(format!(
                "model path is not valid UTF-8: {}",
                path.as_ref().display()
            ))
        })?;
        let c_path = CString::new(path_str).map_err(|e| {
            Error::invalid_argument(format!("model path contains interior nul: {e}"))
        })?;
        let fns = env.functions();
        let mut model: RawModel = std::ptr::null_mut();
        // SAFETY: `c_path` is a valid NUL-terminated C string that outlives the
        // call, and `env` is a live `LiteRtEnvironment`.
        let status =
            unsafe { (fns.create_model_from_file)(env.as_raw(), c_path.as_ptr(), &raw mut model) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtCreateModelFromFile"))?;
        if model.is_null() {
            return Err(Error::null_pointer(
                "LiteRtCreateModelFromFile returned null",
            ));
        }
        Ok(Self {
            ptr: model,
            _buffer: None,
            fns,
        })
    }

    /// Load a model from an in-memory flatbuffer.
    ///
    /// The bytes are moved into the `Model` and kept alive for as long as it
    /// exists, because the runtime parses the buffer in place rather than
    /// copying it.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if the buffer is not a valid `.tflite`
    /// flatbuffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use edgefirst_tflite::{litert, Library};
    /// # let lib = Library::new()?;
    /// # let env = litert::Environment::new(&lib)?;
    /// let bytes = std::fs::read("model.tflite")?;
    /// let model = litert::Model::from_buffer(&env, bytes)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_buffer(env: &'env Environment<'_>, data: impl Into<Vec<u8>>) -> Result<Self> {
        let buffer = data.into();
        let fns = env.functions();
        let mut model: RawModel = std::ptr::null_mut();
        // SAFETY: the pointer/length pair describes `buffer`, which is moved
        // into `Self` below. Moving a `Vec` does not move its heap allocation,
        // so the address handed to the runtime stays valid for the model's
        // lifetime, as `LiteRtCreateModelFromBuffer` requires.
        let status = unsafe {
            (fns.create_model_from_buffer)(
                env.as_raw(),
                buffer.as_ptr().cast::<c_void>(),
                buffer.len(),
                &raw mut model,
            )
        };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtCreateModelFromBuffer"))?;
        if model.is_null() {
            return Err(Error::null_pointer(
                "LiteRtCreateModelFromBuffer returned null",
            ));
        }
        Ok(Self {
            ptr: model,
            _buffer: Some(buffer),
            fns,
        })
    }

    /// Number of signatures (named entry points) in the model.
    ///
    /// Almost always `1`. Signature indices passed elsewhere in this module
    /// must be less than this value.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if the model cannot be queried.
    pub fn num_signatures(&self) -> Result<usize> {
        let mut n: ParamIndex = 0;
        // SAFETY: `self.ptr` is a live model; `n` is a valid out-parameter.
        let status = unsafe { (self.fns.get_num_model_signatures)(self.ptr, &raw mut n) };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetNumModelSignatures"))?;
        Ok(n)
    }

    /// Number of input tensors for `signature_index` (default signature is `0`).
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error, including when `signature_index` is out of
    /// range.
    pub fn num_inputs(&self, signature_index: usize) -> Result<usize> {
        let sig = self.signature(signature_index)?;
        let mut n: ParamIndex = 0;
        // SAFETY: `sig` is borrowed from a live model and remains valid.
        let status = unsafe { (self.fns.get_num_signature_inputs)(sig, &raw mut n) };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetNumSignatureInputs"))?;
        Ok(n)
    }

    /// Number of output tensors for `signature_index`.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error, including when `signature_index` is out of
    /// range.
    pub fn num_outputs(&self, signature_index: usize) -> Result<usize> {
        let sig = self.signature(signature_index)?;
        let mut n: ParamIndex = 0;
        // SAFETY: `sig` is borrowed from a live model and remains valid.
        let status = unsafe { (self.fns.get_num_signature_outputs)(sig, &raw mut n) };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetNumSignatureOutputs"))?;
        Ok(n)
    }

    /// Element type and shape of an input tensor.
    ///
    /// Needed when allocating a [`TensorBuffer`](crate::litert::TensorBuffer)
    /// by hand; [`CompiledModel::create_input_buffer`](crate::litert::CompiledModel::create_input_buffer)
    /// looks it up for you.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if either index is out of range, or if the
    /// tensor is unranked (dynamic shape).
    pub fn input_tensor_type(
        &self,
        signature_index: usize,
        input_index: usize,
    ) -> Result<RankedTensorType> {
        let sig = self.signature(signature_index)?;
        let mut tensor: Tensor = std::ptr::null_mut();
        // SAFETY: `sig` is live; `tensor` is a valid out-parameter.
        let status = unsafe {
            (self.fns.get_signature_input_tensor_by_index)(sig, input_index, &raw mut tensor)
        };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetSignatureInputTensorByIndex"))?;
        self.ranked_type(tensor)
    }

    /// Element type and shape of an output tensor.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if either index is out of range, or if the
    /// tensor is unranked (dynamic shape).
    pub fn output_tensor_type(
        &self,
        signature_index: usize,
        output_index: usize,
    ) -> Result<RankedTensorType> {
        let sig = self.signature(signature_index)?;
        let mut tensor: Tensor = std::ptr::null_mut();
        // SAFETY: `sig` is live; `tensor` is a valid out-parameter.
        let status = unsafe {
            (self.fns.get_signature_output_tensor_by_index)(sig, output_index, &raw mut tensor)
        };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetSignatureOutputTensorByIndex"))?;
        self.ranked_type(tensor)
    }

    pub(crate) fn as_raw(&self) -> RawModel {
        self.ptr
    }

    /// Borrow a signature handle. The returned pointer is owned by the model
    /// and is only used within the calling method, never stored.
    fn signature(&self, index: usize) -> Result<Signature> {
        let mut sig: Signature = std::ptr::null_mut();
        // SAFETY: `self.ptr` is a live model; `sig` is a valid out-parameter.
        let status = unsafe { (self.fns.get_model_signature)(self.ptr, index, &raw mut sig) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtGetModelSignature"))?;
        if sig.is_null() {
            return Err(Error::null_pointer("LiteRtGetModelSignature returned null"));
        }
        Ok(sig)
    }

    fn ranked_type(&self, tensor: Tensor) -> Result<RankedTensorType> {
        // SAFETY: `LiteRtRankedTensorType` is a plain-data C struct (element
        // type plus an inline dimension array), so an all-zero value is a valid
        // bit pattern to hand the runtime as an out-parameter.
        let mut ty = unsafe { std::mem::zeroed::<RankedTensorType>() };
        // SAFETY: `tensor` was just obtained from a live signature of a live
        // model; `ty` is a valid out-parameter.
        let status = unsafe { (self.fns.get_ranked_tensor_type)(tensor, &raw mut ty) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtGetRankedTensorType"))?;
        Ok(ty)
    }
}

// SAFETY: `Model` owns a `LiteRtModel` (an opaque heap handle with no thread
// affinity) plus an optional `Vec<u8>`, which is `Send`. Moving the pair
// between threads is sound.
unsafe impl Send for Model<'_> {}

// SAFETY: every method reachable through `&Model` reads parsed flatbuffer
// metadata — signature lookup and tensor type queries. None mutates the model,
// so concurrent shared access is sound.
unsafe impl Sync for Model<'_> {}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        // SAFETY: `ptr` was created by a `LiteRtCreateModel*` call, is non-null
        // (checked at construction), and is destroyed once. Any `CompiledModel`
        // built from this model borrows it, so none can still be alive here.
        unsafe {
            (self.fns.destroy_model)(self.ptr);
        }
    }
}
