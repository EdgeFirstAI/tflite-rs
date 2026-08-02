// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! `LiteRT` compiled model — the synchronous inference path.

use edgefirst_tflite_sys::litert::{
    CompiledModel as RawCompiled, LiteRtFunctions, TensorBuffer as RawBuffer,
    TensorBufferRequirements as RawReqs,
};

use crate::error::{litert_status_to_result, Error, Result};
use crate::litert::tensor_buffer::{BufferRequirements, TensorBuffer};
use crate::litert::{Environment, Model, Options};

/// A model compiled for a specific set of hardware accelerators, ready to run.
///
/// Compilation is where `LiteRT` decides which operators go to which
/// accelerator and prepares the buffers each one needs. It is the expensive
/// step — do it once and run many times.
///
/// # Typical flow
///
/// ```no_run
/// use edgefirst_tflite::{litert, Library};
///
/// let lib = Library::new()?;
/// let env = litert::Environment::new(&lib)?;
/// let model = litert::Model::from_file(&env, "model.tflite")?;
/// let opts = litert::Options::new(&lib)?
///     .hardware_accelerators(litert::HwAccelerators::NPU | litert::HwAccelerators::CPU)?;
///
/// let mut compiled = litert::CompiledModel::create(&env, &model, &opts)?;
/// if !compiled.is_fully_accelerated()? {
///     eprintln!("warning: part of this model fell back to the CPU");
/// }
///
/// let mut inputs = vec![compiled.create_input_buffer(0, 0)?];
/// let mut outputs = vec![compiled.create_output_buffer(0, 0)?];
///
/// let input_size = inputs[0].size();
/// inputs[0].write_bytes(&vec![0u8; input_size])?;
/// compiled.run_default(&mut inputs, &mut outputs)?;
/// let result = outputs[0].read_bytes()?;
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
///
/// # Lifetime
///
/// `CompiledModel<'env>` borrows **both** the [`Environment`] and the
/// [`Model`]. The model borrow is not incidental: `LiteRtCreateCompiledModel`
/// does not take ownership, and the compiled model reads the model's flatbuffer
/// on every inference. Dropping the [`Model`] first is a use-after-free that
/// segfaults in practice, so the borrow checker rejects it:
///
/// ```compile_fail
/// # use edgefirst_tflite::{litert, Library};
/// # let lib = Library::new().unwrap();
/// # let env = litert::Environment::new(&lib).unwrap();
/// # let opts = litert::Options::new(&lib).unwrap();
/// let mut compiled = {
///     let model = litert::Model::from_file(&env, "m.tflite").unwrap();
///     litert::CompiledModel::create(&env, &model, &opts).unwrap()
/// }; // `model` dropped here — the compiled model would dangle
/// compiled.run_default(&mut [], &mut []).unwrap();
/// ```
///
/// [`Options`], by contrast, are *not* borrowed: the C API documents them as
/// caller-owned and not retained, so they may be dropped or reused for another
/// compilation immediately.
///
/// # Thread safety
///
/// `CompiledModel` is [`Send`] but not [`Sync`]. Inference mutates internal
/// runtime state, and [`CompiledModel::run`] takes `&mut self`, so a compiled
/// model can be moved to a worker thread but not shared between threads. For
/// concurrent inference, compile one model per thread.
#[derive(Debug)]
pub struct CompiledModel<'env> {
    ptr: RawCompiled,
    fns: &'env LiteRtFunctions,
    env: &'env Environment<'env>,
    /// The model this was compiled from. Retained because the runtime reads it
    /// during every inference — see the type-level docs.
    model: &'env Model<'env>,
    /// Reused across [`CompiledModel::run`] calls so that inference does not
    /// allocate. Cleared and refilled per call.
    input_handles: Vec<RawBuffer>,
    output_handles: Vec<RawBuffer>,
}

impl<'env> CompiledModel<'env> {
    /// Compile `model` for the accelerators selected in `options`.
    ///
    /// This is the expensive call: the runtime partitions the graph, loads any
    /// accelerator plugins, and prepares delegate kernels.
    ///
    /// `options` must select at least one accelerator. A freshly created
    /// [`Options`] selects none, and compiling with an empty set is rejected
    /// with `ERROR_INVALID_ARGUMENT`.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error — `ERROR_INVALID_ARGUMENT` when no accelerator
    /// is selected, `ERROR_COMPILATION` when a requested accelerator cannot
    /// compile the graph, or `ERROR_RUNTIME_FAILURE` when a plugin fails to
    /// load.
    pub fn create(
        env: &'env Environment<'env>,
        model: &'env Model<'env>,
        options: &Options<'_>,
    ) -> Result<Self> {
        let fns = env.functions();
        let mut compiled: RawCompiled = std::ptr::null_mut();
        // SAFETY: `env`, `model`, and `options` are live owned handles for the
        // duration of the call; `compiled` is a valid out-parameter. The model
        // borrow is retained in `Self`, so it outlives the compiled model as
        // the C API requires.
        let status = unsafe {
            (fns.create_compiled_model)(
                env.as_raw(),
                model.as_raw(),
                options.as_raw(),
                &raw mut compiled,
            )
        };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtCreateCompiledModel"))?;
        if compiled.is_null() {
            return Err(Error::null_pointer(
                "LiteRtCreateCompiledModel returned null",
            ));
        }
        Ok(Self {
            ptr: compiled,
            fns,
            env,
            model,
            input_handles: Vec::new(),
            output_handles: Vec::new(),
        })
    }

    /// The model this was compiled from.
    #[must_use]
    pub const fn model(&self) -> &'env Model<'env> {
        self.model
    }

    /// Whether every operator was dispatched to a selected hardware
    /// accelerator.
    ///
    /// `false` means at least one operator fell back — typically to the CPU —
    /// which is the usual explanation for an NPU-targeted model running far
    /// slower than expected. Note the predicate is about the *selected* set:
    /// with `GPU | NPU` selected, a model delegated entirely to the GPU still
    /// reports `true`.
    ///
    /// A model compiled with only [`HwAccelerators::CPU`](crate::litert::HwAccelerators::CPU)
    /// generally reports `true` once XNNPACK claims the graph, so this is most
    /// informative when a GPU or NPU was requested.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if the compiled model cannot be queried.
    pub fn is_fully_accelerated(&self) -> Result<bool> {
        let mut fully = false;
        // SAFETY: `self.ptr` is a live compiled model; `fully` is a valid
        // out-parameter for a C `bool`.
        let status =
            unsafe { (self.fns.compiled_model_is_fully_accelerated)(self.ptr, &raw mut fully) };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtCompiledModelIsFullyAccelerated"))?;
        Ok(fully)
    }

    /// How the runtime needs input tensor `input_index` to be backed.
    ///
    /// The returned value borrows `self` — see [`BufferRequirements`] for why.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if either index is out of range.
    pub fn input_buffer_requirements(
        &self,
        signature_index: usize,
        input_index: usize,
    ) -> Result<BufferRequirements<'_>> {
        let mut reqs: RawReqs = std::ptr::null_mut();
        // SAFETY: `self.ptr` is live; `reqs` is a valid out-parameter. The
        // handle written back is owned by this compiled model, which is why the
        // wrapper borrows `self`.
        let status = unsafe {
            (self.fns.get_compiled_model_input_buffer_requirements)(
                self.ptr,
                signature_index,
                input_index,
                &raw mut reqs,
            )
        };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetCompiledModelInputBufferRequirements"))?;
        if reqs.is_null() {
            return Err(Error::null_pointer(
                "LiteRtGetCompiledModelInputBufferRequirements returned null",
            ));
        }
        BufferRequirements::from_raw(self.fns, reqs)
    }

    /// How the runtime needs output tensor `output_index` to be backed.
    ///
    /// The returned value borrows `self` — see [`BufferRequirements`] for why.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if either index is out of range.
    pub fn output_buffer_requirements(
        &self,
        signature_index: usize,
        output_index: usize,
    ) -> Result<BufferRequirements<'_>> {
        let mut reqs: RawReqs = std::ptr::null_mut();
        // SAFETY: as in `input_buffer_requirements`.
        let status = unsafe {
            (self.fns.get_compiled_model_output_buffer_requirements)(
                self.ptr,
                signature_index,
                output_index,
                &raw mut reqs,
            )
        };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetCompiledModelOutputBufferRequirements"))?;
        if reqs.is_null() {
            return Err(Error::null_pointer(
                "LiteRtGetCompiledModelOutputBufferRequirements returned null",
            ));
        }
        BufferRequirements::from_raw(self.fns, reqs)
    }

    /// Allocate an input buffer that satisfies this model's requirements.
    ///
    /// Looks up the tensor type from the compiled model's own [`Model`] and the
    /// backing-store requirements from the compiled model, so the result is
    /// always consistent with what inference expects. This is the recommended
    /// way to obtain buffers.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if either index is out of range or allocation
    /// fails.
    pub fn create_input_buffer(
        &self,
        signature_index: usize,
        input_index: usize,
    ) -> Result<TensorBuffer<'env>> {
        let ty = self.model.input_tensor_type(signature_index, input_index)?;
        let reqs = self.input_buffer_requirements(signature_index, input_index)?;
        TensorBuffer::managed_from_requirements(self.env, &ty, reqs)
    }

    /// Allocate an output buffer that satisfies this model's requirements.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if either index is out of range or allocation
    /// fails.
    pub fn create_output_buffer(
        &self,
        signature_index: usize,
        output_index: usize,
    ) -> Result<TensorBuffer<'env>> {
        let ty = self
            .model
            .output_tensor_type(signature_index, output_index)?;
        let reqs = self.output_buffer_requirements(signature_index, output_index)?;
        TensorBuffer::managed_from_requirements(self.env, &ty, reqs)
    }

    /// Run inference synchronously for `signature_index`.
    ///
    /// `inputs` and `outputs` must match the signature's tensor counts and
    /// order. Buffers are taken by `&mut` because the runtime may write to
    /// either — outputs obviously, and inputs when a backend rewrites them in
    /// place.
    ///
    /// The handle arrays handed to the C API are reused across calls, so a warm
    /// inference loop performs no allocation.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error — commonly `ERROR_INVALID_ARGUMENT` for a buffer
    /// count or type mismatch, or `ERROR_RUNTIME_FAILURE` from the accelerator.
    pub fn run(
        &mut self,
        signature_index: usize,
        inputs: &mut [TensorBuffer<'env>],
        outputs: &mut [TensorBuffer<'env>],
    ) -> Result<()> {
        self.input_handles.clear();
        self.input_handles
            .extend(inputs.iter().map(TensorBuffer::as_raw));
        self.output_handles.clear();
        self.output_handles
            .extend(outputs.iter().map(TensorBuffer::as_raw));

        let run = self.fns.run_compiled_model;
        let ptr = self.ptr;
        let num_inputs = self.input_handles.len();
        let num_outputs = self.output_handles.len();
        let input_ptr = self.input_handles.as_mut_ptr();
        let output_ptr = self.output_handles.as_mut_ptr();

        // SAFETY: `ptr` is a live compiled model. The two arrays are valid for
        // `num_inputs`/`num_outputs` elements and are exclusively borrowed for
        // the call. Each handle they contain is owned by a `TensorBuffer` that
        // is mutably borrowed for the duration, so none can be freed or
        // concurrently mapped while the runtime uses it.
        let status = unsafe {
            run(
                ptr,
                signature_index,
                num_inputs,
                input_ptr,
                num_outputs,
                output_ptr,
            )
        };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtRunCompiledModel"))
    }

    /// Run inference on the default signature (index `0`).
    ///
    /// Equivalent to `run(0, inputs, outputs)`. Most models have exactly one
    /// signature; see [`Model::num_signatures`] if yours might not.
    ///
    /// # Errors
    ///
    /// As [`CompiledModel::run`].
    pub fn run_default(
        &mut self,
        inputs: &mut [TensorBuffer<'env>],
        outputs: &mut [TensorBuffer<'env>],
    ) -> Result<()> {
        self.run(0, inputs, outputs)
    }
}

// SAFETY: `CompiledModel` owns a `LiteRtCompiledModel`, an opaque heap handle
// with no thread affinity. It also holds `&Environment` and `&Model`, both of
// which are `Sync`, so sharing them across the move is sound.
unsafe impl Send for CompiledModel<'_> {}

// Deliberately NOT `Sync`: inference mutates runtime state internal to the
// compiled model. `run` takes `&mut self` to enforce that.

impl Drop for CompiledModel<'_> {
    fn drop(&mut self) {
        // SAFETY: `ptr` was created by `LiteRtCreateCompiledModel`, is non-null
        // (checked in `create`), and is destroyed exactly once. Any
        // `BufferRequirements` derived from it borrows `self`, so none can
        // still be alive here.
        unsafe {
            (self.fns.destroy_compiled_model)(self.ptr);
        }
    }
}
