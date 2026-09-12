// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Interpreter and builder for `TFLite` model inference.
//!
//! The [`Interpreter`] is created through a builder pattern:
//!
//! ```no_run
//! use edgefirst_tflite::{Library, Model, Interpreter};
//!
//! let lib = Library::new()?;
//! let model = Model::from_file(&lib, "model.tflite")?;
//!
//! let mut interpreter = Interpreter::builder(&lib)?
//!     .num_threads(4)
//!     .build(&model)?;
//!
//! interpreter.invoke()?;
//! # Ok::<(), edgefirst_tflite::Error>(())
//! ```

use std::ptr::NonNull;

use edgefirst_tflite_sys::experimental_ffi::{
    kTfLiteCustomAllocationFlagsNone, TfLiteCustomAllocation, TENSOR_ALIGNMENT,
};
use edgefirst_tflite_sys::{TfLiteInterpreter, TfLiteInterpreterOptions};

use crate::delegate::Delegate;
use crate::error::{self, Error, Result};
use crate::model::Model;
use crate::profiler::Profiler;
use crate::tensor::{Tensor, TensorMut};
use crate::Library;

// ---------------------------------------------------------------------------
// InterpreterBuilder
// ---------------------------------------------------------------------------

/// Builder for configuring and creating a `TFLite` [`Interpreter`].
///
/// Created via [`Interpreter::builder`].
pub struct InterpreterBuilder<'lib> {
    options: NonNull<TfLiteInterpreterOptions>,
    delegates: Vec<Delegate>,
    lib: &'lib Library,
}

impl<'lib> InterpreterBuilder<'lib> {
    /// Set the number of threads for inference.
    ///
    /// A value of -1 lets `TFLite` choose based on the platform.
    #[must_use]
    pub fn num_threads(self, n: i32) -> Self {
        // SAFETY: `self.options` is a valid non-null options pointer created by
        // `TfLiteInterpreterOptionsCreate`.
        unsafe {
            self.lib
                .as_sys()
                .TfLiteInterpreterOptionsSetNumThreads(self.options.as_ptr(), n);
        }
        self
    }

    /// Add a delegate for hardware acceleration.
    ///
    /// The delegate is moved into the builder and will be owned by the
    /// resulting [`Interpreter`].
    #[must_use]
    pub fn delegate(mut self, d: Delegate) -> Self {
        // SAFETY: `self.options` and the delegate pointer are both valid. The
        // delegate is stored in `self.delegates` to keep it alive.
        unsafe {
            self.lib
                .as_sys()
                .TfLiteInterpreterOptionsAddDelegate(self.options.as_ptr(), d.as_ptr());
        }
        self.delegates.push(d);
        self
    }

    /// Attach a telemetry [`Profiler`] that collects per-op timing events.
    ///
    /// The profiler must outlive the resulting [`Interpreter`]. This is
    /// naturally guaranteed when the `Profiler` is declared before the
    /// interpreter in the same scope.
    ///
    /// The telemetry profiler API is optional — if the loaded `TFLite`
    /// library does not export
    /// `TfLiteInterpreterOptionsSetTelemetryProfiler`, this method returns
    /// an error rather than silently ignoring the profiler.
    ///
    /// # Errors
    ///
    /// Returns an error if the telemetry profiler symbol cannot be resolved
    /// from the loaded `TFLite` library.
    pub fn profiler(self, profiler: &Profiler) -> Result<Self> {
        // Dynamically look up the optional telemetry setter.
        let tflite_lib = self.lib.reopen()?;

        // SAFETY: `tflite_lib` is the same library that was successfully
        // loaded during `Library` construction. The symbol may or may not
        // exist depending on the TFLite build.
        let set_profiler: libloading::Symbol<
            '_,
            unsafe extern "C" fn(*mut TfLiteInterpreterOptions, *mut std::ffi::c_void),
        > = unsafe { tflite_lib.get(b"TfLiteInterpreterOptionsSetTelemetryProfiler\0") }.map_err(
            |_| {
                Error::invalid_argument(
                    "TfLiteInterpreterOptionsSetTelemetryProfiler symbol not found — \
                 the TFLite library may not support the telemetry profiler API",
                )
            },
        )?;

        // SAFETY: `self.options` is a valid options pointer. `profiler.as_ptr()`
        // returns a pointer to a boxed C struct that remains valid for the
        // lifetime of the `Profiler`.
        unsafe {
            set_profiler(self.options.as_ptr(), profiler.as_ptr());
        }

        Ok(self)
    }

    /// Build the interpreter for the given model.
    ///
    /// This creates the interpreter and allocates tensors. After this call,
    /// input tensors can be populated and inference can be run.
    ///
    /// # Errors
    ///
    /// Returns an error if interpreter creation fails or tensor allocation
    /// returns a non-OK status.
    pub fn build(mut self, model: &Model<'lib>) -> Result<Interpreter<'lib>> {
        // SAFETY: `model.as_ptr()` and `self.options` are both valid non-null
        // pointers. The library is loaded and the function pointer is valid.
        let raw = unsafe {
            self.lib
                .as_sys()
                .TfLiteInterpreterCreate(model.as_ptr(), self.options.as_ptr())
        };

        let interp_ptr = NonNull::new(raw)
            .ok_or_else(|| Error::null_pointer("TfLiteInterpreterCreate returned null"))?;

        let interpreter = Interpreter {
            ptr: interp_ptr,
            delegates: std::mem::take(&mut self.delegates),
            lib: self.lib,
        };

        // SAFETY: `interpreter.ptr` is a valid interpreter pointer just created above.
        let status = unsafe {
            self.lib
                .as_sys()
                .TfLiteInterpreterAllocateTensors(interpreter.ptr.as_ptr())
        };
        error::status_to_result(status)
            .map_err(|e| e.with_context("TfLiteInterpreterAllocateTensors"))?;

        Ok(interpreter)
    }
}

impl Drop for InterpreterBuilder<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.options` was created by `TfLiteInterpreterOptionsCreate`
        // and has not been deleted yet.
        unsafe {
            self.lib
                .as_sys()
                .TfLiteInterpreterOptionsDelete(self.options.as_ptr());
        }
    }
}

impl std::fmt::Debug for InterpreterBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterpreterBuilder")
            .field("delegates", &self.delegates.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

/// `TFLite` inference engine.
///
/// Owns its delegates and provides access to input/output tensors.
/// Created via [`Interpreter::builder`].
pub struct Interpreter<'lib> {
    ptr: NonNull<TfLiteInterpreter>,
    delegates: Vec<Delegate>,
    lib: &'lib Library,
}

impl<'lib> Interpreter<'lib> {
    /// Create a new [`InterpreterBuilder`] for configuring an interpreter.
    ///
    /// # Errors
    ///
    /// Returns an error if `TfLiteInterpreterOptionsCreate` returns null.
    pub fn builder(lib: &'lib Library) -> Result<InterpreterBuilder<'lib>> {
        // SAFETY: The library is loaded and the function pointer is valid.
        let options = NonNull::new(unsafe { lib.as_sys().TfLiteInterpreterOptionsCreate() })
            .ok_or_else(|| Error::null_pointer("TfLiteInterpreterOptionsCreate returned null"))?;

        Ok(InterpreterBuilder {
            options,
            delegates: Vec::new(),
            lib,
        })
    }

    /// Re-allocate tensors after an input resize.
    ///
    /// This must be called after [`Interpreter::resize_input`] and before
    /// [`Interpreter::invoke`]. Any previously obtained tensor slices or
    /// pointers are invalidated.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    pub fn allocate_tensors(&mut self) -> Result<()> {
        // SAFETY: `self.ptr` is a valid interpreter pointer.
        let status = unsafe {
            self.lib
                .as_sys()
                .TfLiteInterpreterAllocateTensors(self.ptr.as_ptr())
        };
        error::status_to_result(status)
            .map_err(|e| e.with_context("TfLiteInterpreterAllocateTensors"))
    }

    /// Back an input tensor with caller-owned memory instead of the arena.
    ///
    /// This is what turns a GPU-resident buffer into the tensor the runtime
    /// reads, removing the host copy that otherwise stages every frame into
    /// the arena. The canonical use is a HAL image tensor allocated with
    /// `TensorMemory::DmaBuf` — a DMA-BUF on Linux, an `IOSurface` on Apple
    /// platforms — that the GPU renders into directly.
    ///
    /// Call [`Interpreter::allocate_tensors`] afterwards; the binding does not
    /// take effect until you do. Several inputs can be bound before a single
    /// re-allocation.
    ///
    /// # Safety
    ///
    /// The caller guarantees, for as long as this interpreter can read the
    /// tensor — that is, until the interpreter is dropped or the binding is
    /// replaced:
    ///
    /// - `data` stays valid, mapped, and **at the same address**. A mapping
    ///   that can be revoked or relocated (anything staged or refcounted
    ///   behind a temporary guard) is not eligible; the runtime keeps the raw
    ///   pointer and never re-queries it.
    /// - Nothing else writes the region while [`Interpreter::invoke`] runs.
    /// - The region is readable for at least `bytes`, and writable too if the
    ///   tensor is anything other than a pure input.
    ///
    /// The runtime does not take ownership and will not free the memory.
    ///
    /// # Errors
    ///
    /// - The runtime does not export the experimental C API
    ///   ([`Library::has_custom_allocation`](crate::Library::has_custom_allocation)
    ///   is `false`).
    /// - `input_index` is out of range, or the runtime rejects it.
    /// - `bytes` is smaller than the tensor's byte size.
    /// - `data` is not 64-byte aligned (`kDefaultTensorAlignment`). Page-backed
    ///   mappings — DMA-BUF, `IOSurface` — always satisfy this; a pointer into
    ///   the middle of a `Vec` generally does not.
    /// - The C API returns a non-OK status, e.g. because the tensor is not
    ///   arena-allocated (a delegate may own it instead).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use edgefirst_tflite::{Interpreter, Library, Model};
    /// # let lib = Library::new()?;
    /// # let model = Model::from_file(&lib, "model.tflite")?;
    /// # let mut interpreter = Interpreter::builder(&lib)?.build(&model)?;
    /// # let (ptr, len) = (std::ptr::NonNull::dangling(), 0usize);
    /// if lib.has_custom_allocation() {
    ///     // SAFETY: `ptr` addresses a page-aligned mapping of at least `len`
    ///     // bytes that outlives `interpreter` and is not relocated.
    ///     unsafe { interpreter.set_custom_allocation_for_input(0, ptr, len)? };
    ///     interpreter.allocate_tensors()?;
    /// }
    /// # Ok::<(), edgefirst_tflite::Error>(())
    /// ```
    pub unsafe fn set_custom_allocation_for_input(
        &mut self,
        input_index: usize,
        data: NonNull<u8>,
        bytes: usize,
    ) -> Result<()> {
        let fns = self.lib.experimental().ok_or_else(|| {
            Error::unsupported(
                "TfLiteInterpreterSetCustomAllocationForTensor",
                "this TFLite build does not export the experimental C API; \
                 inputs must be copied into the arena instead",
            )
        })?;

        let inputs = self.input_count();
        if input_index >= inputs {
            return Err(Error::invalid_argument(format!(
                "input index {input_index} out of range (interpreter has {inputs} inputs)"
            )));
        }

        // Condition 3 from `c_api_experimental.h`: the buffer must cover the
        // tensor. Checked here so the failure names both sizes instead of
        // surfacing as a bare status from AllocateTensors much later.
        let tensor_bytes = self
            .inputs()?
            .get(input_index)
            .ok_or_else(|| Error::null_pointer(format!("input tensor {input_index} is null")))?
            .byte_size();
        if bytes < tensor_bytes {
            return Err(Error::invalid_argument(format!(
                "custom allocation of {bytes} bytes is smaller than input tensor \
                 {input_index} ({tensor_bytes} bytes)"
            )));
        }

        // Condition 4: alignment. We validate rather than expose the
        // skip-align flag — upstream warns that skipping it "can cause crashes
        // when calling Invoke()", because kernels issue aligned vector loads
        // regardless of what the runtime was told.
        if !(data.as_ptr() as usize).is_multiple_of(TENSOR_ALIGNMENT) {
            return Err(Error::invalid_argument(format!(
                "custom allocation must be {TENSOR_ALIGNMENT}-byte aligned \
                 (kDefaultTensorAlignment), got {:p}",
                data.as_ptr()
            )));
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let input_index_i32 = input_index as i32;
        // SAFETY: `self.ptr` is a valid interpreter and `input_index` was
        // bounds-checked above.
        let tensor_index =
            unsafe { (fns.get_input_tensor_index)(self.ptr.as_ptr(), input_index_i32) };
        if tensor_index < 0 {
            return Err(Error::invalid_argument(format!(
                "TfLiteInterpreterGetInputTensorIndex returned {tensor_index} for \
                 input {input_index}"
            )));
        }

        let allocation = TfLiteCustomAllocation {
            data: data.as_ptr().cast(),
            bytes,
        };
        // SAFETY: `self.ptr` is a valid interpreter, `tensor_index` came from
        // the runtime itself, and `allocation` is read during the call only —
        // the runtime copies the two fields out. The buffer it points at is
        // the caller's obligation, documented under `# Safety`.
        let status = unsafe {
            (fns.set_custom_allocation_for_tensor)(
                self.ptr.as_ptr(),
                tensor_index,
                &raw const allocation,
                kTfLiteCustomAllocationFlagsNone,
            )
        };
        error::status_to_result(status)
            .map_err(|e| e.with_context("TfLiteInterpreterSetCustomAllocationForTensor"))
    }

    /// Resize an input tensor's dimensions.
    ///
    /// After resizing, [`Interpreter::allocate_tensors`] must be called
    /// before inference can proceed.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status (e.g., the
    /// input index is out of range).
    pub fn resize_input(&mut self, input_index: usize, shape: &[i32]) -> Result<()> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let index = input_index as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let dims_size = shape.len() as i32;
        // SAFETY: `self.ptr` is a valid interpreter pointer. `shape` is a
        // valid slice and `dims_size` is its length. The C API copies the
        // shape data, so the slice only needs to be valid for this call.
        let status = unsafe {
            self.lib.as_sys().TfLiteInterpreterResizeInputTensor(
                self.ptr.as_ptr(),
                index,
                shape.as_ptr(),
                dims_size,
            )
        };
        error::status_to_result(status)
            .map_err(|e| e.with_context("TfLiteInterpreterResizeInputTensor"))
    }

    /// Run model inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    pub fn invoke(&mut self) -> Result<()> {
        // SAFETY: `self.ptr` is a valid interpreter pointer with tensors allocated.
        let status = unsafe { self.lib.as_sys().TfLiteInterpreterInvoke(self.ptr.as_ptr()) };
        error::status_to_result(status).map_err(|e| e.with_context("TfLiteInterpreterInvoke"))
    }

    /// Get immutable views of all input tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if any input tensor pointer is null.
    pub fn inputs(&self) -> Result<Vec<Tensor<'_>>> {
        let count = self.input_count();
        let mut inputs = Vec::with_capacity(count);
        for i in 0..count {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            // SAFETY: `self.ptr` is a valid interpreter and `i` is in bounds
            // (below `input_count`).
            let raw = unsafe {
                self.lib
                    .as_sys()
                    .TfLiteInterpreterGetInputTensor(self.ptr.as_ptr(), i as i32)
            };
            if raw.is_null() {
                return Err(Error::null_pointer(format!(
                    "TfLiteInterpreterGetInputTensor returned null for index {i}"
                )));
            }
            inputs.push(Tensor {
                ptr: raw,
                lib: self.lib.as_sys(),
            });
        }
        Ok(inputs)
    }

    /// Get mutable views of all input tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if any input tensor pointer is null.
    pub fn inputs_mut(&mut self) -> Result<Vec<TensorMut<'_>>> {
        let count = self.input_count();
        let mut inputs = Vec::with_capacity(count);
        for i in 0..count {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            // SAFETY: `self.ptr` is a valid interpreter and `i` is in bounds.
            // We hold `&mut self` ensuring exclusive access to the tensor data.
            let raw = unsafe {
                self.lib
                    .as_sys()
                    .TfLiteInterpreterGetInputTensor(self.ptr.as_ptr(), i as i32)
            };
            let ptr = NonNull::new(raw).ok_or_else(|| {
                Error::null_pointer(format!(
                    "TfLiteInterpreterGetInputTensor returned null for index {i}"
                ))
            })?;
            inputs.push(TensorMut {
                ptr,
                lib: self.lib.as_sys(),
            });
        }
        Ok(inputs)
    }

    /// Get immutable views of all output tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if any output tensor pointer is null.
    pub fn outputs(&self) -> Result<Vec<Tensor<'_>>> {
        let count = self.output_count();
        let mut outputs = Vec::with_capacity(count);
        for i in 0..count {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            // SAFETY: `self.ptr` is a valid interpreter and `i` is in bounds
            // (below `output_count`).
            let raw = unsafe {
                self.lib
                    .as_sys()
                    .TfLiteInterpreterGetOutputTensor(self.ptr.as_ptr(), i as i32)
            };
            if raw.is_null() {
                return Err(Error::null_pointer(format!(
                    "TfLiteInterpreterGetOutputTensor returned null for index {i}"
                )));
            }
            outputs.push(Tensor {
                ptr: raw,
                lib: self.lib.as_sys(),
            });
        }
        Ok(outputs)
    }

    /// Returns the number of input tensors.
    #[must_use]
    pub fn input_count(&self) -> usize {
        // SAFETY: `self.ptr` is a valid interpreter pointer.
        #[allow(clippy::cast_sign_loss)]
        let count = unsafe {
            self.lib
                .as_sys()
                .TfLiteInterpreterGetInputTensorCount(self.ptr.as_ptr())
        } as usize;
        count
    }

    /// Returns the number of output tensors.
    #[must_use]
    pub fn output_count(&self) -> usize {
        // SAFETY: `self.ptr` is a valid interpreter pointer.
        #[allow(clippy::cast_sign_loss)]
        let count = unsafe {
            self.lib
                .as_sys()
                .TfLiteInterpreterGetOutputTensorCount(self.ptr.as_ptr())
        } as usize;
        count
    }

    /// Access all delegates owned by this interpreter.
    #[must_use]
    pub fn delegates(&self) -> &[Delegate] {
        &self.delegates
    }

    /// Access a specific delegate by index.
    #[must_use]
    pub fn delegate(&self, index: usize) -> Option<&Delegate> {
        self.delegates.get(index)
    }
}

// SAFETY: `Interpreter` holds a `NonNull<TfLiteInterpreter>` (opaque C
// handle with no thread affinity), a `Vec<Delegate>` (`Send`), and a
// `&Library` reference (`Sync`). The TFLite C API allows an interpreter
// to be used from any single thread — it just must not be accessed
// concurrently from multiple threads (hence `Send` but not `Sync`).
unsafe impl Send for Interpreter<'_> {}

impl std::fmt::Debug for Interpreter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interpreter")
            .field("ptr", &self.ptr)
            .field("delegates", &self.delegates.len())
            .finish()
    }
}

impl Drop for Interpreter<'_> {
    fn drop(&mut self) {
        // SAFETY: The interpreter was created by `TfLiteInterpreterCreate` and
        // has not been deleted. Delegates are dropped after the interpreter
        // since they are stored in the same struct and Rust drops fields in
        // declaration order.
        unsafe {
            self.lib.as_sys().TfLiteInterpreterDelete(self.ptr.as_ptr());
        }
    }
}
