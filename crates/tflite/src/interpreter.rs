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

use edgefirst_tflite_sys::{TfLiteInterpreter, TfLiteInterpreterOptions};

use crate::delegate::Delegate;
use crate::error::{self, Error, Result};
use crate::model::Model;
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
