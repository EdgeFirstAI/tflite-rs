// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Python `Interpreter` class wrapping the full `Library → Model → Interpreter`
//! ownership chain with erased lifetimes.

use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::pin::Pin;

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::delegate::PyDelegate;
use crate::error::{self, InvalidArgumentError, TfLiteError};
use crate::tensor_utils;

// ---------------------------------------------------------------------------
// InterpreterOwned — lifetime-erased ownership of the full chain
// ---------------------------------------------------------------------------

/// Owns `Library`, `Model`, and `Interpreter` with erased lifetimes.
///
/// # Safety
///
/// Lifetimes are erased via `unsafe` pointer casts. Safety is maintained
/// because:
/// - All references point into the same struct (heap-pinned `Library`).
/// - The custom `Drop` enforces reverse-dependency teardown order:
///   `Interpreter` → `Model` → `Library`.
struct InterpreterOwned {
    interpreter: ManuallyDrop<edgefirst_tflite::Interpreter<'static>>,
    model: ManuallyDrop<edgefirst_tflite::Model<'static>>,
    /// Bumped on `allocate_tensors()` and `resize_tensor_input()`. Used by
    /// `TensorAccessor` to detect stale zero-copy views whose backing
    /// memory may have been freed and reallocated.
    allocation_generation: u64,
    // IMPORTANT: `library` must be the LAST field. After Drop::drop manually
    // drops `interpreter` and `model`, Rust auto-drops remaining fields in
    // declaration order. `library` must outlive both, so it must be last.
    #[allow(dead_code)]
    library: Pin<Box<edgefirst_tflite::Library>>,
}

impl InterpreterOwned {
    fn new(
        library: edgefirst_tflite::Library,
        model_data: Vec<u8>,
        num_threads: Option<i32>,
        delegates: Vec<edgefirst_tflite::Delegate>,
    ) -> Result<Self, edgefirst_tflite::Error> {
        let library = Pin::new(Box::new(library));

        // SAFETY: The Library is heap-allocated via Box, so its address is
        // stable for the lifetime of this struct. Pin prevents external code
        // from replacing the value. The transmuted 'static reference is valid
        // because:
        // 1. Box<Library> ensures the address never changes (heap-stable).
        // 2. The custom Drop impl drops `interpreter` then `model` before
        //    `library` (which is the last field — Rust drops fields in
        //    declaration order after the explicit Drop::drop runs).
        // 3. InterpreterOwned is only ever stored inside a PyO3 #[pyclass],
        //    which heap-allocates the Rust struct.
        let lib_ref: &'static edgefirst_tflite::Library = unsafe { &*std::ptr::addr_of!(*library) };

        let model = edgefirst_tflite::Model::from_bytes(lib_ref, model_data)?;

        let mut builder = edgefirst_tflite::Interpreter::builder(lib_ref)?;
        if let Some(n) = num_threads {
            builder = builder.num_threads(n);
        }
        for d in delegates {
            builder = builder.delegate(d);
        }
        let interpreter = builder.build(&model)?;

        Ok(Self {
            interpreter: ManuallyDrop::new(interpreter),
            model: ManuallyDrop::new(model),
            allocation_generation: 0,
            library,
        })
    }
}

impl Drop for InterpreterOwned {
    fn drop(&mut self) {
        // SAFETY: Drop in reverse dependency order: interpreter → model → library.
        // Each ManuallyDrop value is dropped exactly once here. After this
        // Drop::drop returns, the remaining field `library: Pin<Box<Library>>`
        // is dropped automatically by the compiler. This ordering is correct
        // because `library` is the LAST declared field — do NOT reorder the
        // struct fields without updating this Drop impl.
        unsafe {
            ManuallyDrop::drop(&mut self.interpreter);
            ManuallyDrop::drop(&mut self.model);
        }
    }
}

// ---------------------------------------------------------------------------
// TensorAccessor — callable returning zero-copy numpy view
// ---------------------------------------------------------------------------

/// Callable returned by `Interpreter.tensor()`.
///
/// Holds a reference to the interpreter to prevent garbage collection.
/// Each call returns a zero-copy numpy view of the tensor data.
///
/// # Safety
///
/// The returned numpy array shares memory with `TFLite` C-allocated buffers.
/// The view is invalidated by `allocate_tensors()` or `resize_tensor_input()`
/// which may free and reallocate the underlying C buffers. The generation
/// counter detects this and raises an error on the next `__call__`.
///
/// Note: `invoke()` overwrites tensor data **in-place** but does not
/// reallocate — the accessor and its views remain valid across `invoke()`
/// calls (the view simply reflects the latest inference results).
#[pyclass(name = "_TensorAccessor", unsendable)]
struct TensorAccessor {
    interp: Py<PyInterpreter>,
    tensor_index: usize,
    is_input: bool,
    /// Snapshot of `InterpreterOwned::allocation_generation` at creation time.
    generation: u64,
}

#[pymethods]
impl TensorAccessor {
    fn __call__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let interp = self.interp.bind(py);
        let interp_ref = interp.borrow();

        // Check for stale view after allocate_tensors() / resize.
        if self.generation != interp_ref.inner.allocation_generation {
            return Err(TfLiteError::new_err(
                "tensor view invalidated by allocate_tensors() — call tensor() again to get a fresh accessor",
            ));
        }

        let container = interp.clone().into_any().unbind().into_bound(py);

        let (tensors, kind) = if self.is_input {
            (interp_ref.inner.interpreter.inputs(), "input")
        } else {
            (interp_ref.inner.interpreter.outputs(), "output")
        };
        let tensors = tensors.map_err(error::to_py_err)?;

        let tensor = tensors.get(self.tensor_index).ok_or_else(|| {
            InvalidArgumentError::new_err(format!(
                "{kind} tensor index {} out of range",
                self.tensor_index
            ))
        })?;
        tensor_utils::tensor_view_numpy(py, tensor, container)
    }
}

// ---------------------------------------------------------------------------
// PyInterpreter — the main Python class
// ---------------------------------------------------------------------------

/// `TFLite` model interpreter.
///
/// API-compatible with `tflite_runtime.interpreter.Interpreter` for the core
/// inference path, with `EdgeFirst` extensions for DMA-BUF and `CameraAdaptor`.
#[pyclass(name = "Interpreter", unsendable)]
pub struct PyInterpreter {
    inner: InterpreterOwned,
}

impl std::fmt::Debug for PyInterpreter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interpreter").finish_non_exhaustive()
    }
}

#[pymethods]
impl PyInterpreter {
    /// Create a new Interpreter.
    ///
    /// Args:
    ///     `model_path`: Path to a `.tflite` model file.
    ///     `model_content`: Raw model bytes.
    ///     `num_threads`: Number of inference threads (None = auto).
    ///     `experimental_delegates`: List of `Delegate` objects for HW accel.
    ///     `library_path`: Path to `libtensorflowlite_c.so` (`EdgeFirst` extension).
    #[new]
    #[pyo3(signature = (model_path=None, model_content=None, num_threads=None, experimental_delegates=None, *, library_path=None))]
    fn new(
        _py: Python<'_>,
        model_path: Option<PathBuf>,
        model_content: Option<Vec<u8>>,
        num_threads: Option<i32>,
        experimental_delegates: Option<Bound<'_, PyList>>,
        library_path: Option<PathBuf>,
    ) -> PyResult<Self> {
        // Load the TFLite shared library.
        let library = if let Some(path) = library_path {
            edgefirst_tflite::Library::from_path(path)
        } else {
            edgefirst_tflite::Library::new()
        }
        .map_err(error::to_py_err)?;

        // Load model data.
        let model_data = if let Some(content) = model_content {
            content
        } else if let Some(path) = model_path {
            std::fs::read(&path)
                .map_err(|e| InvalidArgumentError::new_err(format!("{}: {e}", path.display())))?
        } else {
            return Err(InvalidArgumentError::new_err(
                "either model_path or model_content must be provided",
            ));
        };

        // Extract delegates (consumes them from the Python Delegate objects).
        let mut delegates = Vec::new();
        if let Some(py_delegates) = experimental_delegates {
            for item in py_delegates.iter() {
                let mut del: PyRefMut<'_, PyDelegate> = item.extract()?;
                let d = del.take_inner().ok_or_else(|| {
                    InvalidArgumentError::new_err(
                        "delegate already consumed by another Interpreter",
                    )
                })?;
                delegates.push(d);
            }
        }

        let inner = InterpreterOwned::new(library, model_data, num_threads, delegates)
            .map_err(error::to_py_err)?;

        Ok(Self { inner })
    }

    /// Re-allocate tensors. Required after `resize_tensor_input()`.
    ///
    /// Invalidates any existing zero-copy tensor views obtained via `tensor()`.
    /// Tensors are also allocated during `__init__`, so this only needs to be
    /// called explicitly after resizing inputs.
    fn allocate_tensors(&mut self) -> PyResult<()> {
        self.inner
            .interpreter
            .allocate_tensors()
            .map_err(error::to_py_err)?;
        self.inner.allocation_generation = self.inner.allocation_generation.wrapping_add(1);
        Ok(())
    }

    /// Resize an input tensor's dimensions.
    ///
    /// Must call `allocate_tensors()` after resizing before calling `invoke()`.
    /// Immediately invalidates any existing zero-copy tensor views from
    /// `tensor()` — the underlying buffers are in an inconsistent state
    /// until `allocate_tensors()` is called.
    #[allow(clippy::needless_pass_by_value)]
    fn resize_tensor_input(&mut self, input_index: usize, tensor_size: Vec<i32>) -> PyResult<()> {
        self.inner
            .interpreter
            .resize_input(input_index, &tensor_size)
            .map_err(error::to_py_err)?;
        self.inner.allocation_generation = self.inner.allocation_generation.wrapping_add(1);
        Ok(())
    }

    /// Run model inference.
    fn invoke(&mut self) -> PyResult<()> {
        self.inner.interpreter.invoke().map_err(error::to_py_err)
    }

    /// Return input tensor details as a list of dicts.
    fn get_input_details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let tensors = self.inner.interpreter.inputs().map_err(error::to_py_err)?;
        build_tensor_details_list(py, &tensors)
    }

    /// Return output tensor details as a list of dicts.
    fn get_output_details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let tensors = self.inner.interpreter.outputs().map_err(error::to_py_err)?;
        build_tensor_details_list(py, &tensors)
    }

    /// Return a copy of an input tensor's data as a numpy array.
    fn get_input_tensor(&self, py: Python<'_>, input_index: usize) -> PyResult<PyObject> {
        let tensors = self.inner.interpreter.inputs().map_err(error::to_py_err)?;
        copy_tensor_to_numpy(py, &tensors, input_index, "input")
    }

    /// Return a copy of an output tensor's data as a numpy array.
    fn get_output_tensor(&self, py: Python<'_>, output_index: usize) -> PyResult<PyObject> {
        let tensors = self.inner.interpreter.outputs().map_err(error::to_py_err)?;
        copy_tensor_to_numpy(py, &tensors, output_index, "output")
    }

    /// Copy numpy array data into an input tensor.
    #[allow(clippy::needless_pass_by_value)]
    fn set_tensor(&mut self, input_index: usize, value: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut tensors = self
            .inner
            .interpreter
            .inputs_mut()
            .map_err(error::to_py_err)?;
        let count = tensors.len();
        let tensor = tensors.get_mut(input_index).ok_or_else(|| {
            InvalidArgumentError::new_err(format!(
                "input tensor index {input_index} out of range (count: {count})",
            ))
        })?;
        tensor_utils::numpy_to_tensor(py, &value, tensor)
    }

    /// Return a callable that yields a zero-copy numpy view of tensor data.
    ///
    /// The view shares memory with the `TFLite` C-allocated buffer and
    /// reflects the latest inference results after each `invoke()` call.
    /// The accessor is invalidated by `allocate_tensors()` or
    /// `resize_tensor_input()` — call `tensor()` again to get a fresh one.
    fn tensor(slf: Bound<'_, Self>, tensor_index: usize) -> PyResult<TensorAccessor> {
        let interp = slf.borrow();
        let input_count = interp.inner.interpreter.input_count();
        let output_count = interp.inner.interpreter.output_count();

        let (actual_index, is_input) = if tensor_index < input_count {
            (tensor_index, true)
        } else if tensor_index < input_count + output_count {
            (tensor_index - input_count, false)
        } else {
            return Err(InvalidArgumentError::new_err(format!(
                "tensor index {tensor_index} out of range (inputs: {input_count}, outputs: {output_count})"
            )));
        };

        let generation = interp.inner.allocation_generation;

        Ok(TensorAccessor {
            interp: slf.unbind(),
            tensor_index: actual_index,
            is_input,
            generation,
        })
    }

    /// Number of input tensors.
    #[getter]
    fn input_count(&self) -> usize {
        self.inner.interpreter.input_count()
    }

    /// Number of output tensors.
    #[getter]
    fn output_count(&self) -> usize {
        self.inner.interpreter.output_count()
    }

    /// Access a delegate owned by this interpreter.
    #[pyo3(signature = (index=0))]
    fn delegate(slf: Bound<'_, Self>, index: usize) -> Option<crate::delegate::PyDelegateRef> {
        let interp = slf.borrow();
        interp
            .inner
            .interpreter
            .delegate(index)
            .map(|_| crate::delegate::PyDelegateRef {
                interp: slf.unbind(),
                delegate_index: index,
            })
    }

    /// Get a `DmaBuf` interface for a delegate's DMA-BUF extensions.
    ///
    /// Returns `None` if the delegate does not support DMA-BUF.
    #[pyo3(signature = (delegate_index=0))]
    fn dmabuf(slf: Bound<'_, Self>, delegate_index: usize) -> Option<crate::dmabuf::PyDmaBuf> {
        delegate_has_feature(&slf, delegate_index, edgefirst_tflite::Delegate::has_dmabuf).then(
            || crate::dmabuf::PyDmaBuf {
                interp: slf.unbind(),
                delegate_index,
            },
        )
    }

    /// Get a `CameraAdaptor` interface for a delegate's NPU preprocessing.
    ///
    /// Returns `None` if the delegate does not support `CameraAdaptor`.
    #[pyo3(signature = (delegate_index=0))]
    fn camera_adaptor(
        slf: Bound<'_, Self>,
        delegate_index: usize,
    ) -> Option<crate::camera_adaptor::PyCameraAdaptor> {
        delegate_has_feature(
            &slf,
            delegate_index,
            edgefirst_tflite::Delegate::has_camera_adaptor,
        )
        .then(|| crate::camera_adaptor::PyCameraAdaptor {
            interp: slf.unbind(),
            delegate_index,
        })
    }

    /// Extract model metadata (requires `metadata` feature in the Rust crate).
    fn get_metadata(&self) -> Option<crate::metadata::PyMetadata> {
        let data = self.inner.model.data();
        let meta = edgefirst_tflite::metadata::Metadata::from_model_bytes(data);
        // Return None if all fields are empty.
        if meta == edgefirst_tflite::metadata::Metadata::default() {
            None
        } else {
            Some(crate::metadata::PyMetadata { inner: meta })
        }
    }
}

/// Copy tensor data to a numpy array, with bounds checking.
fn copy_tensor_to_numpy(
    py: Python<'_>,
    tensors: &[edgefirst_tflite::Tensor<'_>],
    index: usize,
    kind: &str,
) -> PyResult<PyObject> {
    let tensor = tensors.get(index).ok_or_else(|| {
        InvalidArgumentError::new_err(format!(
            "{kind} tensor index {index} out of range (count: {})",
            tensors.len()
        ))
    })?;
    tensor_utils::tensor_to_numpy(py, tensor)
}

/// Check if a delegate at the given index supports a feature.
fn delegate_has_feature(
    slf: &Bound<'_, PyInterpreter>,
    index: usize,
    check: fn(&edgefirst_tflite::Delegate) -> bool,
) -> bool {
    let interp = slf.borrow();
    interp.inner.interpreter.delegate(index).is_some_and(check)
}

/// Build a list of tensor detail dicts from a slice of tensors.
fn build_tensor_details_list<'py>(
    py: Python<'py>,
    tensors: &[edgefirst_tflite::Tensor<'_>],
) -> PyResult<Bound<'py, PyList>> {
    let mut details = Vec::with_capacity(tensors.len());
    for (i, t) in tensors.iter().enumerate() {
        let shape = t.shape().map_err(error::to_py_err)?;
        let qp = t.quantization_params();
        let dict = tensor_utils::build_detail_dict(
            py,
            t.name(),
            i,
            &shape,
            t.tensor_type(),
            qp.scale,
            qp.zero_point,
        )?;
        details.push(dict);
    }
    PyList::new(py, details)
}

/// Access the inner interpreter from delegate/dmabuf/`camera_adaptor` wrappers.
impl PyInterpreter {
    pub(crate) fn with_delegate<F, R>(&self, index: usize, f: F) -> PyResult<R>
    where
        F: FnOnce(&edgefirst_tflite::Delegate) -> PyResult<R>,
    {
        let delegate =
            self.inner.interpreter.delegate(index).ok_or_else(|| {
                TfLiteError::new_err(format!("delegate index {index} out of range"))
            })?;
        f(delegate)
    }
}
