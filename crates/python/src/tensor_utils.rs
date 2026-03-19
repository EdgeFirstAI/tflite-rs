// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Tensor type mapping, numpy conversion, and detail dict builder.

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use edgefirst_tflite::tensor::TensorType;

use crate::error::{self, TfLiteError};

/// Return an error for unsupported tensor types.
fn unsupported_type(tt: TensorType) -> PyErr {
    TfLiteError::new_err(format!("unsupported tensor type for numpy: {tt:?}"))
}

/// Return the numpy dtype string for a [`TensorType`].
pub fn dtype_str(tt: TensorType) -> PyResult<&'static str> {
    match tt {
        TensorType::Float32 => Ok("float32"),
        TensorType::Float64 => Ok("float64"),
        TensorType::Float16 => Ok("float16"),
        TensorType::Int8 => Ok("int8"),
        TensorType::UInt8 => Ok("uint8"),
        TensorType::Int16 => Ok("int16"),
        TensorType::UInt16 => Ok("uint16"),
        TensorType::Int32 => Ok("int32"),
        TensorType::UInt32 => Ok("uint32"),
        TensorType::Int64 => Ok("int64"),
        TensorType::UInt64 => Ok("uint64"),
        TensorType::Bool => Ok("bool"),
        _ => Err(unsupported_type(tt)),
    }
}

/// Build a tensor detail dict matching the `tflite_runtime` format.
pub fn build_detail_dict<'py>(
    py: Python<'py>,
    name: &str,
    index: usize,
    shape: &[usize],
    tensor_type: TensorType,
    scale: f32,
    zero_point: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let np = py.import("numpy")?;

    let dict = PyDict::new(py);
    dict.set_item("name", name)?;
    dict.set_item("index", index)?;

    // shape as numpy int32 array
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    let shape_arr = numpy::PyArray1::from_vec(py, shape_i32);
    dict.set_item("shape", shape_arr)?;

    // dtype as numpy dtype object
    let dtype_s = dtype_str(tensor_type)?;
    let dtype_obj = np.call_method1("dtype", (dtype_s,))?;
    dict.set_item("dtype", dtype_obj)?;

    // Legacy quantization: (scale, zero_point)
    dict.set_item("quantization", (scale, zero_point))?;

    // quantization_parameters dict
    let qparams = PyDict::new(py);
    let scales_arr = numpy::PyArray1::from_vec(py, vec![scale]);
    qparams.set_item("scales", scales_arr)?;
    let zp_arr = numpy::PyArray1::from_vec(py, vec![zero_point]);
    qparams.set_item("zero_points", zp_arr)?;
    qparams.set_item("quantized_dimension", 0)?;
    dict.set_item("quantization_parameters", qparams)?;

    Ok(dict)
}

/// Copy tensor data into a new numpy array (for `get_tensor`).
///
/// Dispatches on [`TensorType`] to create a correctly-typed numpy array.
pub fn tensor_to_numpy(
    py: Python<'_>,
    tensor: &edgefirst_tflite::Tensor<'_>,
) -> PyResult<PyObject> {
    let shape = tensor.shape().map_err(error::to_py_err)?;

    macro_rules! copy_to_numpy {
        ($T:ty) => {{
            let data = tensor.as_slice::<$T>().map_err(error::to_py_err)?;
            let arr = ArrayD::<$T>::from_shape_vec(IxDyn(&shape), data.to_vec())
                .map_err(|e| TfLiteError::new_err(e.to_string()))?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }};
    }

    match tensor.tensor_type() {
        TensorType::Float32 => copy_to_numpy!(f32),
        TensorType::Float64 => copy_to_numpy!(f64),
        TensorType::Float16 => copy_to_numpy!(half::f16),
        TensorType::Int8 => copy_to_numpy!(i8),
        TensorType::UInt8 => copy_to_numpy!(u8),
        TensorType::Int16 => copy_to_numpy!(i16),
        TensorType::UInt16 => copy_to_numpy!(u16),
        TensorType::Int32 => copy_to_numpy!(i32),
        TensorType::UInt32 => copy_to_numpy!(u32),
        TensorType::Int64 => copy_to_numpy!(i64),
        TensorType::UInt64 => copy_to_numpy!(u64),
        TensorType::Bool => copy_to_numpy!(bool),
        other => Err(unsupported_type(other)),
    }
}

/// Copy data from a numpy array into a mutable tensor (for `set_tensor`).
///
/// Dispatches on [`TensorType`] to extract the correct element type.
pub fn numpy_to_tensor(
    _py: Python<'_>,
    value: &Bound<'_, pyo3::PyAny>,
    tensor: &mut edgefirst_tflite::TensorMut<'_>,
) -> PyResult<()> {
    macro_rules! copy_from_numpy {
        ($T:ty) => {{
            let arr: numpy::PyReadonlyArrayDyn<'_, $T> = value.extract()?;
            let slice = arr
                .as_slice()
                .map_err(|e| TfLiteError::new_err(e.to_string()))?;
            tensor.copy_from_slice(slice).map_err(error::to_py_err)?;
            Ok(())
        }};
    }

    match tensor.tensor_type() {
        TensorType::Float32 => copy_from_numpy!(f32),
        TensorType::Float64 => copy_from_numpy!(f64),
        TensorType::Float16 => copy_from_numpy!(half::f16),
        TensorType::Int8 => copy_from_numpy!(i8),
        TensorType::UInt8 => copy_from_numpy!(u8),
        TensorType::Int16 => copy_from_numpy!(i16),
        TensorType::UInt16 => copy_from_numpy!(u16),
        TensorType::Int32 => copy_from_numpy!(i32),
        TensorType::UInt32 => copy_from_numpy!(u32),
        TensorType::Int64 => copy_from_numpy!(i64),
        TensorType::UInt64 => copy_from_numpy!(u64),
        TensorType::Bool => copy_from_numpy!(bool),
        other => Err(unsupported_type(other)),
    }
}

/// Create a zero-copy numpy view of tensor data (for `tensor()`).
///
/// The returned numpy array shares memory with the `TFLite` C-allocated buffer.
/// The `container` Python object is stored as the array's base to prevent the
/// interpreter from being garbage collected while the view is alive.
///
/// # Safety
///
/// The data pointer must remain valid for the lifetime of the returned array.
/// The view is valid across `invoke()` calls (which write in-place without
/// reallocating) but is invalidated by `allocate_tensors()` or
/// `resize_input()` which may free and reallocate the underlying C buffers.
pub fn tensor_view_numpy<'py>(
    _py: Python<'py>,
    tensor: &edgefirst_tflite::Tensor<'_>,
    container: Bound<'py, pyo3::PyAny>,
) -> PyResult<PyObject> {
    let shape = tensor.shape().map_err(error::to_py_err)?;

    macro_rules! borrow_view {
        ($T:ty) => {{
            let data = tensor.as_slice::<$T>().map_err(error::to_py_err)?;
            let view = numpy::ndarray::ArrayViewD::<$T>::from_shape(IxDyn(&shape), data)
                .map_err(|e| TfLiteError::new_err(e.to_string()))?;
            // SAFETY: The data pointer comes from TFLite C-allocated memory that
            // lives as long as the interpreter. `container` prevents the interpreter
            // from being garbage collected.
            let py_arr = unsafe { PyArrayDyn::<$T>::borrow_from_array(&view, container) };
            Ok(py_arr.into_any().unbind())
        }};
    }

    match tensor.tensor_type() {
        TensorType::Float32 => borrow_view!(f32),
        TensorType::Float64 => borrow_view!(f64),
        TensorType::Float16 => borrow_view!(half::f16),
        TensorType::Int8 => borrow_view!(i8),
        TensorType::UInt8 => borrow_view!(u8),
        TensorType::Int16 => borrow_view!(i16),
        TensorType::UInt16 => borrow_view!(u16),
        TensorType::Int32 => borrow_view!(i32),
        TensorType::UInt32 => borrow_view!(u32),
        TensorType::Int64 => borrow_view!(i64),
        TensorType::UInt64 => borrow_view!(u64),
        TensorType::Bool => borrow_view!(bool),
        other => Err(unsupported_type(other)),
    }
}
