// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Type-safe tensor wrappers for the TensorFlow Lite C API.
//!
//! This module provides [`Tensor`] (immutable) and [`TensorMut`] (mutable)
//! views over the raw `TfLiteTensor` pointers returned by the C API. Both
//! types expose shape introspection, quantization parameters, and typed
//! data access via slices.
//!
//! # Tensor types
//!
//! The [`TensorType`] enum mirrors the `TfLiteType` constants from the C
//! header, providing a safe Rust-side representation that can be pattern
//! matched.
//!
//! # Data access
//!
//! Use [`Tensor::as_slice`] for read-only access and
//! [`TensorMut::as_mut_slice`] or [`TensorMut::copy_from_slice`] for
//! write access to the underlying tensor buffer.

use std::ffi::CStr;
use std::fmt;
use std::ptr::NonNull;

use edgefirst_tflite_sys::{
    self as sys, TfLiteTensor, TfLiteType_kTfLiteBFloat16, TfLiteType_kTfLiteBool,
    TfLiteType_kTfLiteComplex128, TfLiteType_kTfLiteComplex64, TfLiteType_kTfLiteFloat16,
    TfLiteType_kTfLiteFloat32, TfLiteType_kTfLiteFloat64, TfLiteType_kTfLiteInt16,
    TfLiteType_kTfLiteInt32, TfLiteType_kTfLiteInt4, TfLiteType_kTfLiteInt64,
    TfLiteType_kTfLiteInt8, TfLiteType_kTfLiteNoType, TfLiteType_kTfLiteResource,
    TfLiteType_kTfLiteString, TfLiteType_kTfLiteUInt16, TfLiteType_kTfLiteUInt32,
    TfLiteType_kTfLiteUInt64, TfLiteType_kTfLiteUInt8, TfLiteType_kTfLiteVariant,
};
use num_traits::FromPrimitive;

use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// TensorType
// ---------------------------------------------------------------------------

/// Element data type of a TensorFlow Lite tensor.
///
/// Each variant corresponds to a `kTfLite*` constant from the C API header
/// `common.h`. The discriminant values match the C constants so that
/// conversion via [`FromPrimitive`] is a zero-cost identity check.
///
/// # Example
///
/// ```ignore
/// let ty = tensor.tensor_type();
/// match ty {
///     TensorType::Float32 => println!("32-bit float tensor"),
///     TensorType::UInt8   => println!("quantized uint8 tensor"),
///     _ => println!("other type: {ty:?}"),
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_derive::FromPrimitive)]
#[repr(isize)]
#[allow(clippy::cast_possible_wrap)] // C constants are small u32 values; no wrap on any target.
pub enum TensorType {
    /// No type information (`kTfLiteNoType`).
    NoType = TfLiteType_kTfLiteNoType as isize,
    /// 32-bit IEEE 754 float (`kTfLiteFloat32`).
    Float32 = TfLiteType_kTfLiteFloat32 as isize,
    /// 32-bit signed integer (`kTfLiteInt32`).
    Int32 = TfLiteType_kTfLiteInt32 as isize,
    /// 8-bit unsigned integer (`kTfLiteUInt8`).
    UInt8 = TfLiteType_kTfLiteUInt8 as isize,
    /// 64-bit signed integer (`kTfLiteInt64`).
    Int64 = TfLiteType_kTfLiteInt64 as isize,
    /// Variable-length string (`kTfLiteString`).
    String = TfLiteType_kTfLiteString as isize,
    /// Boolean (`kTfLiteBool`).
    Bool = TfLiteType_kTfLiteBool as isize,
    /// 16-bit signed integer (`kTfLiteInt16`).
    Int16 = TfLiteType_kTfLiteInt16 as isize,
    /// 64-bit complex float (`kTfLiteComplex64`).
    Complex64 = TfLiteType_kTfLiteComplex64 as isize,
    /// 8-bit signed integer (`kTfLiteInt8`).
    Int8 = TfLiteType_kTfLiteInt8 as isize,
    /// 16-bit IEEE 754 half-precision float (`kTfLiteFloat16`).
    Float16 = TfLiteType_kTfLiteFloat16 as isize,
    /// 64-bit IEEE 754 double-precision float (`kTfLiteFloat64`).
    Float64 = TfLiteType_kTfLiteFloat64 as isize,
    /// 128-bit complex float (`kTfLiteComplex128`).
    Complex128 = TfLiteType_kTfLiteComplex128 as isize,
    /// 64-bit unsigned integer (`kTfLiteUInt64`).
    UInt64 = TfLiteType_kTfLiteUInt64 as isize,
    /// Resource handle (`kTfLiteResource`).
    Resource = TfLiteType_kTfLiteResource as isize,
    /// Variant type (`kTfLiteVariant`).
    Variant = TfLiteType_kTfLiteVariant as isize,
    /// 32-bit unsigned integer (`kTfLiteUInt32`).
    UInt32 = TfLiteType_kTfLiteUInt32 as isize,
    /// 16-bit unsigned integer (`kTfLiteUInt16`).
    UInt16 = TfLiteType_kTfLiteUInt16 as isize,
    /// 4-bit signed integer (`kTfLiteInt4`).
    Int4 = TfLiteType_kTfLiteInt4 as isize,
    /// Brain floating-point 16-bit (`kTfLiteBFloat16`).
    BFloat16 = TfLiteType_kTfLiteBFloat16 as isize,
}

// ---------------------------------------------------------------------------
// QuantizationParams
// ---------------------------------------------------------------------------

/// Affine quantization parameters for a tensor.
///
/// Quantized values can be converted back to floating point using:
///
/// ```text
/// real_value = scale * (quantized_value - zero_point)
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizationParams {
    /// Scale factor for dequantization.
    pub scale: f32,
    /// Zero-point offset for dequantization.
    pub zero_point: i32,
}

// ---------------------------------------------------------------------------
// Tensor (immutable view)
// ---------------------------------------------------------------------------

/// An immutable view of a TensorFlow Lite tensor.
///
/// `Tensor` borrows the underlying C tensor pointer and the dynamically
/// loaded library handle for the duration of its lifetime `'a`. It provides
/// read-only access to tensor metadata (name, shape, type) and data.
///
/// Use [`Tensor::as_slice`] to obtain a typed slice over the tensor data.
pub struct Tensor<'a> {
    /// Raw pointer to the C `TfLiteTensor`.
    ///
    /// This is a raw `*const` pointer (not `NonNull`) because the C API
    /// returns `*const TfLiteTensor` for output tensors.
    pub(crate) ptr: *const TfLiteTensor,

    /// Reference to the dynamically loaded `TFLite` C library.
    pub(crate) lib: &'a sys::tensorflowlite_c,
}

impl Tensor<'_> {
    /// Returns the element data type of this tensor.
    ///
    /// If the C API returns a type value not represented by [`TensorType`],
    /// this method defaults to [`TensorType::NoType`].
    #[must_use]
    pub fn tensor_type(&self) -> TensorType {
        // SAFETY: `self.ptr` is a valid tensor pointer obtained from the
        // interpreter and `self.lib` is a valid reference to the loaded library.
        let raw = unsafe { self.lib.TfLiteTensorType(self.ptr) };
        FromPrimitive::from_u32(raw).unwrap_or(TensorType::NoType)
    }

    /// Returns the name of this tensor as a string slice.
    ///
    /// Returns `"<invalid-utf8>"` if the C API returns a name that is not
    /// valid UTF-8.
    #[must_use]
    pub fn name(&self) -> &str {
        // SAFETY: `self.ptr` is a valid tensor pointer; the C API returns a
        // NUL-terminated string that lives as long as the tensor.
        unsafe { CStr::from_ptr(self.lib.TfLiteTensorName(self.ptr)) }
            .to_str()
            .unwrap_or("<invalid-utf8>")
    }

    /// Returns the number of dimensions (rank) of this tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor does not have its dimensions set
    /// (the C API returns -1).
    pub fn num_dims(&self) -> Result<usize> {
        // SAFETY: `self.ptr` is a valid tensor pointer.
        let n = unsafe { self.lib.TfLiteTensorNumDims(self.ptr) };
        usize::try_from(n).map_err(|_| {
            Error::invalid_argument(format!(
                "tensor `{}` does not have dimensions set",
                self.name()
            ))
        })
    }

    /// Returns the size of the `index`-th dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of bounds (>= `num_dims`).
    pub fn dim(&self, index: usize) -> Result<usize> {
        let num_dims = self.num_dims()?;
        if index >= num_dims {
            return Err(Error::invalid_argument(format!(
                "dimension index {index} out of bounds for tensor with {num_dims} dimensions"
            )));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let i = index as i32;
        // SAFETY: `self.ptr` is valid and `i` is bounds-checked above.
        let d = unsafe { self.lib.TfLiteTensorDim(self.ptr, i) };
        // `d` is non-negative because the C API guarantees valid dimension
        // sizes for in-bounds indices.
        #[allow(clippy::cast_sign_loss)]
        Ok(d as usize)
    }

    /// Returns the full shape of this tensor as a `Vec<usize>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor dimensions are not set.
    pub fn shape(&self) -> Result<Vec<usize>> {
        let num_dims = self.num_dims()?;
        let mut dims = Vec::with_capacity(num_dims);
        for i in 0..num_dims {
            dims.push(self.dim(i)?);
        }
        Ok(dims)
    }

    /// Returns the total number of bytes required to store this tensor's data.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        // SAFETY: `self.ptr` is a valid tensor pointer.
        unsafe { self.lib.TfLiteTensorByteSize(self.ptr) }
    }

    /// Returns the total number of elements in this tensor (product of all
    /// dimensions).
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor dimensions are not set.
    pub fn volume(&self) -> Result<usize> {
        Ok(self.shape()?.iter().product::<usize>())
    }

    /// Returns the affine quantization parameters for this tensor.
    #[must_use]
    pub fn quantization_params(&self) -> QuantizationParams {
        // SAFETY: `self.ptr` is a valid tensor pointer.
        let params = unsafe { self.lib.TfLiteTensorQuantizationParams(self.ptr) };
        QuantizationParams {
            scale: params.scale,
            zero_point: params.zero_point,
        }
    }

    /// Returns an immutable slice over the tensor data, interpreted as
    /// elements of type `T`.
    ///
    /// The slice length equals [`Tensor::volume`]. The caller must ensure
    /// that `T` matches the tensor's actual element type (e.g., `f32` for
    /// a `Float32` tensor, `u8` for a `UInt8` tensor).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `size_of::<T>() * volume` exceeds [`Tensor::byte_size`]
    /// - The underlying data pointer is null (tensor not yet allocated)
    pub fn as_slice<T: Copy>(&self) -> Result<&[T]> {
        let volume = self.volume()?;
        if std::mem::size_of::<T>() * volume > self.byte_size() {
            return Err(Error::invalid_argument(format!(
                "tensor byte size {} is too small for {} elements of {}",
                self.byte_size(),
                volume,
                std::any::type_name::<T>(),
            )));
        }
        // SAFETY: `self.ptr` is a valid tensor pointer.
        let ptr = unsafe { self.lib.TfLiteTensorData(self.ptr) };
        if ptr.is_null() {
            return Err(Error::null_pointer("TfLiteTensorData returned null"));
        }
        // SAFETY: `ptr` is non-null and points to at least `volume * size_of::<T>()`
        // bytes (checked above). The data is valid for reads for the tensor's lifetime
        // which is tied to the interpreter borrow. `T: Copy` ensures no drop glue.
        Ok(unsafe { std::slice::from_raw_parts(ptr.cast::<T>(), volume) })
    }
}

/// Formats the tensor as `"name: 1x224x224x3 Float32"`.
impl fmt::Debug for Tensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_tensor_debug(
            f,
            self.name(),
            self.num_dims(),
            |i| self.dim(i),
            self.tensor_type(),
        )
    }
}

/// Displays the tensor as `"name: 1x224x224x3 Float32"`.
impl fmt::Display for Tensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_tensor_debug(
            f,
            self.name(),
            self.num_dims(),
            |i| self.dim(i),
            self.tensor_type(),
        )
    }
}

// ---------------------------------------------------------------------------
// TensorMut (mutable view)
// ---------------------------------------------------------------------------

/// A mutable view of a TensorFlow Lite tensor.
///
/// `TensorMut` provides all the read-only operations of [`Tensor`] plus
/// mutable data access via [`TensorMut::as_mut_slice`] and
/// [`TensorMut::copy_from_slice`].
///
/// The pointer is stored as [`NonNull`] because the C API returns
/// `*mut TfLiteTensor` for input tensors, which must be non-null after
/// successful interpreter creation.
pub struct TensorMut<'a> {
    /// Non-null pointer to the C `TfLiteTensor`.
    pub(crate) ptr: NonNull<TfLiteTensor>,

    /// Reference to the dynamically loaded `TFLite` C library.
    pub(crate) lib: &'a sys::tensorflowlite_c,
}

impl TensorMut<'_> {
    /// Returns the element data type of this tensor.
    ///
    /// If the C API returns a type value not represented by [`TensorType`],
    /// this method defaults to [`TensorType::NoType`].
    #[must_use]
    pub fn tensor_type(&self) -> TensorType {
        // SAFETY: `self.ptr` is a valid non-null tensor pointer obtained from
        // the interpreter and `self.lib` is a valid reference to the loaded library.
        let raw = unsafe { self.lib.TfLiteTensorType(self.ptr.as_ptr()) };
        FromPrimitive::from_u32(raw).unwrap_or(TensorType::NoType)
    }

    /// Returns the name of this tensor as a string slice.
    ///
    /// Returns `"<invalid-utf8>"` if the C API returns a name that is not
    /// valid UTF-8.
    #[must_use]
    pub fn name(&self) -> &str {
        // SAFETY: `self.ptr` is a valid tensor pointer; the C API returns a
        // NUL-terminated string that lives as long as the tensor.
        unsafe { CStr::from_ptr(self.lib.TfLiteTensorName(self.ptr.as_ptr())) }
            .to_str()
            .unwrap_or("<invalid-utf8>")
    }

    /// Returns the number of dimensions (rank) of this tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor does not have its dimensions set
    /// (the C API returns -1).
    pub fn num_dims(&self) -> Result<usize> {
        // SAFETY: `self.ptr` is a valid tensor pointer.
        let n = unsafe { self.lib.TfLiteTensorNumDims(self.ptr.as_ptr()) };
        usize::try_from(n).map_err(|_| {
            Error::invalid_argument(format!(
                "tensor `{}` does not have dimensions set",
                self.name()
            ))
        })
    }

    /// Returns the size of the `index`-th dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of bounds (>= `num_dims`).
    pub fn dim(&self, index: usize) -> Result<usize> {
        let num_dims = self.num_dims()?;
        if index >= num_dims {
            return Err(Error::invalid_argument(format!(
                "dimension index {index} out of bounds for tensor with {num_dims} dimensions"
            )));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let i = index as i32;
        // SAFETY: `self.ptr` is valid and `i` is bounds-checked above.
        let d = unsafe { self.lib.TfLiteTensorDim(self.ptr.as_ptr(), i) };
        // `d` is non-negative because the C API guarantees valid dimension
        // sizes for in-bounds indices.
        #[allow(clippy::cast_sign_loss)]
        Ok(d as usize)
    }

    /// Returns the full shape of this tensor as a `Vec<usize>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor dimensions are not set.
    pub fn shape(&self) -> Result<Vec<usize>> {
        let num_dims = self.num_dims()?;
        let mut dims = Vec::with_capacity(num_dims);
        for i in 0..num_dims {
            dims.push(self.dim(i)?);
        }
        Ok(dims)
    }

    /// Returns the total number of bytes required to store this tensor's data.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        // SAFETY: `self.ptr` is a valid tensor pointer.
        unsafe { self.lib.TfLiteTensorByteSize(self.ptr.as_ptr()) }
    }

    /// Returns the total number of elements in this tensor (product of all
    /// dimensions).
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor dimensions are not set.
    pub fn volume(&self) -> Result<usize> {
        Ok(self.shape()?.iter().product::<usize>())
    }

    /// Returns the affine quantization parameters for this tensor.
    #[must_use]
    pub fn quantization_params(&self) -> QuantizationParams {
        // SAFETY: `self.ptr` is a valid tensor pointer.
        let params = unsafe { self.lib.TfLiteTensorQuantizationParams(self.ptr.as_ptr()) };
        QuantizationParams {
            scale: params.scale,
            zero_point: params.zero_point,
        }
    }

    /// Returns an immutable slice over the tensor data, interpreted as
    /// elements of type `T`.
    ///
    /// The slice length equals [`TensorMut::volume`]. The caller must
    /// ensure that `T` matches the tensor's actual element type (e.g.,
    /// `f32` for a `Float32` tensor, `u8` for a `UInt8` tensor).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `size_of::<T>() * volume` exceeds [`TensorMut::byte_size`]
    /// - The underlying data pointer is null (tensor not yet allocated)
    pub fn as_slice<T: Copy>(&self) -> Result<&[T]> {
        let volume = self.volume()?;
        if std::mem::size_of::<T>() * volume > self.byte_size() {
            return Err(Error::invalid_argument(format!(
                "tensor byte size {} is too small for {} elements of {}",
                self.byte_size(),
                volume,
                std::any::type_name::<T>(),
            )));
        }
        // SAFETY: `self.ptr` is a valid tensor pointer.
        let ptr = unsafe { self.lib.TfLiteTensorData(self.ptr.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::null_pointer("TfLiteTensorData returned null"));
        }
        // SAFETY: `ptr` is non-null and points to at least `volume * size_of::<T>()`
        // bytes (checked above). The data is valid for reads for the tensor's lifetime
        // which is tied to the interpreter borrow. `T: Copy` ensures no drop glue.
        Ok(unsafe { std::slice::from_raw_parts(ptr.cast::<T>(), volume) })
    }

    /// Returns a mutable slice over the tensor data, interpreted as elements
    /// of type `T`.
    ///
    /// The slice length equals [`TensorMut::volume`]. The caller must
    /// ensure that `T` matches the tensor's actual element type (e.g.,
    /// `f32` for a `Float32` tensor, `u8` for a `UInt8` tensor).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `size_of::<T>() * volume` exceeds [`TensorMut::byte_size`]
    /// - The underlying data pointer is null (tensor not yet allocated)
    pub fn as_mut_slice<T: Copy>(&mut self) -> Result<&mut [T]> {
        let volume = self.volume()?;
        if std::mem::size_of::<T>() * volume > self.byte_size() {
            return Err(Error::invalid_argument(format!(
                "tensor byte size {} is too small for {} elements of {}",
                self.byte_size(),
                volume,
                std::any::type_name::<T>(),
            )));
        }
        // SAFETY: `self.ptr` is a valid tensor pointer.
        let ptr = unsafe { self.lib.TfLiteTensorData(self.ptr.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::null_pointer("TfLiteTensorData returned null"));
        }
        // SAFETY: `ptr` is non-null and points to at least `volume * size_of::<T>()`
        // bytes (checked above). We hold `&mut self` ensuring exclusive access.
        // `T: Copy` ensures no drop glue.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr.cast::<T>(), volume) })
    }

    /// Copies the contents of `data` into this tensor's buffer.
    ///
    /// This is a convenience wrapper around [`TensorMut::as_mut_slice`] that
    /// copies elements from the provided slice into the tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor cannot be mapped as a mutable slice of `T`
    /// - `data.len()` does not match [`TensorMut::volume`]
    pub fn copy_from_slice<T: Copy>(&mut self, data: &[T]) -> Result<()> {
        let slice = self.as_mut_slice::<T>()?;
        if data.len() != slice.len() {
            return Err(Error::invalid_argument(format!(
                "data length {} does not match tensor volume {}",
                data.len(),
                slice.len(),
            )));
        }
        slice.copy_from_slice(data);
        Ok(())
    }
}

/// Formats the tensor as `"name: 1x224x224x3 Float32"`.
impl fmt::Debug for TensorMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_tensor_debug(
            f,
            self.name(),
            self.num_dims(),
            |i| self.dim(i),
            self.tensor_type(),
        )
    }
}

/// Displays the tensor as `"name: 1x224x224x3 Float32"`.
impl fmt::Display for TensorMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_tensor_debug(
            f,
            self.name(),
            self.num_dims(),
            |i| self.dim(i),
            self.tensor_type(),
        )
    }
}

// ---------------------------------------------------------------------------
// Shared formatting helper
// ---------------------------------------------------------------------------

/// Writes the common tensor representation: `"name: 1x224x224x3 Float32"`.
///
/// Used by both `Tensor` and `TensorMut` `Debug` and `Display` implementations
/// to avoid code duplication.
fn write_tensor_debug(
    f: &mut fmt::Formatter<'_>,
    name: &str,
    num_dims: Result<usize>,
    dim_fn: impl Fn(usize) -> Result<usize>,
    tensor_type: TensorType,
) -> fmt::Result {
    let num_dims = num_dims.unwrap_or(0);
    write!(f, "{name}: ")?;
    for i in 0..num_dims {
        if i > 0 {
            f.write_str("x")?;
        }
        write!(f, "{}", dim_fn(i).unwrap_or(0))?;
    }
    write!(f, " {tensor_type:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    // -----------------------------------------------------------------------
    // TensorType -- FromPrimitive conversion
    // -----------------------------------------------------------------------

    #[test]
    fn tensor_type_from_primitive_all_variants() {
        let cases: &[(isize, TensorType)] = &[
            (0, TensorType::NoType),
            (1, TensorType::Float32),
            (2, TensorType::Int32),
            (3, TensorType::UInt8),
            (4, TensorType::Int64),
            (5, TensorType::String),
            (6, TensorType::Bool),
            (7, TensorType::Int16),
            (8, TensorType::Complex64),
            (9, TensorType::Int8),
            (10, TensorType::Float16),
            (11, TensorType::Float64),
            (12, TensorType::Complex128),
            (13, TensorType::UInt64),
            (14, TensorType::Resource),
            (15, TensorType::Variant),
            (16, TensorType::UInt32),
            (17, TensorType::UInt16),
            (18, TensorType::Int4),
            (19, TensorType::BFloat16),
        ];

        for &(raw, expected) in cases {
            let result = TensorType::from_isize(raw);
            assert_eq!(
                result,
                Some(expected),
                "TensorType::from_isize({raw}) should be Some({expected:?})"
            );
        }
    }

    #[test]
    fn tensor_type_from_u32_all_variants() {
        for raw in 0u32..=19 {
            let result = TensorType::from_u32(raw);
            assert!(
                result.is_some(),
                "TensorType::from_u32({raw}) should be Some"
            );
        }
    }

    #[test]
    fn tensor_type_unknown_value_returns_none() {
        assert_eq!(TensorType::from_isize(999), None);
        assert_eq!(TensorType::from_u32(999), None);
        assert_eq!(TensorType::from_isize(-1), None);
        assert_eq!(TensorType::from_isize(20), None);
    }

    // -----------------------------------------------------------------------
    // TensorType -- Clone, PartialEq, Hash
    // -----------------------------------------------------------------------

    #[test]
    fn tensor_type_clone() {
        let original = TensorType::Float32;
        let cloned = original;
        assert_eq!(original, cloned);
    }

    #[test]
    fn tensor_type_partial_eq() {
        assert_eq!(TensorType::Int8, TensorType::Int8);
        assert_ne!(TensorType::Int8, TensorType::UInt8);
    }

    #[test]
    fn tensor_type_hash() {
        let mut set = HashSet::new();
        set.insert(TensorType::Float32);
        set.insert(TensorType::Float32);
        set.insert(TensorType::Int32);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn tensor_type_all_variants_unique_in_hashset() {
        let all = [
            TensorType::NoType,
            TensorType::Float32,
            TensorType::Int32,
            TensorType::UInt8,
            TensorType::Int64,
            TensorType::String,
            TensorType::Bool,
            TensorType::Int16,
            TensorType::Complex64,
            TensorType::Int8,
            TensorType::Float16,
            TensorType::Float64,
            TensorType::Complex128,
            TensorType::UInt64,
            TensorType::Resource,
            TensorType::Variant,
            TensorType::UInt32,
            TensorType::UInt16,
            TensorType::Int4,
            TensorType::BFloat16,
        ];
        let set: HashSet<_> = all.iter().copied().collect();
        assert_eq!(set.len(), 20);
    }

    // -----------------------------------------------------------------------
    // TensorType -- Debug formatting
    // -----------------------------------------------------------------------

    #[test]
    fn tensor_type_debug_format() {
        assert_eq!(format!("{:?}", TensorType::Float32), "Float32");
        assert_eq!(format!("{:?}", TensorType::NoType), "NoType");
        assert_eq!(format!("{:?}", TensorType::BFloat16), "BFloat16");
        assert_eq!(format!("{:?}", TensorType::Complex128), "Complex128");
    }

    // -----------------------------------------------------------------------
    // QuantizationParams -- construction and field access
    // -----------------------------------------------------------------------

    #[test]
    fn quantization_params_construction() {
        let params = QuantizationParams {
            scale: 0.5,
            zero_point: 128,
        };
        assert!((params.scale - 0.5).abs() < f32::EPSILON);
        assert_eq!(params.zero_point, 128);
    }

    #[test]
    fn quantization_params_zero_values() {
        let params = QuantizationParams {
            scale: 0.0,
            zero_point: 0,
        };
        assert!((params.scale - 0.0).abs() < f32::EPSILON);
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn quantization_params_negative_zero_point() {
        let params = QuantizationParams {
            scale: 0.007_812_5,
            zero_point: -128,
        };
        assert!((params.scale - 0.007_812_5).abs() < f32::EPSILON);
        assert_eq!(params.zero_point, -128);
    }

    // -----------------------------------------------------------------------
    // QuantizationParams -- Debug, Clone, PartialEq
    // -----------------------------------------------------------------------

    #[test]
    fn quantization_params_debug() {
        let params = QuantizationParams {
            scale: 1.0,
            zero_point: 0,
        };
        let debug = format!("{params:?}");
        assert!(debug.contains("QuantizationParams"));
        assert!(debug.contains("scale"));
        assert!(debug.contains("zero_point"));
    }

    #[test]
    fn quantization_params_clone() {
        let original = QuantizationParams {
            scale: 0.25,
            zero_point: 64,
        };
        let cloned = original;
        assert_eq!(original, cloned);
    }

    #[test]
    fn quantization_params_partial_eq() {
        let a = QuantizationParams {
            scale: 0.5,
            zero_point: 128,
        };
        let b = QuantizationParams {
            scale: 0.5,
            zero_point: 128,
        };
        let c = QuantizationParams {
            scale: 0.25,
            zero_point: 128,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
