// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Hand-written mirrors of `TFLite` structs that custom delegate kernels read
//! but that no bound C function names, so bindgen does not emit them.
//!
//! These are plain C structs read through pointers `TFLite` hands to delegate
//! callbacks. They contain no function symbols, so they add nothing to the
//! runtime-resolved symbol table. Layouts follow TensorFlow Lite 2.19:
//! `TfLiteOpaqueDelegateParams` and `TfLiteAffineQuantization` from
//! `tensorflow/lite/core/c/common.h`, the operator parameter structs from
//! `tensorflow/compiler/mlir/lite/core/c/builtin_op_data.h`. The size and
//! offset assertions below were checked against those headers with clang on
//! `x86_64` and `aarch64`.

use std::ffi::{c_int, c_void};

use crate::{TfLiteFloatArray, TfLiteIntArray, TfLiteOpaqueDelegate};

/// Parameters `TFLite` passes to a delegate kernel's `init` callback.
///
/// Mirrors `TfLiteOpaqueDelegateParams`. All arrays are owned by the runtime
/// and valid for the duration of the `init` call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TfLiteOpaqueDelegateParams {
    /// The delegate that claimed the partition.
    pub delegate: *mut TfLiteOpaqueDelegate,
    /// The `data` field of the delegate's `TfLiteOpaqueDelegateBuilder`.
    pub delegate_data: *mut c_void,
    /// Node indices replaced by this partition, in execution order.
    pub nodes_to_replace: *mut TfLiteIntArray,
    /// Tensor indices the partition reads from outside itself.
    pub input_tensors: *mut TfLiteIntArray,
    /// Tensor indices the partition produces for consumers outside itself.
    pub output_tensors: *mut TfLiteIntArray,
}

/// Affine quantization parameters, per tensor or per axis.
///
/// Mirrors `TfLiteAffineQuantization`. Pointed to by `TfLiteQuantization::params`
/// when its type is `kTfLiteAffineQuantization`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TfLiteAffineQuantization {
    /// One scale per tensor, or one per slice along `quantized_dimension`.
    pub scale: *mut TfLiteFloatArray,
    /// Zero points, matching `scale` in length.
    pub zero_point: *mut TfLiteIntArray,
    /// Axis the per-axis parameters apply to.
    pub quantized_dimension: i32,
}

/// Fused activation applied by an operator (`TfLiteFusedActivation`).
pub type TfLiteFusedActivation = c_int;

/// No fused activation (`kTfLiteActNone`).
pub const K_TFLITE_ACT_NONE: TfLiteFusedActivation = 0;

/// Builtin data of `SOFTMAX` (`TfLiteSoftmaxParams`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TfLiteSoftmaxParams {
    /// Inverse temperature applied to the logits.
    pub beta: f32,
}

/// Builtin data of `ADD` (`TfLiteAddParams`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TfLiteAddParams {
    /// Fused activation.
    pub activation: TfLiteFusedActivation,
    /// Power-of-two scales for `int16` (operator version 4+).
    pub pot_scale_int16: bool,
}

/// Builtin data of `SUB` (`TfLiteSubParams`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TfLiteSubParams {
    /// Fused activation.
    pub activation: TfLiteFusedActivation,
    /// Power-of-two scales for `int16` (operator version 5+).
    pub pot_scale_int16: bool,
}

/// Builtin data of `MUL` (`TfLiteMulParams`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TfLiteMulParams {
    /// Fused activation.
    pub activation: TfLiteFusedActivation,
}

/// Builtin data of `BATCH_MATMUL` (`TfLiteBatchMatMulParams`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // Mirrors the C struct field for field.
pub struct TfLiteBatchMatMulParams {
    /// Transpose the last two dimensions of the left operand.
    pub adj_x: bool,
    /// Transpose the last two dimensions of the right operand.
    pub adj_y: bool,
    /// Quantize non-constant inputs at run time (operator version 4+).
    pub asymmetric_quantize_inputs: bool,
}

/// Builtin data of `CONCATENATION` (`TfLiteConcatenationParams`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TfLiteConcatenationParams {
    /// Axis to concatenate along; negative values count from the end.
    pub axis: c_int,
    /// Fused activation.
    pub activation: TfLiteFusedActivation,
}

/// Padding of a convolution (`TfLitePadding`).
pub type TfLitePadding = c_int;

/// Builtin data of `CONV_2D` (`TfLiteConvParams`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TfLiteConvParams {
    /// `kTfLitePaddingUnknown` (0), `kTfLitePaddingSame` (1) or `kTfLitePaddingValid` (2).
    pub padding: TfLitePadding,
    pub stride_width: c_int,
    pub stride_height: c_int,
    /// Fused activation.
    pub activation: TfLiteFusedActivation,
    /// Dilation (operator version 2+).
    pub dilation_width_factor: c_int,
    pub dilation_height_factor: c_int,
    /// Default type of a quantized bias (operator version 7+).
    pub quantized_bias_type: crate::TfLiteType,
}

const _: () = {
    use std::mem::offset_of;
    assert!(size_of::<TfLiteOpaqueDelegateParams>() == 5 * size_of::<*const c_void>());
    assert!(
        offset_of!(TfLiteAffineQuantization, quantized_dimension) == 2 * size_of::<*const c_void>()
    );
    assert!(size_of::<TfLiteSoftmaxParams>() == 4 && align_of::<TfLiteSoftmaxParams>() == 4);
    assert!(size_of::<TfLiteAddParams>() == 8 && offset_of!(TfLiteAddParams, pot_scale_int16) == 4);
    assert!(size_of::<TfLiteSubParams>() == 8 && offset_of!(TfLiteSubParams, pot_scale_int16) == 4);
    assert!(size_of::<TfLiteMulParams>() == 4);
    assert!(
        size_of::<TfLiteBatchMatMulParams>() == 3 && align_of::<TfLiteBatchMatMulParams>() == 1
    );
    assert!(offset_of!(TfLiteBatchMatMulParams, adj_y) == 1);
    assert!(offset_of!(TfLiteBatchMatMulParams, asymmetric_quantize_inputs) == 2);
    assert!(size_of::<TfLiteConcatenationParams>() == 8);
    assert!(offset_of!(TfLiteConcatenationParams, activation) == 4);
    assert!(size_of::<TfLiteConvParams>() == 28 && align_of::<TfLiteConvParams>() == 4);
    assert!(offset_of!(TfLiteConvParams, activation) == 12);
    assert!(offset_of!(TfLiteConvParams, quantized_bias_type) == 24);
};
