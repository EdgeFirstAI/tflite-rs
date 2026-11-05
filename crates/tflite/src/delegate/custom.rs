// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! In-process custom delegates written in Rust.
//!
//! A custom delegate claims nodes of a model's graph and runs them with its
//! own kernels instead of the builtin `TFLite` CPU kernels. It is built on the
//! ABI-stable opaque delegate API (`TfLiteOpaqueDelegateCreate`,
//! `TfLiteOperator`), so it works with any `TFLite` runtime that exports it
//! (2.17 and later).
//!
//! # How it fits together
//!
//! 1. Implement [`CustomDelegate`] and wrap it with [`Delegate::from_custom`].
//!    Add it to an [`InterpreterBuilder`](crate::InterpreterBuilder) like any
//!    other delegate. Delegates are applied in the order they are added, and
//!    each one only sees the nodes earlier ones left unclaimed.
//! 2. When the interpreter is built, [`CustomDelegate::select_nodes`] inspects
//!    the graph through a [`GraphView`] and returns the nodes to claim.
//! 3. `TFLite` groups connected claimed nodes into partitions. For each one it
//!    calls [`CustomDelegate::create_kernel`] with a [`Partition`].
//! 4. The resulting [`PartitionKernel`] is prepared when tensors are
//!    allocated and invoked on every inference, reading and writing the
//!    partition's boundary tensors through [`KernelIo`].
//!
//! A delegate must run every node it claims; there is no fallback to the
//! builtin kernel. Claim only nodes whose types, quantization and parameters
//! the kernels implement exactly.
//!
//! Tensors produced and consumed inside a partition are not allocated by
//! `TFLite`; kernels hold their own scratch buffers for those. Their metadata
//! (shape, quantization) is available through [`Partition::graph`] when the
//! kernel is created.
//!
//! # Errors and panics
//!
//! Errors returned by the trait methods are reported through the `TFLite`
//! error reporter and surface as a failed interpreter build or a failed
//! invoke. Panics are caught at the C boundary and reported the same way;
//! they never unwind into `TFLite`.
//!
//! # Example
//!
//! ```no_run
//! use std::ffi::CStr;
//! use edgefirst_tflite::delegate::custom::{
//!     BoxError, BuiltinOperator, CustomDelegate, GraphView, KernelIo, NodeId, Partition,
//!     PartitionKernel,
//! };
//! use edgefirst_tflite::{Delegate, Interpreter, Library, Model};
//!
//! struct Relu6Delegate;
//! struct Relu6Kernel;
//!
//! impl CustomDelegate for Relu6Delegate {
//!     type Kernel = Relu6Kernel;
//!
//!     fn name(&self) -> &CStr {
//!         c"Relu6Delegate"
//!     }
//!
//!     fn select_nodes(&self, graph: &GraphView<'_>) -> Result<Vec<NodeId>, BoxError> {
//!         let mut claimed = Vec::new();
//!         for id in graph.execution_plan()? {
//!             if graph.node(id)?.builtin_op() == BuiltinOperator::RELU6 {
//!                 claimed.push(id);
//!             }
//!         }
//!         Ok(claimed)
//!     }
//!
//!     fn create_kernel(&self, _partition: &Partition<'_>) -> Result<Relu6Kernel, BoxError> {
//!         Ok(Relu6Kernel)
//!     }
//! }
//!
//! impl PartitionKernel for Relu6Kernel {
//!     fn invoke(&mut self, io: &mut KernelIo<'_>) -> Result<(), BoxError> {
//!         let input = io.input(0)?;
//!         let mut output = io.output(0)?;
//!         for (y, x) in output.as_mut_slice::<f32>()?.iter_mut().zip(input.as_slice::<f32>()?) {
//!             *y = x.clamp(0.0, 6.0);
//!         }
//!         Ok(())
//!     }
//! }
//!
//! let lib = Library::new()?;
//! let model = Model::from_file(&lib, "model.tflite")?;
//! let mut interpreter = Interpreter::builder(&lib)?
//!     .delegate(Delegate::from_custom(&lib, Relu6Delegate)?)
//!     .build(&model)?;
//! interpreter.invoke()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::any::Any;
use std::cell::Cell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::{self, NonNull};

use edgefirst_tflite_sys as sys;
use edgefirst_tflite_sys::kernel_ffi::{TfLiteAffineQuantization, TfLiteOpaqueDelegateParams};
use num_traits::FromPrimitive;

pub use edgefirst_tflite_sys::kernel_ffi::{
    TfLiteAddParams as AddParams, TfLiteBatchMatMulParams as BatchMatMulParams,
    TfLiteConcatenationParams as ConcatenationParams, TfLiteConvParams as ConvParams,
    TfLiteMulParams as MulParams, TfLiteSoftmaxParams as SoftmaxParams,
    TfLiteSubParams as SubParams,
};

use super::Delegate;
use crate::error::{status_to_result, Error, Result};
use crate::tensor::{check_element_type, QuantizationParams, TensorType};
use crate::Library;

/// Error type returned by custom delegate and kernel callbacks.
///
/// Any error type converts into it with `?`, including [`crate::Error`].
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// A delegate implemented in Rust.
///
/// One value is shared by every partition of every graph the delegate is
/// applied to, so it is `Sync`. Wrap it with [`Delegate::from_custom`].
pub trait CustomDelegate: Send + Sync + 'static {
    /// The kernel that runs one claimed partition.
    type Kernel: PartitionKernel;

    /// Name of the delegate's partition kernels, used in error messages.
    ///
    /// The `TFLite` telemetry profiler does not read it: up to at least
    /// `TFLite` 2.19 it labels every opaque-delegate partition `DELEGATE`.
    fn name(&self) -> &CStr;

    /// Choose the nodes to claim.
    ///
    /// Called while the interpreter is built, and again whenever `TFLite`
    /// re-applies delegates (for example after an input is resized). Nodes
    /// already claimed by an earlier delegate appear as
    /// [`BuiltinOperator::DELEGATE`] nodes.
    ///
    /// Every returned id must come from this graph's execution plan, at most once;
    /// otherwise the delegate is not applied and the build reports the error.
    fn select_nodes(&self, graph: &GraphView<'_>) -> std::result::Result<Vec<NodeId>, BoxError>;

    /// Create the kernel for one partition of connected claimed nodes.
    fn create_kernel(
        &self,
        partition: &Partition<'_>,
    ) -> std::result::Result<Self::Kernel, BoxError>;
}

/// Runs one partition claimed by a [`CustomDelegate`].
///
/// All allocation belongs in [`CustomDelegate::create_kernel`] or
/// [`PartitionKernel::prepare`]; [`PartitionKernel::invoke`] runs on every
/// inference.
pub trait PartitionKernel: Send + 'static {
    /// Validate shapes and prepare for inference.
    ///
    /// Called when tensors are (re)allocated. Tensor metadata is available
    /// but, apart from constants, tensor data is not.
    fn prepare(&mut self, io: &mut KernelIo<'_>) -> std::result::Result<(), BoxError> {
        let _ = io;
        Ok(())
    }

    /// Run the partition: read its inputs and write all of its outputs.
    fn invoke(&mut self, io: &mut KernelIo<'_>) -> std::result::Result<(), BoxError>;
}

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Index of a node in the graph's execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(c_int);

impl NodeId {
    fn from_raw(raw: c_int) -> Option<Self> {
        (raw >= 0).then_some(Self(raw))
    }

    /// The node index as `usize`.
    #[must_use]
    pub fn index(self) -> usize {
        // Constructed only from non-negative values.
        self.0.unsigned_abs() as usize
    }
}

/// Index of a tensor in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(c_int);

impl TensorId {
    fn from_raw(raw: c_int) -> Option<Self> {
        (raw >= 0).then_some(Self(raw))
    }

    /// The tensor index as `usize`.
    #[must_use]
    pub fn index(self) -> usize {
        // Constructed only from non-negative values.
        self.0.unsigned_abs() as usize
    }
}

/// A `TFLite` builtin operator code.
///
/// Common codes are provided as constants; any other code can be built from
/// its raw value with [`BuiltinOperator::from_raw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuiltinOperator(u32);

impl BuiltinOperator {
    /// `ADD`
    pub const ADD: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinAdd);
    /// `BATCH_MATMUL`
    pub const BATCH_MATMUL: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinBatchMatmul);
    /// `CONCATENATION`
    pub const CONCATENATION: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinConcatenation);
    /// `CONV_2D`
    pub const CONV_2D: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinConv2d);
    /// `CUSTOM` (a custom operator; see [`NodeView::custom_name`])
    pub const CUSTOM: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinCustom);
    /// `DELEGATE` (a partition already claimed by a delegate)
    pub const DELEGATE: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinDelegate);
    /// `DEPTHWISE_CONV_2D`
    pub const DEPTHWISE_CONV_2D: Self =
        Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinDepthwiseConv2d);
    /// `DEQUANTIZE`
    pub const DEQUANTIZE: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinDequantize);
    /// `FULLY_CONNECTED`
    pub const FULLY_CONNECTED: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinFullyConnected);
    /// `LOGISTIC`
    pub const LOGISTIC: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinLogistic);
    /// `MUL`
    pub const MUL: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinMul);
    /// `QUANTIZE`
    pub const QUANTIZE: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinQuantize);
    /// `RELU6`
    pub const RELU6: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinRelu6);
    /// `RESHAPE`
    pub const RESHAPE: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinReshape);
    /// `SLICE`
    pub const SLICE: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinSlice);
    /// `SOFTMAX`
    pub const SOFTMAX: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinSoftmax);
    /// `STRIDED_SLICE`
    pub const STRIDED_SLICE: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinStridedSlice);
    /// `SUB`
    pub const SUB: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinSub);
    /// `TRANSPOSE`
    pub const TRANSPOSE: Self = Self(sys::TfLiteBuiltinOperator_kTfLiteBuiltinTranspose);

    /// Wrap a raw `TfLiteBuiltinOperator` code.
    #[must_use]
    pub const fn from_raw(code: u32) -> Self {
        Self(code)
    }

    /// The raw `TfLiteBuiltinOperator` code.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Builtin parameters
// ---------------------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
    pub trait Element {}
}

/// Element types a kernel can read and write tensor data as: plain integers and floats, for
/// which every bit pattern is a valid value.
///
/// Sealed: [`TensorInfo::constant_data`], [`KernelTensor::as_slice`] and
/// [`KernelTensorMut::as_mut_slice`] reinterpret tensor memory as `[T]`, which would be
/// undefined for types such as `bool` or `char` with invalid bit patterns.
///
/// ```compile_fail
/// fn read(t: &edgefirst_tflite::delegate::custom::KernelTensor<'_>) {
///     let _ = t.as_slice::<bool>();
/// }
/// ```
pub trait TensorElement: sealed::Element + Copy {}

macro_rules! tensor_elements {
    ($($ty:ty),+ $(,)?) => {$(
        impl sealed::Element for $ty {}
        impl TensorElement for $ty {}
    )+};
}

tensor_elements!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

/// Builtin parameter structs readable through [`NodeView::params`].
///
/// Sealed: implemented only for the parameter types this module exports,
/// whose layouts are checked against the `TFLite` headers.
pub trait BuiltinParams: sealed::Sealed + Copy {
    /// Operators whose builtin data has this layout.
    const OPS: &'static [BuiltinOperator];
}

macro_rules! builtin_params {
    ($($ty:ty => [$($op:ident),+]),+ $(,)?) => {$(
        impl sealed::Sealed for $ty {}
        impl BuiltinParams for $ty {
            const OPS: &'static [BuiltinOperator] = &[$(BuiltinOperator::$op),+];
        }
    )+};
}

builtin_params! {
    SoftmaxParams => [SOFTMAX],
    AddParams => [ADD],
    SubParams => [SUB],
    MulParams => [MUL],
    BatchMatMulParams => [BATCH_MATMUL],
    ConcatenationParams => [CONCATENATION],
    ConvParams => [CONV_2D],
}

// ---------------------------------------------------------------------------
// Graph inspection
// ---------------------------------------------------------------------------

/// A read-only view of the graph a delegate is being applied to.
///
/// Valid only for the duration of the callback it is passed to.
pub struct GraphView<'a> {
    fns: &'a sys::tensorflowlite_c,
    context: *mut sys::TfLiteOpaqueContext,
}

impl<'a> GraphView<'a> {
    fn new(fns: &'a sys::tensorflowlite_c, context: *mut sys::TfLiteOpaqueContext) -> Self {
        Self { fns, context }
    }

    /// Node indices in execution order.
    pub fn execution_plan(&self) -> Result<Vec<NodeId>> {
        let mut plan: *mut sys::TfLiteIntArray = ptr::null_mut();
        // SAFETY: `context` is the live context passed to the current
        // callback; `plan` receives a runtime-owned array.
        let status = unsafe {
            self.fns
                .TfLiteOpaqueContextGetExecutionPlan(self.context, &raw mut plan)
        };
        status_to_result(status)
            .map_err(|e| e.with_context("TfLiteOpaqueContextGetExecutionPlan"))?;
        // SAFETY: on success `plan` points to a valid runtime-owned array
        // that outlives this call.
        let raw = unsafe { int_array(plan) };
        Ok(raw.iter().filter_map(|&n| NodeId::from_raw(n)).collect())
    }

    /// The node at `id`.
    pub fn node(&self, id: NodeId) -> Result<NodeView<'a>> {
        let mut node: *mut sys::TfLiteOpaqueNode = ptr::null_mut();
        let mut registration: *mut sys::TfLiteOperator = ptr::null_mut();
        // SAFETY: `context` is live; the out-pointers are valid locals.
        let status = unsafe {
            self.fns.TfLiteOpaqueContextGetNodeAndRegistration(
                self.context,
                id.0,
                &raw mut node,
                &raw mut registration,
            )
        };
        status_to_result(status).map_err(|e| {
            e.with_context(format!(
                "TfLiteOpaqueContextGetNodeAndRegistration({})",
                id.0
            ))
        })?;
        if node.is_null() || registration.is_null() {
            return Err(Error::null_pointer(format!(
                "node {} has no registration",
                id.0
            )));
        }
        Ok(NodeView {
            id,
            fns: self.fns,
            node,
            registration,
            _graph: PhantomData,
        })
    }

    /// The tensor at `id`.
    pub fn tensor(&self, id: TensorId) -> Result<TensorInfo<'a>> {
        if id.index() >= self.num_tensors() {
            return Err(Error::invalid_argument(format!(
                "tensor {} out of range ({} tensors)",
                id.0,
                self.num_tensors()
            )));
        }
        // SAFETY: `context` is live and `id` is in range.
        let tensor = unsafe {
            self.fns
                .TfLiteOpaqueContextGetOpaqueTensor(self.context, id.0)
        };
        TensorInfo::new(self.fns, tensor, id)
    }

    /// Number of tensors in the graph.
    #[must_use]
    pub fn num_tensors(&self) -> usize {
        // SAFETY: `context` is live.
        unsafe { self.fns.TfLiteOpaqueContextGetNumTensors(self.context) }
    }

    /// The graph's input tensors.
    pub fn graph_inputs(&self) -> Result<Vec<TensorId>> {
        let mut ids: *const c_int = ptr::null();
        let mut n: c_int = 0;
        // SAFETY: `context` is live; the out-pointers are valid locals.
        let status = unsafe {
            self.fns
                .TfLiteOpaqueContextGetInputs(self.context, &raw mut ids, &raw mut n)
        };
        status_to_result(status).map_err(|e| e.with_context("TfLiteOpaqueContextGetInputs"))?;
        // SAFETY: on success `ids` points to `n` runtime-owned indices.
        Ok(unsafe { tensor_ids(ids, n) })
    }

    /// The graph's output tensors.
    pub fn graph_outputs(&self) -> Result<Vec<TensorId>> {
        let mut ids: *const c_int = ptr::null();
        let mut n: c_int = 0;
        // SAFETY: `context` is live; the out-pointers are valid locals.
        let status = unsafe {
            self.fns
                .TfLiteOpaqueContextGetOutputs(self.context, &raw mut ids, &raw mut n)
        };
        status_to_result(status).map_err(|e| e.with_context("TfLiteOpaqueContextGetOutputs"))?;
        // SAFETY: on success `ids` points to `n` runtime-owned indices.
        Ok(unsafe { tensor_ids(ids, n) })
    }

    /// Nodes in the execution plan that read `tensor`.
    pub fn consumers(&self, tensor: TensorId) -> Result<Vec<NodeId>> {
        let mut out = Vec::new();
        for id in self.execution_plan()? {
            if self.node(id)?.inputs().contains(&Some(tensor)) {
                out.push(id);
            }
        }
        Ok(out)
    }
}

impl fmt::Debug for GraphView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphView")
            .field("num_tensors", &self.num_tensors())
            .finish_non_exhaustive()
    }
}

/// One node of a [`GraphView`].
pub struct NodeView<'a> {
    id: NodeId,
    fns: &'a sys::tensorflowlite_c,
    node: *mut sys::TfLiteOpaqueNode,
    registration: *mut sys::TfLiteOperator,
    _graph: PhantomData<&'a GraphView<'a>>,
}

impl<'a> NodeView<'a> {
    /// The node's index.
    #[must_use]
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// The node's builtin operator.
    #[must_use]
    pub fn builtin_op(&self) -> BuiltinOperator {
        // SAFETY: `registration` was returned for this node by the runtime.
        BuiltinOperator(unsafe { self.fns.TfLiteOperatorGetBuiltInCode(self.registration) })
    }

    /// The custom operator name, for [`BuiltinOperator::CUSTOM`] and
    /// [`BuiltinOperator::DELEGATE`] nodes.
    #[must_use]
    pub fn custom_name(&self) -> Option<&'a str> {
        // SAFETY: `registration` is valid; the name is runtime-owned and
        // outlives the graph view.
        let name = unsafe { self.fns.TfLiteOperatorGetCustomName(self.registration) };
        if name.is_null() {
            return None;
        }
        // SAFETY: non-null names are NUL-terminated C strings.
        unsafe { CStr::from_ptr(name) }.to_str().ok()
    }

    /// The operator version.
    #[must_use]
    pub fn version(&self) -> i32 {
        // SAFETY: `registration` is valid.
        unsafe { self.fns.TfLiteOperatorGetVersion(self.registration) }
    }

    /// Input tensors; `None` marks an omitted optional input.
    #[must_use]
    pub fn inputs(&self) -> Vec<Option<TensorId>> {
        let mut ids: *const c_int = ptr::null();
        let mut n: c_int = 0;
        // SAFETY: `node` is valid; the out-pointers are valid locals.
        let status = unsafe {
            self.fns
                .TfLiteOpaqueNodeInputs(self.node, &raw mut ids, &raw mut n)
        };
        if status != sys::TfLiteStatus_kTfLiteOk {
            return Vec::new();
        }
        // SAFETY: on success `ids` points to `n` runtime-owned indices.
        unsafe { raw_ids(ids, n) }
            .iter()
            .map(|&t| TensorId::from_raw(t))
            .collect()
    }

    /// Output tensors; `None` marks an omitted optional output.
    #[must_use]
    pub fn outputs(&self) -> Vec<Option<TensorId>> {
        let mut ids: *const c_int = ptr::null();
        let mut n: c_int = 0;
        // SAFETY: `node` is valid; the out-pointers are valid locals.
        let status = unsafe {
            self.fns
                .TfLiteOpaqueNodeOutputs(self.node, &raw mut ids, &raw mut n)
        };
        if status != sys::TfLiteStatus_kTfLiteOk {
            return Vec::new();
        }
        // SAFETY: on success `ids` points to `n` runtime-owned indices.
        unsafe { raw_ids(ids, n) }
            .iter()
            .map(|&t| TensorId::from_raw(t))
            .collect()
    }

    /// The node's builtin parameters, if its operator uses layout `P`.
    ///
    /// Returns `None` when the operator is not one of `P::OPS` or the node
    /// has no builtin data.
    #[must_use]
    pub fn params<P: BuiltinParams>(&self) -> Option<P> {
        if !P::OPS.contains(&self.builtin_op()) {
            return None;
        }
        // SAFETY: `node` is valid.
        let data = unsafe { self.fns.TfLiteOpaqueNodeGetBuiltinData(self.node) };
        if data.is_null() {
            return None;
        }
        // SAFETY: the runtime stores a `P`-layout struct as the builtin data
        // of every operator in `P::OPS` (checked above). `read_unaligned`
        // makes no assumption about the allocation's alignment.
        Some(unsafe { data.cast::<P>().read_unaligned() })
    }
}

impl fmt::Debug for NodeView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeView")
            .field("id", &self.id)
            .field("builtin_op", &self.builtin_op())
            .field("custom_name", &self.custom_name())
            .finish_non_exhaustive()
    }
}

/// Quantization of a tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quantization<'a> {
    /// The tensor is not quantized.
    None,
    /// One scale and zero point for the whole tensor.
    PerTensor(QuantizationParams),
    /// One scale and zero point per slice along `axis`.
    PerAxis {
        /// Scales, one per slice.
        scales: &'a [f32],
        /// Zero points, one per slice.
        zero_points: &'a [i32],
        /// Axis the parameters apply to.
        axis: usize,
    },
}

/// Metadata of one tensor in a [`GraphView`].
pub struct TensorInfo<'a> {
    id: TensorId,
    fns: &'a sys::tensorflowlite_c,
    tensor: *const sys::TfLiteOpaqueTensor,
}

impl<'a> TensorInfo<'a> {
    fn new(
        fns: &'a sys::tensorflowlite_c,
        tensor: *const sys::TfLiteOpaqueTensor,
        id: TensorId,
    ) -> Result<Self> {
        if tensor.is_null() {
            return Err(Error::null_pointer(format!("tensor {} is null", id.0)));
        }
        Ok(Self { id, fns, tensor })
    }

    /// The tensor's index in the graph.
    #[must_use]
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// The tensor name, or `""` if it has none.
    #[must_use]
    pub fn name(&self) -> &'a str {
        // SAFETY: `tensor` is valid; the name is runtime-owned.
        let name = unsafe { self.fns.TfLiteOpaqueTensorName(self.tensor) };
        if name.is_null() {
            return "";
        }
        // SAFETY: non-null names are NUL-terminated C strings.
        unsafe { CStr::from_ptr(name) }
            .to_str()
            .unwrap_or("<invalid-utf8>")
    }

    /// The element type.
    #[must_use]
    pub fn tensor_type(&self) -> TensorType {
        // SAFETY: `tensor` is valid.
        let raw = unsafe { self.fns.TfLiteOpaqueTensorType(self.tensor) };
        FromPrimitive::from_u32(raw).unwrap_or(TensorType::NoType)
    }

    /// Number of dimensions.
    #[must_use]
    pub fn num_dims(&self) -> usize {
        // SAFETY: `tensor` is valid.
        let n = unsafe { self.fns.TfLiteOpaqueTensorNumDims(self.tensor) };
        usize::try_from(n).unwrap_or(0)
    }

    /// Size of dimension `index`.
    pub fn dim(&self, index: usize) -> Result<usize> {
        let n = self.num_dims();
        if index >= n {
            return Err(Error::invalid_argument(format!(
                "dimension {index} out of range for a {n}-d tensor"
            )));
        }
        let index = i32::try_from(index)
            .map_err(|_| Error::invalid_argument(format!("dimension {index} out of range")))?;
        // SAFETY: `tensor` is valid and `index` is below `num_dims`.
        let d = unsafe { self.fns.TfLiteOpaqueTensorDim(self.tensor, index) };
        usize::try_from(d)
            .map_err(|_| Error::invalid_argument(format!("dimension {index} is unknown ({d})")))
    }

    /// All dimensions.
    pub fn shape(&self) -> Result<Vec<usize>> {
        (0..self.num_dims()).map(|i| self.dim(i)).collect()
    }

    /// Number of elements.
    pub fn volume(&self) -> Result<usize> {
        (0..self.num_dims()).try_fold(1usize, |acc, i| Ok(acc * self.dim(i)?))
    }

    /// Size of the tensor's data in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        // SAFETY: `tensor` is valid.
        unsafe { self.fns.TfLiteOpaqueTensorByteSize(self.tensor) }
    }

    /// The tensor's quantization.
    #[must_use]
    pub fn quantization(&self) -> Quantization<'a> {
        // SAFETY: `tensor` is valid.
        let q = unsafe { self.fns.TfLiteOpaqueTensorGetQuantization(self.tensor) };
        if q.type_ != sys::TfLiteQuantizationType_kTfLiteAffineQuantization || q.params.is_null() {
            return Quantization::None;
        }
        // SAFETY: affine quantization params are a `TfLiteAffineQuantization`
        // owned by the tensor.
        let affine = unsafe { &*q.params.cast::<TfLiteAffineQuantization>() };
        // SAFETY: the arrays are runtime-owned and live as long as the tensor.
        let scales = unsafe { float_array(affine.scale) };
        // SAFETY: as above.
        let zero_points = unsafe { int_array(affine.zero_point) };
        match scales {
            [] => Quantization::None,
            [scale] => Quantization::PerTensor(QuantizationParams {
                scale: *scale,
                zero_point: zero_points.first().copied().unwrap_or(0),
            }),
            _ => Quantization::PerAxis {
                scales,
                zero_points,
                axis: usize::try_from(affine.quantized_dimension).unwrap_or(0),
            },
        }
    }

    /// Whether the tensor is a constant stored in the model.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        // SAFETY: `tensor` is valid.
        let alloc = unsafe { self.fns.TfLiteOpaqueTensorGetAllocationType(self.tensor) };
        alloc == sys::TfLiteAllocationType_kTfLiteMmapRo
    }

    /// The tensor's data if it is a model constant, `None` otherwise.
    ///
    /// Constants can be read during node selection and kernel creation,
    /// before any tensors are allocated.
    pub fn constant_data<T: TensorElement>(&self) -> Result<Option<&'a [T]>> {
        if !self.is_constant() {
            return Ok(None);
        }
        // SAFETY: constant tensors point into the model buffer, which
        // outlives the graph; `slice` checks the element type and size.
        unsafe { self.slice::<T>() }.map(Some)
    }

    /// Typed view of the tensor's current data.
    ///
    /// # Safety
    ///
    /// The data must not be written for the lifetime `'a`.
    unsafe fn slice<T: TensorElement>(&self) -> Result<&'a [T]> {
        let (data, volume) = self.typed_ptr::<T>()?;
        // SAFETY: `data` is non-null, aligned, holds at least `volume`
        // elements of `T` (checked by `typed_ptr`), and is not written while
        // borrowed (caller).
        Ok(unsafe { std::slice::from_raw_parts(data, volume) })
    }

    fn data_ptr(&self) -> Result<*mut c_void> {
        // SAFETY: `tensor` is valid.
        let data = unsafe { self.fns.TfLiteOpaqueTensorData(self.tensor) };
        if data.is_null() {
            return Err(Error::null_pointer(format!(
                "tensor {} ({}) has no data yet",
                self.id.0,
                self.name()
            )));
        }
        Ok(data)
    }

    /// The data pointer, checked for `T`'s element width, size and alignment.
    fn typed_ptr<T: TensorElement>(&self) -> Result<(*mut T, usize)> {
        let volume = self.volume()?;
        check_element_type::<T>(self.tensor_type(), self.byte_size(), volume)?;
        let data = self.data_ptr()?.cast::<T>();
        if !data.is_aligned() {
            return Err(Error::invalid_argument(format!(
                "tensor {} data is not aligned for {}",
                self.id.0,
                std::any::type_name::<T>()
            )));
        }
        Ok((data, volume))
    }
}

impl fmt::Debug for TensorInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorInfo")
            .field("id", &self.id)
            .field("name", &self.name())
            .field("type", &self.tensor_type())
            .field("shape", &self.shape().ok())
            .finish_non_exhaustive()
    }
}

/// A group of connected nodes claimed by one delegate.
pub struct Partition<'a> {
    graph: GraphView<'a>,
    nodes: Vec<NodeId>,
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
}

impl<'a> Partition<'a> {
    /// The claimed nodes, in execution order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// Tensors the partition reads from outside itself, in the order
    /// [`KernelIo::input`] indexes them. Includes model constants.
    #[must_use]
    pub fn input_tensors(&self) -> &[TensorId] {
        &self.inputs
    }

    /// Tensors the partition writes for consumers outside itself, in the
    /// order [`KernelIo::output`] indexes them.
    #[must_use]
    pub fn output_tensors(&self) -> &[TensorId] {
        &self.outputs
    }

    /// The whole graph, for reading node parameters and the metadata of
    /// tensors internal to the partition.
    #[must_use]
    pub fn graph(&self) -> &GraphView<'a> {
        &self.graph
    }
}

impl fmt::Debug for Partition<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Partition")
            .field("nodes", &self.nodes)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Kernel I/O
// ---------------------------------------------------------------------------

/// A partition kernel's boundary tensors during `prepare` or `invoke`.
///
/// Inputs can be borrowed any number of times. Each output can be borrowed
/// mutably once per call.
pub struct KernelIo<'a> {
    fns: &'a sys::tensorflowlite_c,
    context: *mut sys::TfLiteOpaqueContext,
    node: *mut sys::TfLiteOpaqueNode,
    claimed: &'a [Cell<bool>],
}

impl<'a> KernelIo<'a> {
    /// Number of partition inputs.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        // SAFETY: `node` is the live delegate node.
        let n = unsafe { self.fns.TfLiteOpaqueNodeNumberOfInputs(self.node) };
        usize::try_from(n).unwrap_or(0)
    }

    /// Number of partition outputs.
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        // SAFETY: `node` is the live delegate node.
        let n = unsafe { self.fns.TfLiteOpaqueNodeNumberOfOutputs(self.node) };
        usize::try_from(n).unwrap_or(0)
    }

    /// The thread count the application gave the interpreter
    /// ([`InterpreterBuilder::num_threads`](crate::InterpreterBuilder::num_threads),
    /// `TfLiteInterpreterOptionsSetNumThreads` or `Interpreter::SetNumThreads`), or `None`
    /// when it gave none. `TFLite`'s own CPU kernels size their thread pools from the same
    /// value and run single-threaded when it is unset.
    ///
    /// Read it on every call: the application may change it between invokes.
    #[must_use]
    pub fn recommended_num_threads(&self) -> Option<std::num::NonZeroUsize> {
        // SAFETY: the runtime implements `TfLiteOpaqueContext` as its `TfLiteContext` (every
        // `TfLiteOpaqueContext*` function casts one to the other), `context` is live for this
        // call, and `recommended_num_threads` is a plain `int` of that C struct.
        let n = unsafe { (*self.context.cast::<sys::TfLiteContext>()).recommended_num_threads };
        usize::try_from(n)
            .ok()
            .and_then(std::num::NonZeroUsize::new)
    }

    /// Input `index`, in [`Partition::input_tensors`] order.
    pub fn input(&self, index: usize) -> Result<KernelTensor<'a>> {
        let n = self.num_inputs();
        let pos = position(index, n, "input")?;
        // SAFETY: `context` and `node` are live and `pos` is in range.
        let tensor = unsafe {
            self.fns
                .TfLiteOpaqueNodeGetInput(self.context, self.node, pos)
        };
        // SAFETY: `node` is live and `pos` is in range.
        let id = unsafe { self.fns.TfLiteOpaqueNodeGetInputTensorIndex(self.node, pos) };
        let id = TensorId::from_raw(id)
            .ok_or_else(|| Error::invalid_argument(format!("input {index} is omitted")))?;
        Ok(KernelTensor {
            info: TensorInfo::new(self.fns, tensor, id)?,
        })
    }

    /// Output `index`, in [`Partition::output_tensors`] order.
    ///
    /// Fails if the output was already borrowed during this call.
    pub fn output(&self, index: usize) -> Result<KernelTensorMut<'a>> {
        let n = self.num_outputs();
        let pos = position(index, n, "output")?;
        let slot = self
            .claimed
            .get(index)
            .ok_or_else(|| Error::invalid_argument(format!("output {index} is not tracked")))?;
        if slot.replace(true) {
            return Err(Error::invalid_argument(format!(
                "output {index} is already borrowed in this call"
            )));
        }
        // SAFETY: `context` and `node` are live and `pos` is in range.
        let tensor = unsafe {
            self.fns
                .TfLiteOpaqueNodeGetOutput(self.context, self.node, pos)
        };
        // SAFETY: `node` is live and `pos` is in range.
        let id = unsafe {
            self.fns
                .TfLiteOpaqueNodeGetOutputTensorIndex(self.node, pos)
        };
        let id = TensorId::from_raw(id)
            .ok_or_else(|| Error::invalid_argument(format!("output {index} is omitted")))?;
        Ok(KernelTensorMut {
            info: TensorInfo::new(self.fns, tensor, id)?,
        })
    }
}

impl fmt::Debug for KernelIo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelIo")
            .field("num_inputs", &self.num_inputs())
            .field("num_outputs", &self.num_outputs())
            .finish_non_exhaustive()
    }
}

/// A partition input borrowed by a kernel.
#[derive(Debug)]
pub struct KernelTensor<'a> {
    info: TensorInfo<'a>,
}

impl<'a> KernelTensor<'a> {
    /// The tensor's metadata.
    #[must_use]
    pub fn info(&self) -> &TensorInfo<'a> {
        &self.info
    }

    /// The tensor's data as `&[T]`.
    ///
    /// Fails if `T` does not match the element width, or the tensor is not
    /// allocated yet (during `prepare`).
    pub fn as_slice<T: TensorElement>(&self) -> Result<&'a [T]> {
        // SAFETY: partition inputs are not written while the kernel runs;
        // `KernelIo` never hands out an input as an output.
        unsafe { self.info.slice::<T>() }
    }
}

/// A partition output borrowed mutably by a kernel.
#[derive(Debug)]
pub struct KernelTensorMut<'a> {
    info: TensorInfo<'a>,
}

impl KernelTensorMut<'_> {
    /// The tensor's metadata.
    #[must_use]
    pub fn info(&self) -> &TensorInfo<'_> {
        &self.info
    }

    /// The tensor's data as `&mut [T]`.
    ///
    /// Fails if `T` does not match the element width, or the tensor is not
    /// allocated yet (during `prepare`).
    pub fn as_mut_slice<T: TensorElement>(&mut self) -> Result<&mut [T]> {
        let (data, volume) = self.info.typed_ptr::<T>()?;
        // SAFETY: `data` is aligned and holds at least `volume` elements of
        // `T` (checked by `typed_ptr`). `KernelIo::output` hands each output
        // out once per call and partition outputs are distinct from its
        // inputs, so this is the only reference to the data.
        Ok(unsafe { std::slice::from_raw_parts_mut(data, volume) })
    }
}

fn position(index: usize, len: usize, what: &str) -> Result<c_int> {
    if index >= len {
        return Err(Error::invalid_argument(format!(
            "{what} {index} out of range ({len} {what}s)"
        )));
    }
    c_int::try_from(index)
        .map_err(|_| Error::invalid_argument(format!("{what} {index} out of range")))
}

// ---------------------------------------------------------------------------
// Construction and C callbacks
// ---------------------------------------------------------------------------

/// The `TFLite` function table, owned by a custom delegate.
struct Fns(sys::tensorflowlite_c);

// SAFETY: the table holds function pointers and a `libloading::Library`; the
// `TFLite` C API has no thread affinity (same reasoning as `Library`).
unsafe impl Send for Fns {}
// SAFETY: as above; the table is never mutated after loading.
unsafe impl Sync for Fns {}

struct DelegateState<D> {
    fns: Fns,
    name: CString,
    delegate: D,
}

struct KernelState<K> {
    kernel: K,
    claimed: Box<[Cell<bool>]>,
}

/// The first opaque-API symbol a custom delegate needs that the runtime
/// does not export.
fn missing_symbol(fns: &sys::tensorflowlite_c) -> Option<&'static str> {
    macro_rules! first_missing {
        ($($sym:ident),+ $(,)?) => {
            [$((stringify!($sym), fns.$sym.is_ok())),+]
                .into_iter()
                .find_map(|(name, ok)| (!ok).then_some(name))
        };
    }
    first_missing!(
        TfLiteOpaqueDelegateCreate,
        TfLiteOpaqueDelegateDelete,
        TfLiteOpaqueContextGetExecutionPlan,
        TfLiteOpaqueContextGetNodeAndRegistration,
        TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels,
        TfLiteOpaqueContextGetOpaqueTensor,
        TfLiteOpaqueContextGetNumTensors,
        TfLiteOpaqueContextGetInputs,
        TfLiteOpaqueContextGetOutputs,
        TfLiteOpaqueContextReportError,
        TfLiteOpaqueNodeGetUserData,
        TfLiteOpaqueNodeGetBuiltinData,
        TfLiteOpaqueNodeInputs,
        TfLiteOpaqueNodeOutputs,
        TfLiteOpaqueNodeNumberOfInputs,
        TfLiteOpaqueNodeNumberOfOutputs,
        TfLiteOpaqueNodeGetInput,
        TfLiteOpaqueNodeGetOutput,
        TfLiteOpaqueNodeGetInputTensorIndex,
        TfLiteOpaqueNodeGetOutputTensorIndex,
        TfLiteOpaqueTensorType,
        TfLiteOpaqueTensorNumDims,
        TfLiteOpaqueTensorDim,
        TfLiteOpaqueTensorByteSize,
        TfLiteOpaqueTensorData,
        TfLiteOpaqueTensorName,
        TfLiteOpaqueTensorGetQuantization,
        TfLiteOpaqueTensorGetAllocationType,
        TfLiteOperatorCreate,
        TfLiteOperatorDelete,
        TfLiteOperatorGetBuiltInCode,
        TfLiteOperatorGetCustomName,
        TfLiteOperatorGetVersion,
        TfLiteOperatorSetInitWithData,
        TfLiteOperatorSetFreeWithData,
        TfLiteOperatorSetPrepareWithData,
        TfLiteOperatorSetInvokeWithData,
    )
}

impl Delegate {
    /// Wrap a Rust [`CustomDelegate`] as a delegate for an interpreter.
    ///
    /// The delegate value lives until the last interpreter using it is
    /// dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the loaded `TFLite` runtime does not export the
    /// opaque delegate API ([`Error::is_unsupported`] names the first missing
    /// symbol), or if the delegate cannot be created.
    pub fn from_custom<D: CustomDelegate>(lib: &Library, delegate: D) -> Result<Self> {
        // SAFETY: re-opening the already loaded TFLite library; the bindgen
        // loader resolves every symbol it declares, leaving missing optional
        // ones as errors that `missing_symbol` checks below.
        let fns =
            unsafe { sys::tensorflowlite_c::from_library(lib.reopen()?) }.map_err(Error::from)?;
        if let Some(symbol) = missing_symbol(&fns) {
            return Err(Error::unsupported(
                symbol,
                "custom delegates need the TFLite opaque delegate API (TFLite 2.17+)",
            ));
        }
        let free = *fns
            .TfLiteOpaqueDelegateDelete
            .as_ref()
            .map_err(|_| Error::unsupported("TfLiteOpaqueDelegateDelete", "custom delegate"))?;

        let state = Box::new(DelegateState {
            fns: Fns(fns),
            name: CString::from(delegate.name()),
            delegate,
        });
        let builder = sys::TfLiteOpaqueDelegateBuilder {
            data: ptr::from_ref::<DelegateState<D>>(&*state).cast_mut().cast(),
            Prepare: Some(delegate_prepare::<D>),
            CopyFromBufferHandle: None,
            CopyToBufferHandle: None,
            FreeBufferHandle: None,
            flags: 0,
        };
        // SAFETY: `builder` is fully initialised; its `data` points to the
        // boxed state, which `Delegate` owns and drops only after deleting
        // the opaque delegate.
        let raw = unsafe { state.fns.0.TfLiteOpaqueDelegateCreate(&raw const builder) };
        let handle = NonNull::new(raw)
            .ok_or_else(|| Error::null_pointer("TfLiteOpaqueDelegateCreate returned null"))?;
        let state: Box<dyn Any + Send + Sync> = state;

        Ok(Self {
            delegate: handle,
            free,
            _custom: Some(state),
            _lib: lib.reopen()?,
            #[cfg(feature = "dmabuf")]
            hal_dmabuf_fns: None,
            #[cfg(feature = "dmabuf")]
            hal_delegate_handle: None,
            #[cfg(feature = "dmabuf")]
            dmabuf_fns: None,
            #[cfg(feature = "camera_adaptor")]
            hal_camera_fns: None,
            #[cfg(feature = "camera_adaptor")]
            camera_adaptor_fns: None,
        })
    }
}

/// Run `f` at the C boundary: report errors and panics through `context`
/// and turn them into `kTfLiteError`.
fn guarded(
    fns: &sys::tensorflowlite_c,
    context: *mut sys::TfLiteOpaqueContext,
    name: &CStr,
    f: impl FnOnce() -> std::result::Result<(), BoxError>,
) -> sys::TfLiteStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => sys::TfLiteStatus_kTfLiteOk,
        Ok(Err(e)) => {
            report(fns, context, name, &e.to_string());
            sys::TfLiteStatus_kTfLiteError
        }
        Err(panic) => {
            report(fns, context, name, &panic_message(panic.as_ref()));
            sys::TfLiteStatus_kTfLiteError
        }
    }
}

fn panic_message(panic: &(dyn Any + Send)) -> String {
    let what = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic");
    format!("panicked: {what}")
}

fn report(
    fns: &sys::tensorflowlite_c,
    context: *mut sys::TfLiteOpaqueContext,
    name: &CStr,
    message: &str,
) {
    let Ok(report) = fns.TfLiteOpaqueContextReportError.as_ref() else {
        return;
    };
    let text = format!("{}: {message}", name.to_string_lossy()).replace('\0', " ");
    let Ok(text) = CString::new(text) else {
        return;
    };
    // SAFETY: `context` is the live callback context; the format string
    // consumes exactly one NUL-terminated string argument.
    unsafe { (*report)(context, c"%s".as_ptr(), text.as_ptr()) };
}

/// Checks that every selected node is in the execution plan, once: the runtime does not
/// validate the ids it is given.
fn check_selection(nodes: &[NodeId], plan: &[NodeId]) -> std::result::Result<(), BoxError> {
    let plan: std::collections::HashSet<NodeId> = plan.iter().copied().collect();
    let mut seen = std::collections::HashSet::with_capacity(nodes.len());
    for &n in nodes {
        if !plan.contains(&n) {
            return Err(format!(
                "select_nodes returned node {} outside the execution plan",
                n.0
            )
            .into());
        }
        if !seen.insert(n) {
            return Err(format!("select_nodes returned node {} twice", n.0).into());
        }
    }
    Ok(())
}

unsafe extern "C" fn delegate_prepare<D: CustomDelegate>(
    context: *mut sys::TfLiteOpaqueContext,
    opaque_delegate: *mut sys::TfLiteOpaqueDelegate,
    data: *mut c_void,
) -> sys::TfLiteStatus {
    // SAFETY: `data` is the `DelegateState<D>` registered in `from_custom`,
    // alive for as long as the delegate is.
    let state = unsafe { &*data.cast::<DelegateState<D>>() };
    let fns = &state.fns.0;
    guarded(fns, context, &state.name, || {
        let graph = GraphView::new(fns, context);
        let nodes = state.delegate.select_nodes(&graph)?;
        if nodes.is_empty() {
            return Ok(());
        }
        check_selection(&nodes, &graph.execution_plan()?)?;
        let ids: Vec<c_int> = std::iter::once(c_int::try_from(nodes.len())?)
            .chain(nodes.iter().map(|n| n.0))
            .collect();

        // SAFETY: the name is owned by `state`, which outlives every graph
        // the delegate is applied to; `data` is passed back as `user_data`.
        let registration = unsafe {
            fns.TfLiteOperatorCreate(
                sys::TfLiteBuiltinOperator_kTfLiteBuiltinDelegate,
                state.name.as_ptr(),
                1,
                data,
            )
        };
        if registration.is_null() {
            return Err("TfLiteOperatorCreate returned null".into());
        }
        // SAFETY: `registration` is a fresh operator; the callbacks match the
        // `*WithData` signatures.
        let set = unsafe {
            [
                fns.TfLiteOperatorSetInitWithData(registration, Some(kernel_init::<D>)),
                fns.TfLiteOperatorSetFreeWithData(registration, Some(kernel_free::<D>)),
                fns.TfLiteOperatorSetPrepareWithData(registration, Some(kernel_prepare::<D>)),
                fns.TfLiteOperatorSetInvokeWithData(registration, Some(kernel_invoke::<D>)),
            ]
        };
        if set.iter().any(|&s| s != sys::TfLiteStatus_kTfLiteOk) {
            // SAFETY: the runtime has not taken ownership yet.
            unsafe { fns.TfLiteOperatorDelete(registration) };
            return Err("setting delegate kernel callbacks failed".into());
        }
        // SAFETY: `ids` is laid out as a `TfLiteIntArray` (size, then data)
        // and outlives the call; the runtime takes ownership of
        // `registration`.
        let status = unsafe {
            fns.TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
                context,
                registration,
                ids.as_ptr().cast::<sys::TfLiteIntArray>(),
                opaque_delegate,
            )
        };
        status_to_result(status).map_err(|e| {
            e.with_context("TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels")
        })?;
        Ok(())
    })
}

unsafe extern "C" fn kernel_init<D: CustomDelegate>(
    user_data: *mut c_void,
    context: *mut sys::TfLiteOpaqueContext,
    buffer: *const c_char,
    _length: usize,
) -> *mut c_void {
    // SAFETY: `user_data` is the `DelegateState<D>` passed to
    // `TfLiteOperatorCreate`.
    let state = unsafe { &*user_data.cast::<DelegateState<D>>() };
    let fns = &state.fns.0;
    let mut kernel = None;
    guarded(fns, context, &state.name, || {
        if buffer.is_null() {
            return Err("delegate kernel created without parameters".into());
        }
        // SAFETY: for delegate kernels `buffer` is a runtime-owned
        // `TfLiteOpaqueDelegateParams`, valid for this call. The C API types
        // it as `char*`, so it is read without assuming alignment.
        let params = unsafe { buffer.cast::<TfLiteOpaqueDelegateParams>().read_unaligned() };
        // SAFETY: the arrays are runtime-owned and valid for this call.
        let (nodes, inputs, outputs) = unsafe {
            (
                int_array(params.nodes_to_replace),
                int_array(params.input_tensors),
                int_array(params.output_tensors),
            )
        };
        let partition = Partition {
            graph: GraphView::new(fns, context),
            nodes: nodes.iter().filter_map(|&n| NodeId::from_raw(n)).collect(),
            inputs: inputs
                .iter()
                .filter_map(|&t| TensorId::from_raw(t))
                .collect(),
            outputs: outputs
                .iter()
                .filter_map(|&t| TensorId::from_raw(t))
                .collect(),
        };
        let created = state.delegate.create_kernel(&partition)?;
        kernel = Some(Box::new(KernelState {
            kernel: created,
            claimed: (0..partition.outputs.len())
                .map(|_| Cell::new(false))
                .collect(),
        }));
        Ok(())
    });
    kernel.map_or(ptr::null_mut(), |k| Box::into_raw(k).cast())
}

unsafe extern "C" fn kernel_free<D: CustomDelegate>(
    user_data: *mut c_void,
    context: *mut sys::TfLiteOpaqueContext,
    data: *mut c_void,
) {
    // SAFETY: `user_data` is the `DelegateState<D>`.
    let state = unsafe { &*user_data.cast::<DelegateState<D>>() };
    if data.is_null() {
        return;
    }
    guarded(&state.fns.0, context, &state.name, || {
        // SAFETY: `data` was produced by `Box::into_raw` in `kernel_init`
        // and the runtime frees each kernel once.
        drop(unsafe { Box::from_raw(data.cast::<KernelState<D::Kernel>>()) });
        Ok(())
    });
}

unsafe extern "C" fn kernel_prepare<D: CustomDelegate>(
    user_data: *mut c_void,
    context: *mut sys::TfLiteOpaqueContext,
    node: *mut sys::TfLiteOpaqueNode,
) -> sys::TfLiteStatus {
    // SAFETY: `user_data` is the `DelegateState<D>`.
    let state = unsafe { &*user_data.cast::<DelegateState<D>>() };
    // SAFETY: forwarded from the runtime's callback arguments.
    unsafe { run_kernel::<D>(state, context, node, PartitionKernel::prepare) }
}

unsafe extern "C" fn kernel_invoke<D: CustomDelegate>(
    user_data: *mut c_void,
    context: *mut sys::TfLiteOpaqueContext,
    node: *mut sys::TfLiteOpaqueNode,
) -> sys::TfLiteStatus {
    // SAFETY: `user_data` is the `DelegateState<D>`.
    let state = unsafe { &*user_data.cast::<DelegateState<D>>() };
    // SAFETY: forwarded from the runtime's callback arguments.
    unsafe { run_kernel::<D>(state, context, node, PartitionKernel::invoke) }
}

/// # Safety
///
/// `context` and `node` must be the arguments of a kernel callback for a
/// node created by this delegate.
unsafe fn run_kernel<D: CustomDelegate>(
    state: &DelegateState<D>,
    context: *mut sys::TfLiteOpaqueContext,
    node: *mut sys::TfLiteOpaqueNode,
    f: impl FnOnce(&mut D::Kernel, &mut KernelIo<'_>) -> std::result::Result<(), BoxError>,
) -> sys::TfLiteStatus {
    let fns = &state.fns.0;
    guarded(fns, context, &state.name, || {
        // SAFETY: `node` is live; its user data is what `kernel_init`
        // returned.
        let data = unsafe { fns.TfLiteOpaqueNodeGetUserData(node) };
        if data.is_null() {
            return Err("kernel was not created".into());
        }
        // SAFETY: `data` is the `KernelState` boxed by `kernel_init`; the
        // runtime does not call a node's callbacks concurrently.
        let kernel = unsafe { &mut *data.cast::<KernelState<D::Kernel>>() };
        for slot in &kernel.claimed {
            slot.set(false);
        }
        let mut io = KernelIo {
            fns,
            context,
            node,
            claimed: &kernel.claimed,
        };
        f(&mut kernel.kernel, &mut io)
    })
}

// ---------------------------------------------------------------------------
// Runtime array helpers
// ---------------------------------------------------------------------------

/// # Safety
///
/// `array` must be null or point to a valid `TfLiteIntArray` that outlives
/// `'a`.
unsafe fn int_array<'a>(array: *const sys::TfLiteIntArray) -> &'a [c_int] {
    if array.is_null() {
        return &[];
    }
    // SAFETY: `array` is valid (caller).
    let len = usize::try_from(unsafe { (*array).size }).unwrap_or(0);
    // SAFETY: the array's `len` elements follow its size field.
    unsafe { std::slice::from_raw_parts((*array).data.as_ptr(), len) }
}

/// # Safety
///
/// `array` must be null or point to a valid `TfLiteFloatArray` that outlives
/// `'a`.
unsafe fn float_array<'a>(array: *const sys::TfLiteFloatArray) -> &'a [f32] {
    if array.is_null() {
        return &[];
    }
    // SAFETY: `array` is valid (caller).
    let len = usize::try_from(unsafe { (*array).size }).unwrap_or(0);
    // SAFETY: the array's `len` elements follow its size field.
    unsafe { std::slice::from_raw_parts((*array).data.as_ptr(), len) }
}

/// # Safety
///
/// `ids` must be null or point to `n` valid indices that outlive `'a`.
unsafe fn raw_ids<'a>(ids: *const c_int, n: c_int) -> &'a [c_int] {
    let len = usize::try_from(n).unwrap_or(0);
    if ids.is_null() || len == 0 {
        return &[];
    }
    // SAFETY: `ids` points to `len` indices (caller).
    unsafe { std::slice::from_raw_parts(ids, len) }
}

/// # Safety
///
/// As [`raw_ids`].
unsafe fn tensor_ids(ids: *const c_int, n: c_int) -> Vec<TensorId> {
    // SAFETY: forwarded from the caller.
    unsafe { raw_ids(ids, n) }
        .iter()
        .filter_map(|&t| TensorId::from_raw(t))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selections_must_name_plan_nodes_once() {
        let ids = |v: &[c_int]| v.iter().map(|&i| NodeId(i)).collect::<Vec<_>>();
        let plan = ids(&[0, 1, 2, 5]);
        assert!(check_selection(&ids(&[5, 0]), &plan).is_ok());
        assert!(check_selection(&ids(&[3]), &plan).is_err());
        assert!(check_selection(&ids(&[1, 2, 1]), &plan).is_err());
    }

    #[test]
    fn ids_reject_negative_indices() {
        assert_eq!(NodeId::from_raw(-1), None);
        assert_eq!(TensorId::from_raw(-1), None);
        assert_eq!(NodeId::from_raw(3).map(NodeId::index), Some(3));
        assert_eq!(TensorId::from_raw(7).map(TensorId::index), Some(7));
    }

    #[test]
    fn builtin_operator_round_trips_raw_codes() {
        assert_eq!(BuiltinOperator::SOFTMAX.raw(), 25);
        assert_eq!(BuiltinOperator::DELEGATE.raw(), 51);
        assert_eq!(BuiltinOperator::from_raw(25), BuiltinOperator::SOFTMAX);
    }

    #[test]
    fn params_are_bound_to_their_operators() {
        assert_eq!(SoftmaxParams::OPS, &[BuiltinOperator::SOFTMAX]);
        assert_eq!(BatchMatMulParams::OPS, &[BuiltinOperator::BATCH_MATMUL]);
        assert_eq!(ConcatenationParams::OPS, &[BuiltinOperator::CONCATENATION]);
    }

    #[test]
    fn position_rejects_out_of_range() {
        assert_eq!(position(1, 2, "input").unwrap(), 1);
        assert!(position(2, 2, "input").is_err());
    }

    #[test]
    fn panic_messages_are_extracted() {
        let s: Box<dyn Any + Send> = Box::new("boom");
        assert_eq!(panic_message(s.as_ref()), "panicked: boom");
        let s: Box<dyn Any + Send> = Box::new(String::from("bang"));
        assert_eq!(panic_message(s.as_ref()), "panicked: bang");
    }
}
