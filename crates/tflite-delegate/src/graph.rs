// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! An owned snapshot of the parts of a `TFLite` graph the delegate reasons about.
//!
//! Node selection and region planning work on this snapshot rather than on the live
//! [`GraphView`], so they can be tested without a `TFLite` runtime.

use std::collections::HashMap;

use edgefirst_kernels::{Activation, Quant};
use edgefirst_tflite::delegate::custom::{
    AddParams, BatchMatMulParams, BoxError, BuiltinOperator, ConcatenationParams, ConvParams,
    GraphView, MulParams, NodeId, Quantization, SoftmaxParams, SubParams, TensorId,
};
use edgefirst_tflite::TensorType;

/// A tensor index.
pub(crate) type Tid = usize;

/// Element types the delegate distinguishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Elem {
    I8,
    I16,
    I32,
    Other,
}

/// A model constant.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Const {
    I8(Vec<i8>),
    I32(Vec<i32>),
}

/// A tensor's static description.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Tensor {
    pub(crate) elem: Elem,
    pub(crate) shape: Vec<usize>,
    /// Per-tensor quantization, if any.
    pub(crate) quant: Option<Quant>,
    /// Per-axis scales (filters), with every zero point 0 and the axis 0.
    pub(crate) axis_scales: Option<Vec<f32>>,
    pub(crate) constant: Option<Const>,
}

impl Tensor {
    pub(crate) fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub(crate) fn i32s(&self) -> Option<&[i32]> {
        match &self.constant {
            Some(Const::I32(v)) => Some(v),
            _ => None,
        }
    }

    /// Whether two tensors share element type and quantization, so one can be a view of the
    /// other.
    pub(crate) fn same_encoding(&self, other: &Tensor) -> bool {
        self.elem == other.elem && self.quant == other.quant
    }
}

/// The operators the delegate understands, with their parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Op {
    Softmax { beta: f32 },
    Logistic,
    Add(Activation),
    Sub(Activation),
    Mul(Activation),
    BatchMatMul { adj_x: bool, adj_y: bool },
    Conv2d { unit: bool, act: Activation },
    Transpose,
    Reshape,
    Slice,
    Concat { axis: i32, act: Activation },
    Other,
}

/// A node of the execution plan.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Node {
    /// The node's index in the `TFLite` graph.
    pub(crate) index: usize,
    pub(crate) op: Op,
    pub(crate) inputs: Vec<Option<Tid>>,
    pub(crate) outputs: Vec<Option<Tid>>,
}

impl Node {
    pub(crate) fn input(&self, i: usize) -> Option<Tid> {
        self.inputs.get(i).copied().flatten()
    }

    pub(crate) fn output(&self) -> Option<Tid> {
        self.outputs.first().copied().flatten()
    }
}

/// The graph: nodes in execution order and the tensors they touch.
#[derive(Clone, Debug, Default)]
pub(crate) struct Graph {
    pub(crate) nodes: Vec<Node>,
    /// The `NodeId` of each node, for claiming.
    pub(crate) ids: Vec<NodeId>,
    pub(crate) tensors: HashMap<Tid, Tensor>,
    /// Graph outputs, which must be written whatever reads them.
    pub(crate) outputs: Vec<Tid>,
}

impl Graph {
    pub(crate) fn tensor(&self, t: Tid) -> Option<&Tensor> {
        self.tensors.get(&t)
    }

    /// Indices (into `nodes`) of the nodes reading `t`.
    pub(crate) fn consumers(&self, t: Tid) -> impl Iterator<Item = usize> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter(move |(_, n)| n.inputs.contains(&Some(t)))
            .map(|(i, _)| i)
    }

    /// The index of the node writing `t`.
    pub(crate) fn producer(&self, t: Tid) -> Option<usize> {
        self.nodes.iter().position(|n| n.outputs.contains(&Some(t)))
    }

    /// Snapshot of the whole execution plan of `view`.
    pub(crate) fn from_view(view: &GraphView<'_>) -> Result<Self, BoxError> {
        let mut g = Self::from_nodes(view, &view.execution_plan()?)?;
        g.outputs = view
            .graph_outputs()?
            .into_iter()
            .map(TensorId::index)
            .collect();
        Ok(g)
    }

    /// Snapshot of a partition: its nodes, in execution order, with `outputs` (the tensors
    /// read outside it) as the graph outputs.
    pub(crate) fn from_partition(
        view: &GraphView<'_>,
        nodes: &[NodeId],
        outputs: &[TensorId],
    ) -> Result<Self, BoxError> {
        let mut g = Self::from_nodes(view, nodes)?;
        g.outputs = outputs.iter().map(|t| t.index()).collect();
        Ok(g)
    }

    fn from_nodes(view: &GraphView<'_>, ids: &[NodeId]) -> Result<Self, BoxError> {
        let mut g = Graph::default();
        for &id in ids {
            let node = view.node(id)?;
            let op = op_of(&node);
            let (raw_in, raw_out) = (node.inputs(), node.outputs());
            // Constants are copied only for operators the delegate may run.
            let want_const = op != Op::Other;
            for tid in raw_in.iter().chain(&raw_out).flatten() {
                let needed = g
                    .tensors
                    .get(&tid.index())
                    .is_none_or(|t| want_const && t.constant.is_none());
                if needed {
                    g.tensors
                        .insert(tid.index(), tensor_of(view, *tid, want_const)?);
                }
            }
            let inputs: Vec<Option<Tid>> = raw_in.iter().map(|t| t.map(TensorId::index)).collect();
            let outputs: Vec<Option<Tid>> =
                raw_out.iter().map(|t| t.map(TensorId::index)).collect();
            g.ids.push(id);
            g.nodes.push(Node {
                index: id.index(),
                op,
                inputs,
                outputs,
            });
        }
        Ok(g)
    }
}

fn activation(raw: i32) -> Option<Activation> {
    match raw {
        0 => Some(Activation::None),
        1 => Some(Activation::Relu),
        2 => Some(Activation::ReluN1To1),
        3 => Some(Activation::Relu6),
        _ => None,
    }
}

fn op_of(node: &edgefirst_tflite::delegate::custom::NodeView<'_>) -> Op {
    let op = node.builtin_op();
    match op {
        BuiltinOperator::SOFTMAX => node
            .params::<SoftmaxParams>()
            .map_or(Op::Other, |p| Op::Softmax { beta: p.beta }),
        BuiltinOperator::LOGISTIC => Op::Logistic,
        BuiltinOperator::ADD => node
            .params::<AddParams>()
            .and_then(|p| activation(p.activation))
            .map_or(Op::Other, Op::Add),
        BuiltinOperator::SUB => node
            .params::<SubParams>()
            .and_then(|p| activation(p.activation))
            .map_or(Op::Other, Op::Sub),
        BuiltinOperator::MUL => node
            .params::<MulParams>()
            .and_then(|p| activation(p.activation))
            .map_or(Op::Other, Op::Mul),
        BuiltinOperator::BATCH_MATMUL => node
            .params::<BatchMatMulParams>()
            .filter(|p| !p.asymmetric_quantize_inputs)
            .map_or(Op::Other, |p| Op::BatchMatMul {
                adj_x: p.adj_x,
                adj_y: p.adj_y,
            }),
        BuiltinOperator::CONV_2D => node.params::<ConvParams>().map_or(Op::Other, |p| {
            activation(p.activation).map_or(Op::Other, |act| Op::Conv2d {
                unit: p.stride_width == 1
                    && p.stride_height == 1
                    && p.dilation_width_factor == 1
                    && p.dilation_height_factor == 1,
                act,
            })
        }),
        BuiltinOperator::TRANSPOSE => Op::Transpose,
        BuiltinOperator::RESHAPE => Op::Reshape,
        BuiltinOperator::SLICE => Op::Slice,
        BuiltinOperator::CONCATENATION => node
            .params::<ConcatenationParams>()
            .and_then(|p| activation(p.activation).map(|act| (p.axis, act)))
            .map_or(Op::Other, |(axis, act)| Op::Concat { axis, act }),
        _ => Op::Other,
    }
}

fn tensor_of(view: &GraphView<'_>, id: TensorId, want_const: bool) -> Result<Tensor, BoxError> {
    let info = view.tensor(id)?;
    let elem = match info.tensor_type() {
        TensorType::Int8 => Elem::I8,
        TensorType::Int16 => Elem::I16,
        TensorType::Int32 => Elem::I32,
        _ => Elem::Other,
    };
    // Unknown dimensions make the tensor unusable; an empty shape marks it.
    let shape = info.shape().unwrap_or_default();
    let (quant, axis_scales) = match info.quantization() {
        Quantization::None => (None, None),
        Quantization::PerTensor(q) => (Some(Quant::new(q.scale, q.zero_point)), None),
        Quantization::PerAxis {
            scales,
            zero_points,
            axis,
        } => {
            let symmetric = axis == 0 && zero_points.iter().all(|&z| z == 0);
            (None, symmetric.then(|| scales.to_vec()))
        }
    };
    let constant = match elem {
        _ if !want_const => None,
        Elem::I8 => info.constant_data::<i8>()?.map(|d| Const::I8(d.to_vec())),
        Elem::I32 => info.constant_data::<i32>()?.map(|d| Const::I32(d.to_vec())),
        _ => None,
    };
    Ok(Tensor {
        elem,
        shape,
        quant,
        axis_scales,
        constant,
    })
}
