// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Which nodes the delegate claims.
//!
//! A node is supported when the kernels reproduce `TFLite`'s result for its exact types,
//! quantization and parameters, or (view operators) when it only rearranges elements. The
//! delegate claims connected groups of supported nodes that contain at least one compute
//! node: a group of view operators alone would only trade `TFLite`'s copies for the
//! delegate's.

use std::collections::{HashMap, HashSet};

use edgefirst_kernels::{
    BinaryI8Params, BinaryI8Plan, BinaryOp, CpuFeatures, DType, LutI8Plan, MatMulI8Params,
    MatMulI8Plan, Quant, SoftmaxI16Params, SoftmaxI16Plan, SoftmaxI8Params, SoftmaxI8Plan,
    TfliteKernels, MAX_RANK,
};

use crate::attention;
use crate::graph::{Elem, Graph, Node, Op, Tensor};
use crate::Scope;

/// How a supported node is executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    View,
    Compute,
}

/// Whether the delegate runs `n`, and how.
pub(crate) fn support(g: &Graph, n: &Node, f: CpuFeatures, kernels: TfliteKernels) -> Option<Role> {
    let out = g.tensor(n.output()?)?;
    if out.shape.is_empty() && out.numel() != 1 || out.shape.len() > MAX_RANK {
        return None;
    }
    let input = |i: usize| n.input(i).and_then(|t| g.tensor(t));
    match n.op {
        Op::Softmax { beta } => {
            let x = input(0)?;
            if x.shape != out.shape || out.shape.is_empty() {
                return None;
            }
            let (iq, oq) = (x.quant?, out.quant?);
            let ok = match (x.elem, out.elem) {
                (Elem::I8, Elem::I8) => SoftmaxI8Plan::with_features(
                    SoftmaxI8Params {
                        input: iq,
                        output: oq,
                        beta,
                    },
                    f,
                )
                .is_ok(),
                (Elem::I16, Elem::I16) => SoftmaxI16Plan::with_features(
                    SoftmaxI16Params {
                        input: iq,
                        output: oq,
                        beta,
                    },
                    f,
                )
                .is_ok(),
                _ => false,
            };
            ok.then_some(Role::Compute)
        }
        Op::Logistic => {
            let x = input(0)?;
            (x.elem == Elem::I8
                && out.elem == Elem::I8
                && x.shape == out.shape
                && LutI8Plan::sigmoid_with_features(DType::I8, x.quant?, out.quant?, f).is_ok())
            .then_some(Role::Compute)
        }
        Op::Add(act) | Op::Sub(act) | Op::Mul(act) => {
            let (a, b) = (input(0)?, input(1)?);
            if a.elem != Elem::I8 || b.elem != Elem::I8 || out.elem != Elem::I8 {
                return None;
            }
            let op = match n.op {
                Op::Add(_) => BinaryOp::Add,
                Op::Sub(_) => BinaryOp::Sub,
                _ => BinaryOp::Mul,
            };
            // TFLite runs same-shape operands through its elementwise loops, and a scalar
            // `MUL` operand through `MulSimpleBroadcast`; other broadcasts split the NEON
            // blocks differently.
            let same = same_extended(&a.shape, &out.shape) && same_extended(&b.shape, &out.shape);
            let scalar = op == BinaryOp::Mul
                && (a.numel() == 1 && a.constant.is_some()
                    || b.numel() == 1 && b.constant.is_some());
            if !(same || scalar) {
                return None;
            }
            // NEON builds compute whole 16-element blocks differently from the tail; claim only
            // tensors without a tail, so any split of the work keeps TFLite's result.
            if kernels.neon_blocks() && op != BinaryOp::Sub && out.numel() % 16 != 0 {
                return None;
            }
            BinaryI8Plan::with_features(
                BinaryI8Params {
                    op,
                    lhs: a.quant?,
                    rhs: b.quant?,
                    output: out.quant?,
                    activation: act,
                    kernels,
                },
                f,
            )
            .ok()
            .map(|_| Role::Compute)
        }
        Op::BatchMatMul { .. } => {
            let (a, b) = (input(0)?, input(1)?);
            let k = *if matches!(n.op, Op::BatchMatMul { adj_x: true, .. }) {
                a.shape.get(a.shape.len().checked_sub(2)?)?
            } else {
                a.shape.last()?
            };
            if a.elem != Elem::I8
                || b.elem != Elem::I8
                || out.elem != Elem::I8
                || a.shape.len() < 2
                || b.shape.len() < 2
                || MatMulI8Plan::check_depth(k).is_err()
            {
                return None;
            }
            // The plan the planner builds, so a node claimed here always plans.
            let p = MatMulI8Params::tflite_batch_matmul(a.quant?, b.quant?, out.quant?, kernels);
            p.and_then(|p| MatMulI8Plan::with_features(p, f))
                .ok()
                .map(|_| Role::Compute)
        }
        Op::Conv2d { unit: true, act } => {
            let (x, w) = (input(0)?, input(1)?);
            let bias = match n.input(2).and_then(|t| g.tensor(t)) {
                None => Vec::new(),
                Some(b) if b.elem == Elem::I32 && b.constant.is_some() => b.i32s()?.to_vec(),
                Some(_) => return None,
            };
            if x.elem != Elem::I8
                || out.elem != Elem::I8
                || w.elem != Elem::I8
                || w.constant.is_none()
                || x.shape.len() != 4
                || w.shape.len() != 4
                || w.shape[1] != 1
                || w.shape[2] != 1
                || w.shape[3] != x.shape[3]
                || MatMulI8Plan::check_depth(x.shape[3]).is_err()
            {
                return None;
            }
            // The plan the planner builds, so a node claimed here always plans.
            let scales = filter_scales(w)?;
            let p = MatMulI8Params::tflite_conv(x.quant?, &scales, out.quant?, bias, act, kernels);
            p.and_then(|p| MatMulI8Plan::with_features(p, f))
                .ok()
                .map(|_| Role::Compute)
        }
        Op::Transpose | Op::Reshape | Op::Slice => {
            let x = input(0)?;
            if !x.same_encoding(out) || !matches!(x.elem, Elem::I8 | Elem::I16) {
                return None;
            }
            let params_ok = match n.op {
                Op::Transpose => input(1)
                    .and_then(Tensor::i32s)
                    .is_some_and(|p| p.len() == x.shape.len()),
                Op::Slice => input(1)
                    .and_then(Tensor::i32s)
                    .is_some_and(|b| b.len() == x.shape.len() && b.iter().all(|&v| v >= 0)),
                _ => out.numel() == x.numel(),
            };
            params_ok.then_some(Role::View)
        }
        Op::Concat { act, .. } => {
            let same = n.inputs.iter().all(|t| {
                t.and_then(|t| g.tensor(t))
                    .is_some_and(|x| x.same_encoding(out))
            });
            (same
                && act == edgefirst_kernels::Activation::None
                && matches!(out.elem, Elem::I8 | Elem::I16))
            .then_some(Role::View)
        }
        _ => None,
    }
}

/// The filter's per-output-channel scales, when its zero points are 0.
pub(crate) fn filter_scales(w: &Tensor) -> Option<Vec<f32>> {
    match (&w.axis_scales, w.quant) {
        (Some(s), _) if s.len() == *w.shape.first()? => Some(s.clone()),
        (
            None,
            Some(Quant {
                scale,
                zero_point: 0,
            }),
        ) => Some(vec![scale]),
        _ => None,
    }
}

/// Whether `a`, right-aligned and padded with leading 1s, is `b`.
pub(crate) fn same_extended(a: &[usize], b: &[usize]) -> bool {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let extra = long.len() - short.len();
    long[..extra].iter().all(|&d| d == 1) && long[extra..] == *short
}

/// The nodes (indices into `g.nodes`) to claim.
pub(crate) fn select(
    g: &Graph,
    f: CpuFeatures,
    kernels: TfliteKernels,
    scope: Scope,
) -> Vec<usize> {
    let roles: Vec<Option<Role>> = g.nodes.iter().map(|n| support(g, n, f, kernels)).collect();
    if scope == Scope::Attention {
        // The matcher checks each node's operator; views inside a block come with it.
        let mut picked: Vec<usize> = attention::find(g, &|i| roles[i].is_some())
            .into_iter()
            .flat_map(|a| a.nodes().collect::<Vec<_>>())
            .collect();
        picked.sort_unstable();
        picked.dedup();
        return picked;
    }
    // Union-find over supported nodes joined by a tensor one writes and the other reads.
    let mut parent: Vec<usize> = (0..g.nodes.len()).collect();
    let producer: HashMap<usize, usize> = g
        .nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| roles[*i].is_some())
        .flat_map(|(i, n)| n.outputs.iter().flatten().map(move |&t| (t, i)))
        .collect();
    for (i, n) in g.nodes.iter().enumerate() {
        if roles[i].is_none() {
            continue;
        }
        for t in n.inputs.iter().flatten() {
            if let Some(&p) = producer.get(t) {
                let (a, b) = (root(&mut parent, i), root(&mut parent, p));
                parent[a] = b;
            }
        }
    }
    let computing: HashSet<usize> = (0..g.nodes.len())
        .filter(|&i| roles[i] == Some(Role::Compute))
        .map(|i| root(&mut parent, i))
        .collect();
    (0..g.nodes.len())
        .filter(|&i| roles[i].is_some() && computing.contains(&root(&mut parent, i)))
        .collect()
}

/// The root of `i` in the union-find forest `p`, halving the path on the way.
fn root(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}
