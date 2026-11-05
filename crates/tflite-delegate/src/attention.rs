// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Finds scaled dot-product attention blocks: `[MUL] → BATCH_MATMUL → SOFTMAX → BATCH_MATMUL`.

use edgefirst_kernels::AttentionI8Plan;

use crate::graph::{Graph, Op, Tid};

/// One attention block, as node indices into `Graph::nodes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Attention {
    /// The scalar `MUL` on the first operand of `scores`, if it can be folded.
    pub(crate) mul: Option<usize>,
    /// `BATCH_MATMUL` producing the scores `S = A·B` (`[.., queries, tokens]`).
    pub(crate) scores: usize,
    pub(crate) softmax: usize,
    /// `BATCH_MATMUL` applying the probabilities `P` to the values.
    pub(crate) output: usize,
    /// `P` is the second operand of `output`, transposed (`O' = V'·Pᵀ`, `[.., dv, queries]`);
    /// otherwise the first (`O = P·V`, `[.., queries, dv]`).
    pub(crate) p_right: bool,
}

impl Attention {
    pub(crate) fn nodes(&self) -> impl Iterator<Item = usize> {
        self.mul
            .into_iter()
            .chain([self.scores, self.softmax, self.output])
    }
}

/// The only node reading `t`, when `t` is not a graph output.
fn sole_consumer(g: &Graph, t: Tid) -> Option<usize> {
    if g.outputs.contains(&t) {
        return None;
    }
    let mut c = g.consumers(t);
    let first = c.next()?;
    c.next().is_none().then_some(first)
}

/// Every attention block whose nodes all pass `ok`.
pub(crate) fn find(g: &Graph, ok: &dyn Fn(usize) -> bool) -> Vec<Attention> {
    let mut found = Vec::new();
    for (si, s) in g.nodes.iter().enumerate() {
        if !matches!(s.op, Op::Softmax { .. }) || !ok(si) {
            continue;
        }
        let Some(a) = (|| {
            let ts = s.input(0)?;
            let tp = s.output()?;
            let b1 = g.producer(ts)?;
            if !ok(b1) || sole_consumer(g, ts)? != si {
                return None;
            }
            let Op::BatchMatMul { adj_x, .. } = g.nodes[b1].op else {
                return None;
            };
            let b2 = sole_consumer(g, tp)?;
            if !ok(b2) {
                return None;
            }
            let n2 = &g.nodes[b2];
            let Op::BatchMatMul {
                adj_x: x2,
                adj_y: y2,
            } = n2.op
            else {
                return None;
            };
            let p_right = if n2.input(0) == Some(tp) && n2.input(1) != Some(tp) && !x2 {
                false
            } else if n2.input(1) == Some(tp) && n2.input(0) != Some(tp) && y2 {
                true
            } else {
                return None;
            };

            // Shapes within the fused kernel's limits.
            let sh = &g.tensor(ts)?.shape;
            let tokens = *sh.last()?;
            let a1 = &g.tensor(g.nodes[b1].input(0)?)?.shape;
            let dk = *if adj_x {
                a1.get(a1.len().checked_sub(2)?)?
            } else {
                a1.last()?
            };
            let o = &g.tensor(n2.output()?)?.shape;
            let dv = *if p_right {
                o.get(o.len().checked_sub(2)?)?
            } else {
                o.last()?
            };
            AttentionI8Plan::check_shape(tokens, dk, dv).ok()?;

            // A scalar constant `MUL` producing the first operand only for the scores.
            let q = g.nodes[b1].input(0)?;
            let mul = g.producer(q).filter(|&m| {
                ok(m)
                    && matches!(g.nodes[m].op, Op::Mul(_))
                    && sole_consumer(g, q) == Some(b1)
                    && g.nodes[b1].input(1) != Some(q)
                    && scalar_operand(g, m).is_some()
            });
            Some(Attention {
                mul,
                scores: b1,
                softmax: si,
                output: b2,
                p_right,
            })
        })() else {
            continue;
        };
        found.push(a);
    }
    found
}

/// For a `MUL` node with one scalar constant operand: `(constant input slot, other slot)`.
pub(crate) fn scalar_operand(g: &Graph, m: usize) -> Option<(usize, usize)> {
    let n = &g.nodes[m];
    let is_scalar_const = |i: usize| {
        n.input(i)
            .and_then(|t| g.tensor(t))
            .is_some_and(|t| t.numel() == 1 && t.constant.is_some())
    };
    match (is_scalar_const(0), is_scalar_const(1)) {
        (false, true) => Some((1, 0)),
        (true, false) => Some((0, 1)),
        _ => None,
    }
}
