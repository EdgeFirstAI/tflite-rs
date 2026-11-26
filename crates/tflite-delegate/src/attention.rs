// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Finds scaled dot-product attention blocks: `[MUL] → BATCH_MATMUL → SOFTMAX → BATCH_MATMUL`.
//!
//! The scores and the probabilities may pass through views on the way: reshapes that only
//! regroup the batch axes, and on the probabilities one transpose of the last two axes, as
//! Ultralytics' `LiteRT` export writes `Pᵀ` ahead of the second matmul instead of setting
//! `adj_y`. They are part of the block, so none of them is materialized.

use edgefirst_kernels::AttentionI8Plan;

use crate::graph::{Graph, Op, Tid};

/// Views allowed on either side of the softmax.
const MAX_VIEWS: usize = 4;

/// One attention block, as node indices into `Graph::nodes`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Attention {
    /// The scalar `MUL` on the first operand of `scores`, if it can be folded.
    pub(crate) mul: Option<usize>,
    /// `BATCH_MATMUL` producing the scores `S = A·B` (`[.., queries, tokens]`).
    pub(crate) scores: usize,
    /// Reshapes from the scores to the softmax, in graph order.
    pub(crate) pre: Vec<usize>,
    pub(crate) softmax: usize,
    /// Reshapes and at most one last-two-axes transpose from the softmax to `output`, in graph
    /// order.
    pub(crate) post: Vec<usize>,
    /// `BATCH_MATMUL` applying the probabilities `P` to the values.
    pub(crate) output: usize,
    /// `P` is the second operand of `output`, transposed (`O' = V'·Pᵀ`, `[.., dv, queries]`);
    /// otherwise the first (`O = P·V`, `[.., queries, dv]`).
    pub(crate) p_right: bool,
}

impl Attention {
    /// Every node of the block, in graph order.
    pub(crate) fn nodes(&self) -> impl Iterator<Item = usize> + '_ {
        self.mul
            .into_iter()
            .chain(std::iter::once(self.scores))
            .chain(self.pre.iter().copied())
            .chain(std::iter::once(self.softmax))
            .chain(self.post.iter().copied())
            .chain(std::iter::once(self.output))
    }

    /// Every tensor written inside the block, the output matmul's excluded.
    pub(crate) fn inner_tensors<'g>(&'g self, g: &'g Graph) -> impl Iterator<Item = Tid> + 'g {
        self.nodes()
            .filter(move |&n| n != self.output)
            .filter_map(|n| g.nodes[n].output())
    }

    /// The tensor `output` reads the probabilities from.
    pub(crate) fn p_operand(&self, g: &Graph) -> Option<Tid> {
        g.nodes[*self.post.last().unwrap_or(&self.softmax)].output()
    }
}

/// Whether reshape node `n` keeps its last two axes, regrouping only the batch axes.
fn batch_reshape(g: &Graph, n: usize) -> bool {
    let node = &g.nodes[n];
    let (Some(i), Some(o)) = (
        node.input(0).and_then(|t| g.tensor(t)),
        node.output().and_then(|t| g.tensor(t)),
    ) else {
        return false;
    };
    i.shape.len() >= 2
        && o.shape.len() >= 2
        && i.shape[i.shape.len() - 2..] == o.shape[o.shape.len() - 2..]
}

/// Whether transpose node `n` swaps the last two axes and keeps the others in place.
fn swaps_last_two(g: &Graph, n: usize) -> bool {
    let Some(perm) = g.nodes[n]
        .input(1)
        .and_then(|t| g.tensor(t))
        .and_then(|t| t.i32s())
    else {
        return false;
    };
    let r = perm.len();
    r >= 2
        && perm[..r - 2]
            .iter()
            .enumerate()
            .all(|(i, &p)| usize::try_from(p).is_ok_and(|p| p == i))
        && usize::try_from(perm[r - 2]).is_ok_and(|p| p == r - 1)
        && usize::try_from(perm[r - 1]).is_ok_and(|p| p == r - 2)
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

/// Every attention block whose nodes all pass `ok`, in graph order. Blocks share no node: in a
/// chain where one block's output matmul computes the next block's scores, only the first fuses.
pub(crate) fn find(g: &Graph, ok: &dyn Fn(usize) -> bool) -> Vec<Attention> {
    let mut found = Vec::new();
    let mut taken = vec![false; g.nodes.len()];
    for (si, s) in g.nodes.iter().enumerate() {
        if !matches!(s.op, Op::Softmax { .. }) || !ok(si) {
            continue;
        }
        let Some(a) = (|| {
            let ts = s.input(0)?;
            // Back from the softmax through batch reshapes to the scores.
            let mut pre = Vec::new();
            let (mut t, mut next) = (ts, si);
            let b1 = loop {
                let p = g.producer(t)?;
                if !ok(p) || sole_consumer(g, t)? != next {
                    return None;
                }
                match g.nodes[p].op {
                    Op::BatchMatMul { .. } => break p,
                    Op::Reshape if pre.len() < MAX_VIEWS && batch_reshape(g, p) => {
                        pre.push(p);
                        next = p;
                        t = g.nodes[p].input(0)?;
                    }
                    _ => return None,
                }
            };
            pre.reverse();
            let Op::BatchMatMul { adj_x, .. } = g.nodes[b1].op else {
                return None;
            };
            // Forward from the softmax through views to the output matmul.
            let mut post = Vec::new();
            let mut transposed = false;
            let mut tp = s.output()?;
            let b2 = loop {
                let c = sole_consumer(g, tp)?;
                if !ok(c) {
                    return None;
                }
                match g.nodes[c].op {
                    Op::BatchMatMul { .. } => break c,
                    Op::Reshape if post.len() < MAX_VIEWS && batch_reshape(g, c) => {}
                    Op::Transpose
                        if post.len() < MAX_VIEWS
                            && !transposed
                            && g.nodes[c].input(0) == Some(tp)
                            && swaps_last_two(g, c) =>
                    {
                        transposed = true;
                    }
                    _ => return None,
                }
                post.push(c);
                tp = g.nodes[c].output()?;
            };
            let n2 = &g.nodes[b2];
            let Op::BatchMatMul {
                adj_x: x2,
                adj_y: y2,
            } = n2.op
            else {
                return None;
            };
            // With `Pᵀ` read, an adjoint flag cancels the transpose instead of adding one.
            let p_right = if n2.input(0) == Some(tp) && n2.input(1) != Some(tp) && x2 == transposed
            {
                false
            } else if n2.input(1) == Some(tp) && n2.input(0) != Some(tp) && y2 != transposed {
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
                pre,
                softmax: si,
                post,
                output: b2,
                p_right,
            })
        })() else {
            continue;
        };
        if a.nodes().any(|n| taken[n]) {
            continue;
        }
        for n in a.nodes() {
            taken[n] = true;
        }
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
