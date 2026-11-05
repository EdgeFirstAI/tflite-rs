// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Finds a softmax whose probabilities only feed a one-channel 1×1 convolution, through
//! reshapes: `SOFTMAX → [RESHAPE…] → CONV_2D`, the distribution-focal-loss head of YOLO
//! detectors.

use crate::graph::{Elem, Graph, Op, Tid};

/// One softmax-then-projection, as node indices into `Graph::nodes`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SoftmaxDot {
    pub(crate) softmax: usize,
    /// The reshapes between the softmax and the convolution, softmax side first.
    pub(crate) views: Vec<usize>,
    pub(crate) conv: usize,
}

impl SoftmaxDot {
    pub(crate) fn nodes(&self) -> impl Iterator<Item = usize> + '_ {
        std::iter::once(self.softmax)
            .chain(self.views.iter().copied())
            .chain(std::iter::once(self.conv))
    }

    /// Every tensor written inside the block, the convolution's output excluded.
    pub(crate) fn inner_tensors<'g>(&'g self, g: &'g Graph) -> impl Iterator<Item = Tid> + 'g {
        std::iter::once(self.softmax)
            .chain(self.views.iter().copied())
            .filter_map(|n| g.nodes[n].output())
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

/// Every block whose nodes all pass `ok`.
pub(crate) fn find(g: &Graph, ok: &dyn Fn(usize) -> bool) -> Vec<SoftmaxDot> {
    let mut found = Vec::new();
    for (ci, c) in g.nodes.iter().enumerate() {
        if !matches!(c.op, Op::Conv2d { unit: true, .. }) || !ok(ci) {
            continue;
        }
        let Some(block) = (|| {
            let w = g.tensor(c.input(1)?)?;
            let &[1, 1, 1, cin] = w.shape.as_slice() else {
                return None;
            };
            let mut views = Vec::new();
            let mut next = ci;
            let mut t = c.input(0)?;
            loop {
                let p = g.producer(t)?;
                if !ok(p) || sole_consumer(g, t)? != next {
                    return None;
                }
                match g.nodes[p].op {
                    Op::Reshape => {
                        views.push(p);
                        next = p;
                        t = g.nodes[p].input(0)?;
                    }
                    Op::Softmax { .. } => {
                        let x = g.tensor(g.nodes[p].input(0)?)?;
                        if x.elem != Elem::I8 || x.shape.last() != Some(&cin) {
                            return None;
                        }
                        views.reverse();
                        return Some(SoftmaxDot {
                            softmax: p,
                            views,
                            conv: ci,
                        });
                    }
                    _ => return None,
                }
            }
        })() else {
            continue;
        };
        found.push(block);
    }
    found
}
