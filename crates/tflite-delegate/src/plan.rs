// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Lowers one claimed partition to a straight-line program of kernel steps over strided
//! locations.
//!
//! View operators (`TRANSPOSE`, `RESHAPE`, `SLICE`) only transform locations. `CONCATENATION`
//! yields a list of parts along one axis that elementwise and row operators consume part by
//! part. A compute operator writes straight into where its result is finally needed when that
//! is reachable through invertible views (a partition output, or a slot of a concatenation);
//! otherwise into scratch. Copies happen only where a location cannot be expressed as a view.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use edgefirst_kernels::{
    AttentionI8Params, AttentionI8Plan, AttentionQMul, AttentionScratch, BinaryI8Params,
    BinaryI8Plan, BinaryOp, CpuFeatures, DType, LutI8Plan, MatMulI8Params, MatMulI8Plan, Quant,
    SoftmaxDotI8Params, SoftmaxDotI8Plan, SoftmaxI16Params, SoftmaxI16Plan, SoftmaxI8Params,
    SoftmaxI8Plan, TfliteKernels,
};
use edgefirst_tflite::delegate::custom::BoxError;

use crate::attention::{self, Attention};
use crate::graph::{Const, Elem, Graph, Op, Tid};
use crate::select::filter_scales;
use crate::softmax_dot::{self, SoftmaxDot};
use crate::views::{Buf, Loc};

/// Planning options.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanOptions {
    pub(crate) features: CpuFeatures,
    pub(crate) kernels: TfliteKernels,
}

/// Where a tensor's elements are.
#[derive(Clone, Debug)]
enum Place {
    View(Loc),
    /// Consecutive parts along `axis`.
    Parts {
        axis: usize,
        parts: Vec<Loc>,
    },
}

/// One kernel call.
#[derive(Debug)]
pub(crate) enum Step {
    Softmax8 {
        plan: Arc<SoftmaxI8Plan>,
        src: Loc,
        dst: Loc,
    },
    Softmax16 {
        plan: Arc<SoftmaxI16Plan>,
        src: Loc,
        dst: Loc,
    },
    /// A softmax and the one-channel projection of its probabilities.
    SoftmaxDot {
        plan: Arc<SoftmaxDotI8Plan>,
        src: Loc,
        dst: Loc,
    },
    Lut {
        plan: Arc<LutI8Plan>,
        src: Loc,
        dst: Loc,
    },
    Binary {
        plan: BinaryI8Plan,
        lhs: Loc,
        rhs: Loc,
        dst: Loc,
    },
    MatMul {
        plan: MatMulI8Plan,
        lhs: Loc,
        rhs: Loc,
        dst: Loc,
    },
    Attention {
        plan: Box<AttentionI8Plan>,
        scratch: usize,
        q: Loc,
        k: Loc,
        v: Loc,
        dst: Loc,
    },
    Copy {
        elem: Elem,
        src: Loc,
        dst: Loc,
    },
}

/// A planned partition.
#[derive(Debug)]
pub(crate) struct Program {
    pub(crate) steps: Vec<Step>,
    /// Element type and length of each scratch buffer.
    pub(crate) scratch: Vec<(Elem, usize)>,
    pub(crate) attention: Vec<AttentionScratch>,
    /// Element type and shape of each partition input and output.
    pub(crate) inputs: Vec<(Elem, Vec<usize>)>,
    pub(crate) outputs: Vec<(Elem, Vec<usize>)>,
    /// Attention blocks fused into one step.
    pub(crate) fused_attention: usize,
    /// Softmax-then-projection blocks fused into one step.
    pub(crate) fused_softmax_dot: usize,
}

struct Planner<'a> {
    g: &'a Graph,
    o: PlanOptions,
    nodes: HashSet<usize>,
    in_idx: HashMap<Tid, usize>,
    out_idx: HashMap<Tid, usize>,
    loc: HashMap<Tid, Place>,
    /// View nodes whose output location was set from their consumer's side.
    placed_through: HashSet<usize>,
    steps: Vec<Step>,
    scratch: Vec<(Elem, usize)>,
    attention: Vec<AttentionScratch>,
    fused_attention: usize,
    fused_softmax_dot: usize,
}

fn err(msg: &str) -> BoxError {
    msg.to_owned().into()
}

impl Planner<'_> {
    fn shape(&self, t: Tid) -> Result<Vec<usize>, BoxError> {
        Ok(self
            .g
            .tensor(t)
            .ok_or_else(|| err("unknown tensor"))?
            .shape
            .clone())
    }

    fn elem(&self, t: Tid) -> Elem {
        self.g.tensor(t).map_or(Elem::Other, |x| x.elem)
    }

    fn quant(&self, t: Tid) -> Result<Quant, BoxError> {
        self.g
            .tensor(t)
            .and_then(|x| x.quant)
            .ok_or_else(|| err("tensor is not per-tensor quantized"))
    }

    fn new_scratch(&mut self, elem: Elem, shape: &[usize]) -> Loc {
        let k = self.scratch.len();
        self.scratch.push((elem, shape.iter().product()));
        Loc::dense(Buf::Scratch(k), shape)
    }

    /// Partition-internal consumers of `t`.
    fn consumers(&self, t: Tid) -> Vec<usize> {
        self.g
            .consumers(t)
            .filter(|c| self.nodes.contains(c))
            .collect()
    }

    /// The location of an input tensor.
    fn input(&mut self, t: Tid) -> Result<Place, BoxError> {
        if let Some(p) = self.loc.get(&t) {
            return Ok(p.clone());
        }
        let &k = self
            .in_idx
            .get(&t)
            .ok_or_else(|| err("tensor read before it is written"))?;
        let p = Place::View(Loc::dense(Buf::Input(k), &self.shape(t)?));
        self.loc.insert(t, p.clone());
        Ok(p)
    }

    /// Where the consumers of `t` would like it, through invertible views, if anywhere.
    fn lookahead(&mut self, t: Tid) -> Option<Loc> {
        if let Some(Place::View(l)) = self.loc.get(&t) {
            return Some(l.clone());
        }
        let shape = self.shape(t).ok()?;
        if let Some(&k) = self.out_idx.get(&t) {
            let l = Loc::dense(Buf::Output(k), &shape);
            self.loc.insert(t, Place::View(l.clone()));
            return Some(l);
        }
        let cs = self.consumers(t);
        let &[c] = cs.as_slice() else {
            return None;
        };
        let node = self.g.nodes.get(c)?.clone();
        let out = node.output()?;
        let found = match node.op {
            Op::Transpose if node.input(0) == Some(t) => {
                let perm = self.perm(&node.inputs)?;
                let mut inv = vec![0; perm.len()];
                for (a, &p) in perm.iter().enumerate() {
                    *inv.get_mut(p)? = a;
                }
                self.lookahead(out)?.transpose(&inv)
            }
            Op::Reshape if node.input(0) == Some(t) => self.lookahead(out)?.reshape(&shape),
            Op::Concat { axis, .. } => {
                let target = self.lookahead(out)?;
                let axis = norm_axis(axis, target.shape.len())?;
                let mut begin = 0;
                let mut slot = None;
                for x in node.inputs.iter().flatten() {
                    if *x == t {
                        slot = Some(begin);
                        break;
                    }
                    begin += *self.g.tensor(*x)?.shape.get(axis)?;
                }
                target.narrow(axis, slot?, *shape.get(axis)?)
            }
            _ => None,
        }?;
        if !matches!(node.op, Op::Concat { .. }) {
            self.placed_through.insert(c);
        }
        self.loc.insert(t, Place::View(found.clone()));
        Some(found)
    }

    /// Where a compute node writes `t`: its consumers' preferred location, or scratch.
    fn output(&mut self, t: Tid) -> Result<Loc, BoxError> {
        if let Some(l) = self.lookahead(t) {
            return Ok(l);
        }
        let l = self.new_scratch(self.elem(t), &self.shape(t)?);
        self.loc.insert(t, Place::View(l.clone()));
        Ok(l)
    }

    fn perm(&self, inputs: &[Option<Tid>]) -> Option<Vec<usize>> {
        let p = self.g.tensor(inputs.get(1).copied()??)?.i32s()?;
        p.iter().map(|&v| usize::try_from(v).ok()).collect()
    }

    /// `place` as one view, copying parts into scratch when needed.
    fn materialize(&mut self, place: Place, elem: Elem) -> Result<Loc, BoxError> {
        match place {
            Place::View(l) => Ok(l),
            Place::Parts { axis, parts } => {
                let mut shape = parts
                    .first()
                    .ok_or_else(|| err("empty parts"))?
                    .shape
                    .clone();
                shape[axis] = parts.iter().map(|p| p.shape[axis]).sum();
                let dst = self.new_scratch(elem, &shape);
                self.copy_parts(elem, axis, &parts, &dst)?;
                Ok(dst)
            }
        }
    }

    fn copy_parts(
        &mut self,
        elem: Elem,
        axis: usize,
        parts: &[Loc],
        dst: &Loc,
    ) -> Result<(), BoxError> {
        let mut at = 0;
        for p in parts {
            let n = p.shape[axis];
            let d = dst.narrow(axis, at, n).ok_or_else(|| err("bad part"))?;
            if *p != d {
                self.steps.push(Step::Copy {
                    elem,
                    src: p.clone(),
                    dst: d,
                });
            }
            at += n;
        }
        Ok(())
    }

    /// Writes `src` to `t`'s location when it already has one (a partition output, or a
    /// placement from a consumer); otherwise `t` becomes `src`.
    fn define(&mut self, t: Tid, src: Place, elem: Elem) -> Result<(), BoxError> {
        let target = match self.loc.get(&t) {
            Some(Place::View(l)) => Some(l.clone()),
            Some(Place::Parts { .. }) => return Err(err("tensor defined twice")),
            None => self
                .out_idx
                .get(&t)
                .map(|&k| Loc::dense(Buf::Output(k), &src_shape(&src))),
        };
        match target {
            None => {
                self.loc.insert(t, src);
            }
            Some(dst) => {
                match src {
                    Place::View(s) => {
                        if s != dst {
                            self.steps.push(Step::Copy {
                                elem,
                                src: s,
                                dst: dst.clone(),
                            });
                        }
                    }
                    Place::Parts { axis, parts } => self.copy_parts(elem, axis, &parts, &dst)?,
                }
                self.loc.insert(t, Place::View(dst));
            }
        }
        Ok(())
    }

    fn view_op(&mut self, ni: usize) -> Result<(), BoxError> {
        let node = self.g.nodes[ni].clone();
        let (x, out) = (
            node.input(0).ok_or_else(|| err("no input"))?,
            node.output().ok_or_else(|| err("no output"))?,
        );
        let elem = self.elem(x);
        if self.placed_through.contains(&ni) {
            return Ok(());
        }
        let src = self.input(x)?;
        let out_shape = self.shape(out)?;
        let perm = self.perm(&node.inputs);
        let begin: Option<Vec<usize>> = node
            .input(1)
            .and_then(|t| self.g.tensor(t))
            .and_then(|t| t.i32s())
            .and_then(|b| b.iter().map(|&v| usize::try_from(v).ok()).collect());
        let op = node.op;
        let view = |l: &Loc| -> Option<Loc> {
            match op {
                Op::Transpose => perm.as_ref().and_then(|p| l.transpose(p)),
                Op::Reshape => l.reshape(&out_shape),
                Op::Slice => begin.as_ref().and_then(|b| l.slice(b, &out_shape)),
                _ => None,
            }
        };
        let result = match &src {
            Place::View(l) => view(l).map(Place::View),
            Place::Parts { axis, parts } => {
                self.parts_view(&node.op, &node.inputs, *axis, parts, &out_shape)
            }
        };
        // Where the output is headed through further views (a partition output), if anywhere.
        let target = if self.loc.contains_key(&out) {
            None
        } else {
            self.lookahead(out)
        };
        let in_shape = self.shape(x)?;
        let result = match (result, target) {
            (Some(r), _) => r,
            (None, Some(dst)) => {
                // Copy straight into the output, mapping its location back through this view.
                let inverse = match op {
                    Op::Transpose => perm.as_ref().and_then(|p| {
                        let mut inv = vec![0; p.len()];
                        for (a, &q) in p.iter().enumerate() {
                            *inv.get_mut(q)? = a;
                        }
                        dst.transpose(&inv)
                    }),
                    Op::Reshape => dst.reshape(&in_shape),
                    _ => None,
                };
                let l = self.materialize(src.clone(), elem)?;
                if let Some(d) = inverse {
                    self.steps.push(Step::Copy {
                        elem,
                        src: l,
                        dst: d,
                    });
                    return Ok(());
                }
                let dense = self.new_scratch(elem, &l.shape);
                self.steps.push(Step::Copy {
                    elem,
                    src: l,
                    dst: dense.clone(),
                });
                Place::View(view(&dense).ok_or_else(|| err("view op failed"))?)
            }
            (None, None) => {
                let l = self.materialize(src, elem)?;
                if let Some(v) = view(&l) {
                    Place::View(v)
                } else {
                    // Only a reshape of a strided view gets here: copy it dense first.
                    let dense = self.new_scratch(elem, &l.shape);
                    self.steps.push(Step::Copy {
                        elem,
                        src: l,
                        dst: dense.clone(),
                    });
                    Place::View(view(&dense).ok_or_else(|| err("view op failed"))?)
                }
            }
        };
        self.define(out, result, elem)
    }

    /// A view operator applied part by part, when the concatenation axis survives it.
    fn parts_view(
        &self,
        op: &Op,
        inputs: &[Option<Tid>],
        axis: usize,
        parts: &[Loc],
        out_shape: &[usize],
    ) -> Option<Place> {
        match op {
            Op::Transpose => {
                let perm = self.perm(inputs)?;
                let new_axis = perm.iter().position(|&p| p == axis)?;
                let parts = parts
                    .iter()
                    .map(|p| p.transpose(&perm))
                    .collect::<Option<_>>()?;
                Some(Place::Parts {
                    axis: new_axis,
                    parts,
                })
            }
            Op::Reshape => {
                let full = parts.first()?;
                let mut in_shape = full.shape.clone();
                in_shape[axis] = parts.iter().map(|p| p.shape[axis]).sum();
                let before: usize = in_shape[..axis].iter().product();
                let after: usize = in_shape[axis + 1..].iter().product();
                let new_axis = (0..out_shape.len()).find(|&a| {
                    out_shape[..a].iter().product::<usize>() == before
                        && out_shape[a] == in_shape[axis]
                        && out_shape[a + 1..].iter().product::<usize>() == after
                })?;
                let parts = parts
                    .iter()
                    .map(|p| {
                        let mut s = out_shape.to_vec();
                        s[new_axis] = p.shape[axis];
                        p.reshape(&s)
                    })
                    .collect::<Option<_>>()?;
                Some(Place::Parts {
                    axis: new_axis,
                    parts,
                })
            }
            Op::Slice => {
                let b = self.g.tensor(inputs.get(1).copied()??)?.i32s()?;
                let begin: Vec<usize> = b
                    .iter()
                    .map(|&v| usize::try_from(v).ok())
                    .collect::<Option<_>>()?;
                let (lo, hi) = (begin[axis], begin[axis] + out_shape[axis]);
                let mut out = Vec::new();
                let mut at = 0;
                for p in parts {
                    let n = p.shape[axis];
                    let (s, e) = (lo.max(at), hi.min(at + n));
                    if s < e {
                        let mut pb = begin.clone();
                        let mut ps = out_shape.to_vec();
                        pb[axis] = s - at;
                        ps[axis] = e - s;
                        out.push(p.slice(&pb, &ps)?);
                    }
                    at += n;
                }
                if out.len() == 1 {
                    out.pop().map(Place::View)
                } else {
                    Some(Place::Parts { axis, parts: out })
                }
            }
            _ => None,
        }
    }

    fn concat(&mut self, ni: usize) -> Result<(), BoxError> {
        let node = self.g.nodes[ni].clone();
        let Op::Concat { axis, .. } = node.op else {
            return Err(err("not a concatenation"));
        };
        let out = node.output().ok_or_else(|| err("no output"))?;
        let shape = self.shape(out)?;
        let axis = norm_axis(axis, shape.len()).ok_or_else(|| err("bad axis"))?;
        let elem = self.elem(out);
        let mut parts = Vec::new();
        for x in node.inputs.iter().flatten() {
            match self.input(*x)? {
                Place::View(l) => parts.push(l),
                Place::Parts { axis: a, parts: p } if a == axis => parts.extend(p),
                other @ Place::Parts { .. } => parts.push(self.materialize(other, elem)?),
            }
        }
        self.define(out, Place::Parts { axis, parts }, elem)
    }

    /// Runs `f(src part, dst part)` for an elementwise or row operator over every part.
    fn per_part(
        &mut self,
        src: Place,
        dst: &Loc,
        rows: bool,
        elem: Elem,
        mut f: impl FnMut(Loc, Loc) -> Step,
    ) -> Result<(), BoxError> {
        let src = match src {
            Place::Parts { axis, .. } if rows && axis + 1 == dst.shape.len() => {
                Place::View(self.materialize(src, elem)?)
            }
            s => s,
        };
        match src {
            Place::View(s) => self.steps.push(f(s, dst.clone())),
            Place::Parts { axis, parts } => {
                let mut at = 0;
                for p in parts {
                    let n = p.shape[axis];
                    let d = dst.narrow(axis, at, n).ok_or_else(|| err("bad part"))?;
                    self.steps.push(f(p, d));
                    at += n;
                }
            }
        }
        Ok(())
    }

    fn softmax(&mut self, ni: usize) -> Result<(), BoxError> {
        let node = self.g.nodes[ni].clone();
        let Op::Softmax { beta } = node.op else {
            return Err(err("not a softmax"));
        };
        let (x, out) = (
            node.input(0).ok_or_else(|| err("no input"))?,
            node.output().ok_or_else(|| err("no output"))?,
        );
        let (iq, oq) = (self.quant(x)?, self.quant(out)?);
        let elem = self.elem(x);
        let src = self.input(x)?;
        let dst = self.output(out)?;
        let f = self.o.features;
        match elem {
            Elem::I8 => {
                let plan = SoftmaxI8Plan::with_features(
                    SoftmaxI8Params {
                        input: iq,
                        output: oq,
                        beta,
                    },
                    f,
                )?;
                let plan = Arc::new(plan);
                self.per_part(src, &dst, true, elem, |s, d| Step::Softmax8 {
                    plan: Arc::clone(&plan),
                    src: s,
                    dst: d,
                })
            }
            Elem::I16 => {
                let plan = SoftmaxI16Plan::with_features(
                    SoftmaxI16Params {
                        input: iq,
                        output: oq,
                        beta,
                    },
                    f,
                )?;
                let plan = Arc::new(plan);
                self.per_part(src, &dst, true, elem, |s, d| Step::Softmax16 {
                    plan: Arc::clone(&plan),
                    src: s,
                    dst: d,
                })
            }
            _ => Err(err("softmax type")),
        }
    }

    fn logistic(&mut self, ni: usize) -> Result<(), BoxError> {
        let node = self.g.nodes[ni].clone();
        let (x, out) = (
            node.input(0).ok_or_else(|| err("no input"))?,
            node.output().ok_or_else(|| err("no output"))?,
        );
        let plan = LutI8Plan::sigmoid_with_features(
            DType::I8,
            self.quant(x)?,
            self.quant(out)?,
            self.o.features,
        )?;
        let src = self.input(x)?;
        let dst = self.output(out)?;
        let plan = Arc::new(plan);
        self.per_part(src, &dst, false, Elem::I8, |s, d| Step::Lut {
            plan: Arc::clone(&plan),
            src: s,
            dst: d,
        })
    }

    fn binary_params(&self, ni: usize) -> Result<BinaryI8Params, BoxError> {
        let node = &self.g.nodes[ni];
        let (op, activation) = match node.op {
            Op::Add(a) => (BinaryOp::Add, a),
            Op::Sub(a) => (BinaryOp::Sub, a),
            Op::Mul(a) => (BinaryOp::Mul, a),
            _ => return Err(err("not a binary op")),
        };
        Ok(BinaryI8Params {
            op,
            lhs: self.quant(node.input(0).ok_or_else(|| err("no lhs"))?)?,
            rhs: self.quant(node.input(1).ok_or_else(|| err("no rhs"))?)?,
            output: self.quant(node.output().ok_or_else(|| err("no output"))?)?,
            activation,
            kernels: self.o.kernels,
        })
    }

    /// The table of a `MUL` by a scalar constant, as a function of the other operand.
    fn scalar_mul_table(&self, ni: usize) -> Result<Option<(usize, [u8; 256])>, BoxError> {
        let Some((c, other)) = attention::scalar_operand(self.g, ni) else {
            return Ok(None);
        };
        let node = &self.g.nodes[ni];
        let mut p = self.binary_params(ni)?;
        if c == 0 {
            // MUL is symmetric: put the constant on the right.
            core::mem::swap(&mut p.lhs, &mut p.rhs);
        }
        let value = match &self
            .g
            .tensor(node.input(c).ok_or_else(|| err("no const"))?)
            .and_then(|t| t.constant.clone())
        {
            Some(Const::I8(v)) => *v.first().ok_or_else(|| err("empty const"))?,
            _ => return Err(err("scalar constant is not int8")),
        };
        let plan = BinaryI8Plan::with_features(p, self.o.features)?;
        // TFLite's scalar `MUL` runs its blocks and tail over the other operand's elements.
        let len = self
            .shape(node.input(other).ok_or_else(|| err("no input"))?)?
            .iter()
            .product();
        Ok(plan.table_for_rhs(value, len).map(|t| (other, t)))
    }

    fn binary(&mut self, ni: usize) -> Result<(), BoxError> {
        let node = self.g.nodes[ni].clone();
        let out = node.output().ok_or_else(|| err("no output"))?;
        let shape = self.shape(out)?;
        if matches!(node.op, Op::Mul(_)) {
            if let Some((other, table)) = self.scalar_mul_table(ni)? {
                let plan = LutI8Plan::from_table_with_features(DType::I8, table, self.o.features)?;
                let src = self.input(node.input(other).ok_or_else(|| err("no input"))?)?;
                let src = match src {
                    Place::View(l) => Place::View(
                        l.reshape(&shape)
                            .or_else(|| l.broadcast_to(&shape))
                            .ok_or_else(|| err("shape"))?,
                    ),
                    p @ Place::Parts { .. } => p,
                };
                let dst = self.output(out)?;
                let plan = Arc::new(plan);
                return self.per_part(src, &dst, false, Elem::I8, |s, d| Step::Lut {
                    plan: Arc::clone(&plan),
                    src: s,
                    dst: d,
                });
            }
        }
        let plan = BinaryI8Plan::with_features(self.binary_params(ni)?, self.o.features)?;
        let lhs = self.input(node.input(0).ok_or_else(|| err("no lhs"))?)?;
        let rhs = self.input(node.input(1).ok_or_else(|| err("no rhs"))?)?;
        let dst = self.output(out)?;
        // TFLite's NEON `ADD` and `MUL` run whole 16-element blocks over the output, and each
        // call here takes its own tail: split along parts only if every part is whole blocks.
        let blocks = self.o.kernels.neon_blocks() && !matches!(node.op, Op::Sub(_));
        let whole_blocks = |axis: usize, n: usize| {
            let len: usize = shape
                .iter()
                .enumerate()
                .map(|(a, &s)| if a == axis { n } else { s })
                .product();
            len.is_multiple_of(16)
        };
        // Follow the lhs parts when they have the output's rank; a parted rhs is made one view.
        match lhs {
            Place::Parts { axis, parts }
                if parts[0].shape.len() == shape.len()
                    && (!blocks || parts.iter().all(|p| whole_blocks(axis, p.shape[axis]))) =>
            {
                let rhs = self.materialize(rhs, Elem::I8)?;
                let rhs = rhs.broadcast_to(&shape).ok_or_else(|| err("rhs shape"))?;
                let mut at = 0;
                for p in parts {
                    let n = p.shape[axis];
                    let r = rhs.narrow(axis, at, n).ok_or_else(|| err("rhs part"))?;
                    let d = dst.narrow(axis, at, n).ok_or_else(|| err("dst part"))?;
                    let l = p.broadcast_to(&d.shape).ok_or_else(|| err("lhs part"))?;
                    self.steps.push(Step::Binary {
                        plan: plan.clone(),
                        lhs: l,
                        rhs: r,
                        dst: d,
                    });
                    at += n;
                }
                Ok(())
            }
            lhs => {
                let l = self.materialize(lhs, Elem::I8)?;
                let r = self.materialize(rhs, Elem::I8)?;
                let l = l.broadcast_to(&shape).ok_or_else(|| err("lhs shape"))?;
                let r = r.broadcast_to(&shape).ok_or_else(|| err("rhs shape"))?;
                self.steps.push(Step::Binary {
                    plan,
                    lhs: l,
                    rhs: r,
                    dst,
                });
                Ok(())
            }
        }
    }

    /// `(A, B)` of a `BATCH_MATMUL` with the adjoints applied, extended to the output's rank.
    fn matmul_operands(&mut self, ni: usize) -> Result<(Loc, Loc), BoxError> {
        let node = self.g.nodes[ni].clone();
        let Op::BatchMatMul { adj_x, adj_y } = node.op else {
            return Err(err("not a matmul"));
        };
        let out = node.output().ok_or_else(|| err("no output"))?;
        let shape = self.shape(out)?;
        let a = self.input(node.input(0).ok_or_else(|| err("no lhs"))?)?;
        let a = self.materialize(a, Elem::I8)?;
        let b = self.input(node.input(1).ok_or_else(|| err("no rhs"))?)?;
        let b = self.materialize(b, Elem::I8)?;
        let a = if adj_x { a.swap_last2() } else { Some(a) }.ok_or_else(|| err("lhs rank"))?;
        let b = if adj_y { b.swap_last2() } else { Some(b) }.ok_or_else(|| err("rhs rank"))?;
        let r = shape.len();
        let (m, n) = (shape[r - 2], shape[r - 1]);
        let k = *a.shape.last().ok_or_else(|| err("lhs rank"))?;
        let mut sa = shape[..r - 2].to_vec();
        sa.extend([m, k]);
        let mut sb = shape[..r - 2].to_vec();
        sb.extend([k, n]);
        let a = a.broadcast_to(&sa).ok_or_else(|| err("lhs batch"))?;
        let b = b.broadcast_to(&sb).ok_or_else(|| err("rhs batch"))?;
        Ok((a, b))
    }

    fn matmul(&mut self, ni: usize) -> Result<(), BoxError> {
        let node = self.g.nodes[ni].clone();
        let out = node.output().ok_or_else(|| err("no output"))?;
        let (lq, rq, oq) = (
            self.quant(node.input(0).ok_or_else(|| err("no lhs"))?)?,
            self.quant(node.input(1).ok_or_else(|| err("no rhs"))?)?,
            self.quant(out)?,
        );
        let plan = MatMulI8Plan::with_features(
            MatMulI8Params::tflite_batch_matmul(lq, rq, oq, self.o.kernels)?,
            self.o.features,
        )?;
        let (a, b) = self.matmul_operands(ni)?;
        let dst = self.output(out)?;
        self.steps.push(Step::MatMul {
            plan,
            lhs: a,
            rhs: b,
            dst,
        });
        Ok(())
    }

    fn conv(&mut self, ni: usize) -> Result<(), BoxError> {
        let node = self.g.nodes[ni].clone();
        let Op::Conv2d { act, .. } = node.op else {
            return Err(err("not a conv"));
        };
        let (x, w, out) = (
            node.input(0).ok_or_else(|| err("no input"))?,
            node.input(1).ok_or_else(|| err("no filter"))?,
            node.output().ok_or_else(|| err("no output"))?,
        );
        let wt = self.g.tensor(w).ok_or_else(|| err("filter"))?;
        let scales = filter_scales(wt).ok_or_else(|| err("filter scales"))?;
        let bias = match node
            .input(2)
            .and_then(|t| self.g.tensor(t))
            .and_then(|t| t.i32s())
        {
            Some(b) => b.to_vec(),
            None => Vec::new(),
        };
        let (cout, cin) = (wt.shape[0], wt.shape[3]);
        let plan = MatMulI8Plan::with_features(
            MatMulI8Params::tflite_conv(
                self.quant(x)?,
                &scales,
                self.quant(out)?,
                bias,
                act,
                self.o.kernels,
            )?,
            self.o.features,
        )?;
        let xs = self.shape(x)?;
        let pixels = xs[0] * xs[1] * xs[2];
        let src = self.input(x)?;
        let src = self.materialize(src, Elem::I8)?;
        let lhs = if let Some(l) = src.reshape(&[pixels, cin]) {
            l
        } else {
            let dense = self.new_scratch(Elem::I8, &src.shape);
            self.steps.push(Step::Copy {
                elem: Elem::I8,
                src,
                dst: dense.clone(),
            });
            dense
                .reshape(&[pixels, cin])
                .ok_or_else(|| err("conv input"))?
        };
        let wl = match self.input(w)? {
            Place::View(l) => l,
            Place::Parts { .. } => return Err(err("parted filter")),
        };
        let rhs = wl
            .reshape(&[cout, cin])
            .and_then(|l| l.swap_last2())
            .ok_or_else(|| err("filter view"))?;
        let dst = self.output(out)?;
        self.write_via(Elem::I8, dst, &[pixels, cout], |d| Step::MatMul {
            plan,
            lhs,
            rhs,
            dst: d,
        })
    }

    /// Emits `make(target)` with `target` the reshape of `dst` to `shape`, or into dense
    /// scratch followed by a copy when that reshape is not a view.
    fn write_via(
        &mut self,
        elem: Elem,
        dst: Loc,
        shape: &[usize],
        make: impl FnOnce(Loc) -> Step,
    ) -> Result<(), BoxError> {
        if let Some(d) = dst.reshape(shape) {
            self.steps.push(make(d));
            return Ok(());
        }
        let tmp = self.new_scratch(elem, &dst.shape);
        let d = tmp.reshape(shape).ok_or_else(|| err("scratch reshape"))?;
        self.steps.push(make(d));
        self.steps.push(Step::Copy {
            elem,
            src: tmp,
            dst,
        });
        Ok(())
    }

    /// The fused step of `a`, or `None` (nothing emitted) when the views do not fold.
    fn try_attention(&mut self, a: &Attention) -> Result<bool, BoxError> {
        let g = self.g;
        let b1 = &g.nodes[a.scores];
        let b2 = &g.nodes[a.output];
        let s = &g.nodes[a.softmax];
        let Op::Softmax { beta } = s.op else {
            return Ok(false);
        };
        let Op::BatchMatMul { adj_x, adj_y } = b1.op else {
            return Ok(false);
        };
        let Op::BatchMatMul {
            adj_x: x2,
            adj_y: y2,
        } = b2.op
        else {
            return Ok(false);
        };
        let (Some(ts), Some(tp), Some(to)) = (s.input(0), s.output(), b2.output()) else {
            return Ok(false);
        };
        // Q: the MUL's input when it folds, else the first operand of the scores.
        let (q_src, q_mul) = match a.mul {
            Some(m) => {
                let Some((_, other)) = attention::scalar_operand(g, m) else {
                    return Ok(false);
                };
                let Some((_, table)) = self.scalar_mul_table(m)? else {
                    return Ok(false);
                };
                let mn = &g.nodes[m];
                let out_q = self.quant(mn.output().ok_or_else(|| err("mul out"))?)?;
                (
                    mn.input(other).ok_or_else(|| err("mul in"))?,
                    Some(AttentionQMul {
                        table,
                        output: out_q,
                    }),
                )
            }
            _ => (b1.input(0).ok_or_else(|| err("lhs"))?, None),
        };
        let qt = self.input(q_src)?;
        let qt = self.materialize(qt, Elem::I8)?;
        let kt = self.input(b1.input(1).ok_or_else(|| err("rhs"))?)?;
        let kt = self.materialize(kt, Elem::I8)?;
        let q = if adj_x { qt.swap_last2() } else { Some(qt) };
        // B = adj_y ? Kᵀ : K is `[.., dk, tokens]`; K itself is its transpose.
        let k = if adj_y { Some(kt) } else { kt.swap_last2() };
        let (p_pos, v_pos) = if a.p_right { (1, 0) } else { (0, 1) };
        debug_assert_eq!(b2.input(p_pos), a.p_operand(g));
        let vt = self.input(b2.input(v_pos).ok_or_else(|| err("values"))?)?;
        let vt = self.materialize(vt, Elem::I8)?;
        let v = if a.p_right {
            // O' = A·Pᵀ with A = adj_x ? vᵀ : v of shape `[.., dv, tokens]`.
            if x2 {
                Some(vt)
            } else {
                vt.swap_last2()
            }
        } else if y2 {
            vt.swap_last2()
        } else {
            Some(vt)
        };
        let dst = self.output(to)?;
        let out = if a.p_right {
            dst.swap_last2()
        } else {
            Some(dst)
        };
        let (Some(q), Some(k), Some(v), Some(out)) = (q, k, v, out) else {
            return Ok(false);
        };
        // Fold the batch axes into one head axis.
        let heads3 = |l: &Loc| -> Option<Loc> {
            let r = l.shape.len();
            let heads: usize = l.shape[..r - 2].iter().product();
            l.reshape(&[heads, l.shape[r - 2], l.shape[r - 1]])
        };
        let (Some(q), Some(k), Some(v), Some(out)) =
            (heads3(&q), heads3(&k), heads3(&v), heads3(&out))
        else {
            return Ok(false);
        };
        let (heads, queries, dk) = (q.shape[0], q.shape[1], q.shape[2]);
        let (tokens, dv) = (k.shape[1], v.shape[2]);
        if k.shape != [heads, tokens, dk]
            || v.shape != [heads, tokens, dv]
            || out.shape != [heads, queries, dv]
        {
            return Ok(false);
        }
        let params = AttentionI8Params {
            q: self.quant(q_src)?,
            q_mul,
            k: self.quant(b1.input(1).ok_or_else(|| err("rhs"))?)?,
            v: self.quant(b2.input(v_pos).ok_or_else(|| err("values"))?)?,
            scores: self.quant(ts)?,
            beta,
            probs: self.quant(tp)?,
            output: self.quant(to)?,
            kernels: self.o.kernels,
        };
        // A block the fused kernel rejects runs operator by operator.
        let Ok(plan) = AttentionI8Plan::with_features(params, self.o.features) else {
            return Ok(false);
        };
        let Ok(scratch) = plan.scratch(heads, queries, tokens, dk, dv) else {
            return Ok(false);
        };
        let idx = self.attention.len();
        self.attention.push(scratch);
        self.steps.push(Step::Attention {
            plan: Box::new(plan),
            scratch: idx,
            q,
            k,
            v,
            dst: out,
        });
        self.fused_attention += 1;
        Ok(true)
    }

    /// The fused step of `d`, or `None` (nothing emitted) when it does not apply.
    fn try_softmax_dot(&mut self, d: &SoftmaxDot) -> Result<bool, BoxError> {
        let g = self.g;
        let (s, c) = (&g.nodes[d.softmax], &g.nodes[d.conv]);
        let (Op::Softmax { beta }, Op::Conv2d { act, .. }) = (s.op, c.op) else {
            return Ok(false);
        };
        let (Some(x), Some(probs), Some(w), Some(out)) =
            (s.input(0), s.output(), c.input(1), c.output())
        else {
            return Ok(false);
        };
        let wt = g.tensor(w).ok_or_else(|| err("filter"))?;
        let (Some(Const::I8(weights)), Some(scales)) = (&wt.constant, filter_scales(wt)) else {
            return Ok(false);
        };
        let bias = match c.input(2).and_then(|t| g.tensor(t)).and_then(|t| t.i32s()) {
            Some(b) => b.to_vec(),
            None => Vec::new(),
        };
        let pq = self.quant(probs)?;
        let Ok(projection) =
            MatMulI8Params::tflite_conv(pq, &scales, self.quant(out)?, bias, act, self.o.kernels)
        else {
            return Ok(false);
        };
        let params = SoftmaxDotI8Params {
            softmax: SoftmaxI8Params {
                input: self.quant(x)?,
                output: pq,
                beta,
            },
            weights: weights.clone(),
            projection,
        };
        let Ok(plan) = SoftmaxDotI8Plan::with_features(params, self.o.features) else {
            return Ok(false);
        };
        // The convolution's `[.., 1]` output, reshaped to the softmax's rows.
        let xs = self.shape(x)?;
        let mut rows = xs.clone();
        if let Some(last) = rows.last_mut() {
            *last = 1;
        }
        let mut src = self.input(x)?;
        let unit_rows = match &src {
            Place::View(l) => l.strides.last() == Some(&1),
            Place::Parts { axis, parts } => {
                *axis + 1 != xs.len() && parts.iter().all(|l| l.strides.last() == Some(&1))
            }
        };
        if !unit_rows {
            // The fused kernel reads each row of bins contiguously. An NCHW head transposes
            // the bins onto a strided axis, so gather them once into a dense copy.
            let dense = self.new_scratch(Elem::I8, &xs);
            match src {
                Place::View(l) => self.steps.push(Step::Copy {
                    elem: Elem::I8,
                    src: l,
                    dst: dense.clone(),
                }),
                Place::Parts { axis, parts } => self.copy_parts(Elem::I8, axis, &parts, &dense)?,
            }
            src = Place::View(dense);
        }
        let Some(dst) = self.output(out)?.reshape(&rows) else {
            return Ok(false);
        };
        let plan = Arc::new(plan);
        self.per_part(src, &dst, true, Elem::I8, |s, d| Step::SoftmaxDot {
            plan: Arc::clone(&plan),
            src: s,
            dst: d,
        })?;
        self.fused_softmax_dot += 1;
        Ok(true)
    }

    fn node(&mut self, ni: usize) -> Result<(), BoxError> {
        match self.g.nodes[ni].op {
            Op::Transpose | Op::Reshape | Op::Slice => self.view_op(ni),
            Op::Concat { .. } => self.concat(ni),
            Op::Softmax { .. } => self.softmax(ni),
            Op::Logistic => self.logistic(ni),
            Op::Add(_) | Op::Sub(_) | Op::Mul(_) => self.binary(ni),
            Op::BatchMatMul { .. } => self.matmul(ni),
            Op::Conv2d { .. } => self.conv(ni),
            Op::Other => Err(err("unsupported node claimed")),
        }
    }
}

fn src_shape(p: &Place) -> Vec<usize> {
    match p {
        Place::View(l) => l.shape.clone(),
        Place::Parts { axis, parts } => {
            let mut s = parts.first().map(|p| p.shape.clone()).unwrap_or_default();
            if let Some(a) = s.get_mut(*axis) {
                *a = parts.iter().map(|p| p.shape[*axis]).sum();
            }
            s
        }
    }
}

fn norm_axis(axis: i32, rank: usize) -> Option<usize> {
    let a = if axis < 0 {
        i32::try_from(rank).ok()? + axis
    } else {
        axis
    };
    usize::try_from(a).ok().filter(|&a| a < rank)
}

/// Plans the partition of `nodes` (indices into `g.nodes`, in execution order) with the
/// given boundary tensors.
pub(crate) fn plan(
    g: &Graph,
    nodes: &[usize],
    inputs: &[Tid],
    outputs: &[Tid],
    o: PlanOptions,
) -> Result<Program, BoxError> {
    let mut p = Planner {
        g,
        o,
        nodes: nodes.iter().copied().collect(),
        in_idx: inputs.iter().enumerate().map(|(i, &t)| (t, i)).collect(),
        out_idx: outputs.iter().enumerate().map(|(i, &t)| (t, i)).collect(),
        loc: HashMap::new(),
        placed_through: HashSet::new(),
        steps: Vec::new(),
        scratch: Vec::new(),
        attention: Vec::new(),
        fused_attention: 0,
        fused_softmax_dot: 0,
    };
    let in_part = |i: usize| p.nodes.contains(&i);
    let blocks: Vec<Attention> = attention::find(g, &in_part)
        .into_iter()
        .filter(|a| !a.inner_tensors(g).any(|t| outputs.contains(&t)))
        .collect();
    let mut deferred: HashMap<usize, Attention> = HashMap::new();
    let mut skip: HashSet<usize> = HashSet::new();
    for a in blocks {
        skip.extend(a.nodes().filter(|&n| n != a.output));
        deferred.insert(a.output, a);
    }
    let dots: Vec<SoftmaxDot> = softmax_dot::find(g, &in_part)
        .into_iter()
        .filter(|d| {
            !d.inner_tensors(g).any(|t| outputs.contains(&t))
                && !skip.contains(&d.softmax)
                && !deferred.contains_key(&d.conv)
        })
        .collect();
    let mut deferred_dot: HashMap<usize, &SoftmaxDot> = HashMap::new();
    for d in &dots {
        deferred_dot.insert(d.conv, d);
        skip.insert(d.softmax);
        skip.extend(d.views.iter().copied());
    }
    for &ni in nodes {
        if skip.contains(&ni) {
            continue;
        }
        if let Some(d) = deferred_dot.get(&ni).copied() {
            if p.try_softmax_dot(d)? {
                continue;
            }
            for n in d.nodes() {
                p.node(n)?;
            }
            continue;
        }
        if let Some(a) = deferred.get(&ni).cloned() {
            if p.try_attention(&a)? {
                continue;
            }
            // The views do not fold: run the block's nodes one by one, here.
            for n in a.nodes() {
                p.node(n)?;
            }
            continue;
        }
        p.node(ni)?;
    }
    // Every partition output must have been written.
    for &t in outputs {
        match p.loc.get(&t) {
            Some(Place::View(l)) if matches!(l.buf, Buf::Output(_)) => {}
            Some(other) => {
                let other = other.clone();
                let elem = p.elem(t);
                let dst = Loc::dense(Buf::Output(p.out_idx[&t]), &p.shape(t)?);
                p.loc.remove(&t);
                p.loc.insert(t, Place::View(dst.clone()));
                match other {
                    Place::View(s) => p.steps.push(Step::Copy { elem, src: s, dst }),
                    Place::Parts { axis, parts } => p.copy_parts(elem, axis, &parts, &dst)?,
                }
            }
            None => return Err(err("partition output never written")),
        }
    }
    let tensor_io = |ts: &[Tid]| -> Result<Vec<(Elem, Vec<usize>)>, BoxError> {
        ts.iter()
            .map(|&t| {
                let x = g.tensor(t).ok_or_else(|| err("unknown tensor"))?;
                Ok((x.elem, x.shape.clone()))
            })
            .collect()
    };
    Ok(Program {
        steps: p.steps,
        scratch: p.scratch,
        attention: p.attention,
        inputs: tensor_io(inputs)?,
        outputs: tensor_io(outputs)?,
        fused_attention: p.fused_attention,
        fused_softmax_dot: p.fused_softmax_dot,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Node, Tensor};
    use edgefirst_kernels::{Activation, TensorMut, TensorRef, View, ViewMut};

    fn t(shape: &[usize], q: Option<(f32, i32)>) -> Tensor {
        Tensor {
            elem: Elem::I8,
            shape: shape.to_vec(),
            quant: q.map(|(s, z)| Quant::new(s, z)),
            axis_scales: None,
            constant: None,
        }
    }

    fn c32(v: &[i32]) -> Tensor {
        Tensor {
            elem: Elem::I32,
            shape: vec![v.len()],
            quant: None,
            axis_scales: None,
            constant: Some(Const::I32(v.to_vec())),
        }
    }

    fn node(index: usize, op: Op, inputs: &[usize], out: usize) -> Node {
        Node {
            index,
            op,
            inputs: inputs.iter().map(|&i| Some(i)).collect(),
            outputs: vec![Some(out)],
        }
    }

    /// A `BATCH_MATMUL` whose multiplier ruy cannot round (1.0) is declined under ruy's kernels
    /// and claimed under the reference ones, which plan it: a claimed node always plans.
    #[test]
    fn batch_matmul_is_claimed_only_where_it_plans() {
        let unit = Some((1.0, 0));
        let op = Op::BatchMatMul {
            adj_x: false,
            adj_y: false,
        };
        let g = Graph {
            tensors: [
                (0, t(&[1, 4, 8], unit)),
                (1, t(&[1, 8, 4], unit)),
                (2, t(&[1, 4, 4], unit)),
            ]
            .into_iter()
            .collect(),
            nodes: vec![node(0, op, &[0, 1], 2)],
            ..Graph::default()
        };
        let f = CpuFeatures::default();
        for kernels in [TfliteKernels::NeonRuy, TfliteKernels::Ruy] {
            assert!(crate::select::support(&g, &g.nodes[0], f, kernels).is_none());
        }
        for kernels in [TfliteKernels::NeonGemmlowp, TfliteKernels::Reference] {
            assert!(crate::select::support(&g, &g.nodes[0], f, kernels).is_some());
            let o = PlanOptions {
                features: f,
                kernels,
            };
            plan(&g, &[0], &[0, 1], &[2], o).unwrap();
        }
    }

    /// yolo26n's first attention block, from the TRANSPOSE after Neutron to the two RESHAPEs
    /// feeding Neutron again.
    fn yolo26_block() -> (Graph, Vec<usize>, Vec<usize>) {
        let x = Some((0.1251, -15));
        let mut g = Graph::default();
        let tensors = [
            (40, t(&[400, 256], x)),
            (42, c32(&[1, 0])),
            (41, t(&[256, 400], x)),
            (55, c32(&[1, 2, 128, 400])),
            (54, t(&[1, 2, 128, 400], x)),
            (21, c32(&[0, 0, 0, 0])),
            (22, c32(&[1, 2, 32, 400])),
            (24, c32(&[0, 0, 32, 0])),
            (25, c32(&[1, 2, 32, 400])),
            (27, c32(&[0, 0, 64, 0])),
            (28, c32(&[1, 2, 64, 400])),
            (20, t(&[1, 2, 32, 400], x)),
            (23, t(&[1, 2, 32, 400], x)),
            (26, t(&[1, 2, 64, 400], x)),
            (
                83,
                Tensor {
                    constant: Some(Const::I8(vec![127])),
                    ..t(&[], Some((0.000_693_2, -128)))
                },
            ),
            (3, t(&[1, 2, 32, 400], Some((0.010_18, 6)))),
            (61, c32(&[128, 400])),
            (60, t(&[128, 400], x)),
            (46, c32(&[1, 0])),
            (45, t(&[400, 128], x)),
            (75, c32(&[1, 20, 20, 128])),
            (76, t(&[1, 20, 20, 128], x)),
            (4, t(&[1, 2, 400, 400], Some((0.1279, -9)))),
            (5, t(&[1, 2, 400, 400], Some((1.0 / 256.0, -128)))),
            (6, t(&[1, 2, 64, 400], Some((0.063_33, 2)))),
            (59, c32(&[128, 400])),
            (58, t(&[128, 400], Some((0.063_33, 2)))),
            (44, c32(&[1, 0])),
            (43, t(&[400, 128], Some((0.063_33, 2)))),
            (77, c32(&[1, 20, 20, 128])),
            (78, t(&[1, 20, 20, 128], Some((0.063_33, 2)))),
        ];
        g.tensors = tensors.into_iter().collect();
        g.nodes = vec![
            node(1, Op::Transpose, &[40, 42], 41),
            node(2, Op::Reshape, &[41, 55], 54),
            node(3, Op::Slice, &[54, 21, 22], 20),
            node(4, Op::Slice, &[54, 24, 25], 23),
            node(5, Op::Slice, &[54, 27, 28], 26),
            node(6, Op::Mul(Activation::None), &[20, 83], 3),
            node(7, Op::Reshape, &[26, 61], 60),
            node(8, Op::Transpose, &[60, 46], 45),
            node(9, Op::Reshape, &[45, 75], 76),
            node(
                10,
                Op::BatchMatMul {
                    adj_x: true,
                    adj_y: false,
                },
                &[3, 23],
                4,
            ),
            node(11, Op::Softmax { beta: 1.0 }, &[4], 5),
            node(
                12,
                Op::BatchMatMul {
                    adj_x: false,
                    adj_y: true,
                },
                &[26, 5],
                6,
            ),
            node(13, Op::Reshape, &[6, 59], 58),
            node(14, Op::Transpose, &[58, 44], 43),
            node(15, Op::Reshape, &[43, 77], 78),
        ];
        g.outputs = vec![76, 78];
        let inputs = vec![
            40, 42, 55, 21, 22, 24, 25, 27, 28, 83, 61, 46, 75, 59, 44, 77,
        ];
        (g, inputs, vec![76, 78])
    }

    fn opts() -> PlanOptions {
        PlanOptions {
            features: CpuFeatures::default(),
            kernels: TfliteKernels::NeonRuy,
        }
    }

    #[test]
    fn yolo26_block_is_one_fused_step_and_one_copy() {
        let (g, inputs, outputs) = yolo26_block();
        let nodes: Vec<usize> = (0..g.nodes.len()).collect();
        let p = plan(&g, &nodes, &inputs, &outputs, opts()).unwrap();
        assert_eq!(p.fused_attention, 1);
        assert!(p.scratch.is_empty(), "{:?}", p.scratch);
        assert_eq!(p.steps.len(), 2, "{:#?}", p.steps);
        let Step::Attention { q, k, v, dst, .. } = &p.steps[1] else {
            panic!("{:#?}", p.steps[1]);
        };
        assert_eq!(
            (q.shape.clone(), q.strides.clone(), q.off),
            (vec![2, 400, 32], vec![128, 256, 1], 0)
        );
        assert_eq!((k.off, v.off), (32, 64));
        assert_eq!(
            (dst.buf, dst.strides.clone()),
            (Buf::Output(1), vec![64, 128, 1])
        );
    }

    /// The block as `TFLite` computes it, op by op, every view materialized.
    fn reference(g: &Graph, x: &[i8]) -> (Vec<i8>, Vec<i8>) {
        let q = |t: usize| g.tensor(t).unwrap().quant.unwrap();
        // t54[0, h, c, n] = x[n, h·128 + c]
        let at = |h: usize, c: usize, n: usize| x[n * 256 + h * 128 + c];
        let mut q3 = vec![0i8; 2 * 32 * 400];
        let mul = BinaryI8Plan::new(BinaryI8Params {
            op: BinaryOp::Mul,
            lhs: q(20),
            rhs: q(83),
            output: q(3),
            activation: Activation::None,
            kernels: TfliteKernels::NeonRuy,
        })
        .unwrap();
        let table = mul.table_for_rhs(127, 2 * 32 * 400).unwrap();
        let (mut k, mut v) = (vec![0i8; 2 * 32 * 400], vec![0i8; 2 * 64 * 400]);
        for h in 0..2 {
            for c in 0..32 {
                for n in 0..400 {
                    q3[(h * 32 + c) * 400 + n] =
                        table[usize::from(at(h, c, n).cast_unsigned())].cast_signed();
                    k[(h * 32 + c) * 400 + n] = at(h, 32 + c, n);
                }
            }
            for c in 0..64 {
                for n in 0..400 {
                    v[(h * 64 + c) * 400 + n] = at(h, 64 + c, n);
                }
            }
        }
        let f = CpuFeatures::default();
        let bmm1 = MatMulI8Plan::with_features(
            MatMulI8Params::tflite_batch_matmul(q(3), q(23), q(4), TfliteKernels::NeonRuy).unwrap(),
            f,
        )
        .unwrap();
        let mut s = vec![0i8; 2 * 400 * 400];
        bmm1.run(
            TensorRef::I8(View::strided(&q3, &[2, 400, 32], &[32 * 400, 1, 400]).unwrap()),
            TensorRef::I8(View::contiguous(&k, &[2, 32, 400]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut s, &[2, 400, 400]).unwrap()),
            None,
        )
        .unwrap();
        let sm = SoftmaxI8Plan::with_features(
            SoftmaxI8Params {
                input: q(4),
                output: q(5),
                beta: 1.0,
            },
            f,
        )
        .unwrap();
        let mut pr = vec![0i8; 2 * 400 * 400];
        sm.run(
            TensorRef::I8(View::contiguous(&s, &[2, 400, 400]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut pr, &[2, 400, 400]).unwrap()),
            None,
        )
        .unwrap();
        let bmm2 = MatMulI8Plan::with_features(
            MatMulI8Params::tflite_batch_matmul(q(26), q(5), q(6), TfliteKernels::NeonRuy).unwrap(),
            f,
        )
        .unwrap();
        let mut o = vec![0i8; 2 * 64 * 400];
        bmm2.run(
            TensorRef::I8(View::contiguous(&v, &[2, 64, 400]).unwrap()),
            TensorRef::I8(View::strided(&pr, &[2, 400, 400], &[160_000, 1, 400]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut o, &[2, 64, 400]).unwrap()),
            None,
        )
        .unwrap();
        // t78[n, h·64 + c] = t6[h, c, n]; t76[n, h·64 + c] = v[h, c, n].
        let (mut t76, mut t78) = (vec![0i8; 400 * 128], vec![0i8; 400 * 128]);
        for h in 0..2 {
            for c in 0..64 {
                for n in 0..400 {
                    t76[n * 128 + h * 64 + c] = v[(h * 64 + c) * 400 + n];
                    t78[n * 128 + h * 64 + c] = o[(h * 64 + c) * 400 + n];
                }
            }
        }
        (t76, t78)
    }

    #[test]
    fn yolo26_block_matches_the_operators_one_by_one() {
        let (g, inputs, outputs) = yolo26_block();
        let mut s = 7u32;
        let x: Vec<i8> = (0..400 * 256)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 81) as i8 - 40
            })
            .collect();
        let (want76, want78) = reference(&g, &x);
        let nodes: Vec<usize> = (0..g.nodes.len()).collect();
        for fuse in [true, false] {
            let mut nodes = nodes.clone();
            if !fuse {
                // Without the softmax in the partition the block cannot fuse.
                nodes.retain(|&n| n != 9);
            }
            let p = plan(&g, &nodes, &inputs, &outputs, opts());
            let Ok(p) = p else {
                assert!(!fuse);
                continue;
            };
            let consts: Vec<Vec<i8>> = inputs.iter().skip(1).map(|_| vec![127i8; 4]).collect();
            let mut ins: Vec<&[i8]> = vec![&x];
            ins.extend(consts.iter().map(Vec::as_slice));
            let (mut o76, mut o78) = (vec![0i8; 400 * 128], vec![0i8; 400 * 128]);
            crate::exec::run_buffers(p, &ins, &mut [&mut o76, &mut o78]).unwrap();
            assert_eq!(o76, want76, "value path");
            assert_eq!(o78, want78, "attention output");
        }
    }

    /// `yolo26_block` as Ultralytics' `LiteRT` export writes it: the scores reach the softmax
    /// through a batch reshape, and the probabilities are transposed and reshaped ahead of a
    /// second matmul without `adj_y`.
    fn ultralytics_attention_block() -> (Graph, Vec<usize>, Vec<usize>) {
        let (mut g, mut inputs, outputs) = yolo26_block();
        let scores = g.tensor(4).unwrap().quant.map(|q| (q.scale, q.zero_point));
        let probs = g.tensor(5).unwrap().quant.map(|q| (q.scale, q.zero_point));
        g.tensors.extend([
            (101, c32(&[2, 400, 400])),
            (100, t(&[2, 400, 400], scores)),
            (102, t(&[2, 400, 400], probs)),
            (103, c32(&[0, 2, 1])),
            (104, t(&[2, 400, 400], probs)),
            (105, c32(&[1, 2, 400, 400])),
            (106, t(&[1, 2, 400, 400], probs)),
        ]);
        let si = g.nodes.iter().position(|n| n.index == 11).unwrap();
        g.nodes.splice(
            si..=si + 1,
            [
                node(20, Op::Reshape, &[4, 101], 100),
                node(11, Op::Softmax { beta: 1.0 }, &[100], 102),
                node(21, Op::Transpose, &[102, 103], 104),
                node(22, Op::Reshape, &[104, 105], 106),
                node(
                    12,
                    Op::BatchMatMul {
                        adj_x: false,
                        adj_y: false,
                    },
                    &[26, 106],
                    6,
                ),
            ],
        );
        inputs.extend([101, 103, 105]);
        (g, inputs, outputs)
    }

    #[test]
    fn ultralytics_attention_block_fuses_and_matches_the_operators_one_by_one() {
        let (reference_graph, _, _) = yolo26_block();
        let (g, inputs, outputs) = ultralytics_attention_block();
        let blocks = attention::find(&g, &|_| true);
        assert_eq!(blocks.len(), 1, "{blocks:?}");
        assert!(blocks[0].p_right);
        assert_eq!((blocks[0].pre.len(), blocks[0].post.len()), (1, 2));

        let mut s = 11u32;
        let x: Vec<i8> = (0..400 * 256)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 81) as i8 - 40
            })
            .collect();
        let (want76, want78) = reference(&reference_graph, &x);
        let nodes: Vec<usize> = (0..g.nodes.len()).collect();
        let p = plan(&g, &nodes, &inputs, &outputs, opts()).unwrap();
        assert_eq!(p.fused_attention, 1);
        let consts: Vec<Vec<i8>> = inputs.iter().skip(1).map(|_| vec![127i8; 4]).collect();
        let mut ins: Vec<&[i8]> = vec![&x];
        ins.extend(consts.iter().map(Vec::as_slice));
        let (mut o76, mut o78) = (vec![0i8; 400 * 128], vec![0i8; 400 * 128]);
        crate::exec::run_buffers(p, &ins, &mut [&mut o76, &mut o78]).unwrap();
        assert_eq!(o76, want76, "value path");
        assert_eq!(o78, want78, "attention output");
    }

    /// A transpose that is not a swap of the last two axes is not a view of `P`.
    #[test]
    fn attention_rejects_other_transposes_of_the_probabilities() {
        let (mut g, _, _) = ultralytics_attention_block();
        g.tensors.insert(103, c32(&[1, 0, 2]));
        assert_eq!(attention::find(&g, &|_| true), Vec::new());
    }

    /// A two-level YOLO DFL head: NHWC box logits of 16 and 4 anchors, concatenated, split
    /// into four sides of 16 bins, softmax over the bins, then the 1×1 one-channel projection.
    fn dfl_head() -> (Graph, Vec<usize>, Vec<usize>) {
        let x = Some((0.0947, -20));
        let probs = Some((1.0 / 256.0, -128));
        let mut g = Graph::default();
        let filter = Tensor {
            constant: Some(Const::I8(
                (0..16i32)
                    .map(|i| i8::try_from((i * 127 + 7) / 15).unwrap())
                    .collect(),
            )),
            ..t(&[1, 1, 1, 16], Some((15.0 / 127.0, 0)))
        };
        let tensors = [
            (0, t(&[1, 16, 64], x)),
            (1, t(&[1, 4, 64], x)),
            (2, t(&[1, 20, 64], x)),
            (3, c32(&[1, 20, 4, 16])),
            (4, t(&[1, 20, 4, 16], x)),
            (5, c32(&[0, 2, 1, 3])),
            (6, t(&[1, 4, 20, 16], x)),
            (7, t(&[1, 4, 20, 16], probs)),
            (8, filter),
            (9, c32(&[0])),
            (10, t(&[1, 4, 20, 1], Some((0.0579, -128)))),
        ];
        g.tensors = tensors.into_iter().collect();
        g.nodes = vec![
            node(
                0,
                Op::Concat {
                    axis: 1,
                    act: Activation::None,
                },
                &[0, 1],
                2,
            ),
            node(1, Op::Reshape, &[2, 3], 4),
            node(2, Op::Transpose, &[4, 5], 6),
            node(3, Op::Softmax { beta: 1.0 }, &[6], 7),
            node(
                4,
                Op::Conv2d {
                    unit: true,
                    act: Activation::None,
                },
                &[7, 8, 9],
                10,
            ),
        ];
        g.outputs = vec![10];
        (g, vec![0, 1, 3, 5, 8, 9], vec![10])
    }

    #[test]
    fn dfl_head_fuses_and_stays_close_to_the_operators_one_by_one() {
        let (g, inputs, outputs) = dfl_head();
        let mut s = 11u32;
        let mut rnd = |n: usize| -> Vec<i8> {
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    (s % 121) as i8 - 60
                })
                .collect()
        };
        let (x0, x1) = (rnd(16 * 64), rnd(4 * 64));
        // Shapes, the filter and the bias, as the runtime hands them over.
        let filter: Vec<i8> = (0..16i32)
            .map(|i| i8::try_from((i * 127 + 7) / 15).unwrap())
            .collect();
        let consts: Vec<Vec<i8>> = vec![vec![0; 16], vec![0; 16], filter, vec![0; 4]];
        let mut ins: Vec<&[i8]> = vec![&x0, &x1];
        ins.extend(consts.iter().map(Vec::as_slice));
        let nodes: Vec<usize> = (0..g.nodes.len()).collect();

        let fused = plan(&g, &nodes, &inputs, &outputs, opts()).unwrap();
        assert_eq!(fused.fused_softmax_dot, 1);
        let kinds: Vec<&str> = fused
            .steps
            .iter()
            .map(|s| match s {
                Step::SoftmaxDot { .. } => "softmax_dot",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, ["softmax_dot", "softmax_dot"], "one step per level");
        assert!(fused.scratch.is_empty(), "no probabilities in scratch");
        let mut got = vec![0i8; 4 * 20];
        crate::exec::run_buffers(fused, &ins, &mut [&mut got]).unwrap();

        // With the probabilities a partition output, the block runs operator by operator.
        let unfused_outputs = vec![10, 7];
        let p = plan(&g, &nodes, &inputs, &unfused_outputs, opts()).unwrap();
        assert_eq!(p.fused_softmax_dot, 0);
        let (mut want, mut probs) = (vec![0i8; 4 * 20], vec![0i8; 4 * 20 * 16]);
        crate::exec::run_buffers(p, &ins, &mut [&mut want, &mut probs]).unwrap();
        // The fused kernel skips the 8-bit probabilities, so it differs by their rounding.
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (i32::from(g) - i32::from(w)).abs() <= 3,
                "output {i}: {g} vs {w}"
            );
        }
        assert!(got != want, "the fused kernel ran");
    }

    /// A DFL head in NCHW, as Ultralytics' `LiteRT` export writes it: 4 sides × 16 bins on the
    /// channel axis of 20 anchors, so the transpose leaves the bins 20 elements apart.
    fn nchw_dfl_head() -> (Graph, Vec<usize>, Vec<usize>) {
        let x = Some((0.0947, -20));
        let probs = Some((1.0 / 256.0, -128));
        let filter = Tensor {
            constant: Some(Const::I8(
                (0..16i32)
                    .map(|i| i8::try_from((i * 127 + 7) / 15).unwrap())
                    .collect(),
            )),
            ..t(&[1, 1, 1, 16], Some((15.0 / 127.0, 0)))
        };
        let tensors = [
            (0, t(&[1, 64, 20], x)),
            (1, c32(&[4, 16, 20])),
            (2, t(&[4, 16, 20], x)),
            (3, c32(&[0, 2, 1])),
            (4, t(&[4, 20, 16], x)),
            (5, c32(&[1, 4, 20, 16])),
            (6, t(&[1, 4, 20, 16], x)),
            (7, t(&[1, 4, 20, 16], probs)),
            (8, filter),
            (9, c32(&[0])),
            (10, t(&[1, 4, 20, 1], Some((0.0579, -128)))),
        ]
        .into_iter()
        .collect();
        let nodes = vec![
            node(0, Op::Reshape, &[0, 1], 2),
            node(1, Op::Transpose, &[2, 3], 4),
            node(2, Op::Reshape, &[4, 5], 6),
            node(3, Op::Softmax { beta: 1.0 }, &[6], 7),
            node(
                4,
                Op::Conv2d {
                    unit: true,
                    act: Activation::None,
                },
                &[7, 8, 9],
                10,
            ),
        ];
        let g = Graph {
            tensors,
            nodes,
            outputs: vec![10],
            ..Graph::default()
        };
        (g, vec![0, 1, 3, 5, 8, 9], vec![10])
    }

    #[test]
    fn nchw_dfl_head_gathers_its_bins_and_fuses() {
        let (g, inputs, outputs) = nchw_dfl_head();
        let mut s = 5u32;
        let x: Vec<i8> = (0..64 * 20)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 121) as i8 - 60
            })
            .collect();
        let filter: Vec<i8> = (0..16i32)
            .map(|i| i8::try_from((i * 127 + 7) / 15).unwrap())
            .collect();
        let consts: Vec<Vec<i8>> = vec![vec![0; 12], vec![0; 12], vec![0; 16], filter, vec![0; 4]];
        let mut ins: Vec<&[i8]> = vec![&x];
        ins.extend(consts.iter().map(Vec::as_slice));
        let nodes: Vec<usize> = (0..g.nodes.len()).collect();

        let fused = plan(&g, &nodes, &inputs, &outputs, opts()).unwrap();
        assert_eq!(fused.fused_softmax_dot, 1);
        assert_eq!(fused.scratch.len(), 1, "one dense copy of the bins");
        let mut got = vec![0i8; 4 * 20];
        crate::exec::run_buffers(fused, &ins, &mut [&mut got]).unwrap();

        let unfused_outputs = vec![10, 7];
        let p = plan(&g, &nodes, &inputs, &unfused_outputs, opts()).unwrap();
        assert_eq!(p.fused_softmax_dot, 0);
        let (mut want, mut probs) = (vec![0i8; 4 * 20], vec![0i8; 4 * 20 * 16]);
        crate::exec::run_buffers(p, &ins, &mut [&mut want, &mut probs]).unwrap();
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (i32::from(g) - i32::from(w)).abs() <= 3,
                "output {i}: {g} vs {w}"
            );
        }
    }

    /// In `BATCH_MATMUL → SOFTMAX → BATCH_MATMUL → SOFTMAX → BATCH_MATMUL` the middle matmul
    /// ends one block and starts the next: only the first fuses, and the rest runs op by op.
    #[test]
    fn chained_attention_blocks_fuse_without_sharing_a_node() {
        let (x, scores, probs, out) = (
            Some((0.1251, -15)),
            Some((0.1279, -9)),
            Some((1.0 / 256.0, -128)),
            Some((0.063_33, 2)),
        );
        let mm = Op::BatchMatMul {
            adj_x: false,
            adj_y: false,
        };
        let g = Graph {
            tensors: [
                (0, t(&[1, 4, 8], x)),
                (1, t(&[1, 8, 4], x)),
                (2, t(&[1, 4, 4], scores)),
                (3, t(&[1, 4, 4], probs)),
                (4, t(&[1, 4, 4], x)),
                (5, t(&[1, 4, 4], scores)),
                (6, t(&[1, 4, 4], probs)),
                (7, t(&[1, 4, 4], x)),
                (8, t(&[1, 4, 4], out)),
            ]
            .into_iter()
            .collect(),
            nodes: vec![
                node(0, mm, &[0, 1], 2),
                node(1, Op::Softmax { beta: 1.0 }, &[2], 3),
                node(2, mm, &[3, 4], 5),
                node(3, Op::Softmax { beta: 1.0 }, &[5], 6),
                node(4, mm, &[6, 7], 8),
            ],
            outputs: vec![8],
            ..Graph::default()
        };
        let blocks = attention::find(&g, &|_| true);
        assert_eq!(blocks.len(), 1, "{blocks:?}");
        assert_eq!((blocks[0].scores, blocks[0].output), (0, 2));

        let nodes: Vec<usize> = (0..g.nodes.len()).collect();
        let p = plan(&g, &nodes, &[0, 1, 4, 7], &[8], opts()).unwrap();
        assert_eq!(p.fused_attention, 1);
        let ins: Vec<Vec<i8>> = [32, 32, 16, 16]
            .iter()
            .map(|&n| (0..n).map(|i| (i * 7 % 50) as i8 - 25).collect())
            .collect();
        let ins: Vec<&[i8]> = ins.iter().map(Vec::as_slice).collect();
        let mut got = vec![0i8; 16];
        crate::exec::run_buffers(p, &ins, &mut [&mut got]).unwrap();
    }

    /// `TFLite`'s NEON `ADD` runs whole 16-element blocks over the output. An operand concatenated
    /// from 8-element parts must not split the `ADD` into 8-element calls, which take the scalar
    /// tail arithmetic instead.
    #[test]
    fn binary_ops_keep_the_neon_blocks_of_a_concatenated_operand() {
        let x = Some((0.1251, -15));
        let out = Some((0.2, 3));
        let g = Graph {
            tensors: [
                (0, t(&[1, 8], x)),
                (1, t(&[1, 8], x)),
                (2, t(&[1, 16], x)),
                (3, t(&[1, 16], x)),
                (4, t(&[1, 16], out)),
            ]
            .into_iter()
            .collect(),
            nodes: vec![
                node(
                    0,
                    Op::Concat {
                        axis: 1,
                        act: Activation::None,
                    },
                    &[0, 1],
                    2,
                ),
                node(1, Op::Add(Activation::None), &[2, 3], 4),
            ],
            outputs: vec![4],
            ..Graph::default()
        };
        let nodes: Vec<usize> = (0..g.nodes.len()).collect();
        let p = plan(&g, &nodes, &[0, 1, 3], &[4], opts()).unwrap();
        for s in &p.steps {
            if let Step::Binary { dst, .. } = s {
                assert!(dst.numel().is_multiple_of(16), "{:#?}", p.steps);
            }
        }

        let lhs: Vec<i8> = (0..16)
            .map(|i| i8::try_from(i * 16 - 128).unwrap())
            .collect();
        let rhs: Vec<i8> = (0..16)
            .map(|i| i8::try_from(127 - i * 13).unwrap())
            .collect();
        let mut got = vec![0i8; 16];
        crate::exec::run_buffers(p, &[&lhs[..8], &lhs[8..], &rhs], &mut [&mut got]).unwrap();
        let add = BinaryI8Plan::with_features(
            BinaryI8Params {
                op: BinaryOp::Add,
                lhs: Quant::new(0.1251, -15),
                rhs: Quant::new(0.1251, -15),
                output: Quant::new(0.2, 3),
                activation: Activation::None,
                kernels: TfliteKernels::NeonRuy,
            },
            CpuFeatures::default(),
        )
        .unwrap();
        let mut want = vec![0i8; 16];
        add.run(
            TensorRef::I8(View::contiguous(&lhs, &[1, 16]).unwrap()),
            TensorRef::I8(View::contiguous(&rhs, &[1, 16]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut want, &[1, 16]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(got, want);
    }
}
