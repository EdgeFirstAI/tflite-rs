// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Runs a planned partition.

use std::sync::Arc;

use edgefirst_kernels::{
    AttentionScratch, CpuFeatures, DType, LutI8Plan, Pool, TensorMut, TensorRef, View, ViewMut,
};
use edgefirst_tflite::delegate::custom::{BoxError, KernelIo, PartitionKernel};

use crate::graph::Elem;
use crate::plan::{Program, Step};
use crate::views::{Buf, Loc};
use crate::workers::Threads;

/// A buffer's address, length in elements and element size in bytes.
#[derive(Clone, Copy, Debug)]
struct Span {
    addr: *mut u8,
    len: usize,
    size: usize,
}

impl Default for Span {
    fn default() -> Self {
        Self {
            addr: core::ptr::null_mut(),
            len: 0,
            size: 0,
        }
    }
}

// SAFETY: a `Span` is an address the kernel dereferences only during an invoke, on the threads
// that invoke runs, while the buffer it names is valid; the address itself may move between
// threads.
unsafe impl Send for Span {}

impl Span {
    /// A buffer kernels only read.
    fn of<T>(s: &[T]) -> Self {
        Self {
            addr: s.as_ptr().cast::<u8>().cast_mut(),
            len: s.len(),
            size: size_of::<T>(),
        }
    }

    /// A buffer kernels write: its address keeps the write permission of `s`.
    fn of_mut<T>(s: &mut [T]) -> Self {
        Self {
            addr: s.as_mut_ptr().cast::<u8>(),
            len: s.len(),
            size: size_of::<T>(),
        }
    }

    /// The bytes `l` addresses in this buffer, given `l` lies in it.
    fn bytes(&self, l: &Loc) -> std::ops::Range<usize> {
        let start = self.addr.addr() + l.off * self.size;
        let end = self.addr.addr() + l.extent().unwrap_or(0).max(l.off) * self.size;
        start..end
    }
}

/// The memory of one invoke: partition inputs and outputs, and the kernel's scratch.
struct Mem<'a> {
    ins: &'a [Span],
    outs: &'a [Span],
    scratch: &'a [Span],
}

fn err(msg: &str) -> BoxError {
    msg.to_owned().into()
}

impl Mem<'_> {
    fn span(&self, l: &Loc) -> Result<Span, BoxError> {
        match l.buf {
            Buf::Input(k) => self.ins.get(k),
            Buf::Output(k) => self.outs.get(k),
            Buf::Scratch(k) => self.scratch.get(k),
        }
        .copied()
        .ok_or_else(|| err("unknown buffer"))
    }

    /// Whether the bytes `dst` addresses miss those of every `srcs`, which the plan guarantees
    /// for every step: no step writes memory it reads.
    fn disjoint(&self, dst: &Loc, srcs: &[&Loc]) -> bool {
        let Ok(d) = self.span(dst).map(|s| s.bytes(dst)) else {
            return true;
        };
        srcs.iter().all(|l| {
            self.span(l)
                .map(|s| s.bytes(l))
                .is_ok_and(|r| r.is_empty() || d.is_empty() || r.end <= d.start || d.end <= r.start)
        })
    }

    /// The first element of `l`, after checking the view lies in its buffer, whose elements
    /// are `T`s.
    fn base<T>(&self, l: &Loc) -> Result<*mut T, BoxError> {
        let s = self.span(l)?;
        if s.size != size_of::<T>() {
            return Err(err("view element type differs from its buffer's"));
        }
        if l.extent().is_none_or(|e| e > s.len) {
            return Err(err("view outside its buffer"));
        }
        // SAFETY: `off` is inside the buffer (checked above), or the view is empty.
        Ok(unsafe { s.addr.cast::<T>().add(l.off.min(s.len)) })
    }

    fn view<'v, T>(&self, l: &Loc) -> Result<View<'v, T>, BoxError> {
        let p = self.base::<T>(l)?;
        // SAFETY: the view lies in a buffer valid for this invoke (checked by `base`), and the
        // plan never lets a step write memory it reads.
        Ok(unsafe { View::from_raw_parts(p.cast_const(), &l.shape, Some(&l.strides))? })
    }

    fn view_mut<'v, T>(&self, l: &Loc) -> Result<ViewMut<'v, T>, BoxError> {
        let p = self.base::<T>(l)?;
        // SAFETY: as `view`; no other operand of the step addresses this memory.
        Ok(unsafe { ViewMut::from_raw_parts(p, &l.shape, Some(&l.strides))? })
    }

    /// Element-by-element copy between two locations of the same shape.
    fn copy<T: Copy>(&self, src: &Loc, dst: &Loc) -> Result<(), BoxError> {
        if src.shape != dst.shape {
            return Err(err("copy shapes differ"));
        }
        let (sp, dp) = (self.base::<T>(src)?, self.base::<T>(dst)?);
        let rank = src.shape.len();
        let n = src.numel();
        let mut idx = [0usize; edgefirst_kernels::MAX_RANK];
        for _ in 0..n {
            let (mut so, mut d_o) = (0usize, 0usize);
            for ((&i, s), d) in idx[..rank].iter().zip(&src.strides).zip(&dst.strides) {
                so += i * s.unsigned_abs();
                d_o += i * d.unsigned_abs();
            }
            // SAFETY: both offsets address elements of their views, which lie in their buffers.
            unsafe { dp.add(d_o).write(sp.add(so).read()) };
            for a in (0..rank).rev() {
                idx[a] += 1;
                if idx[a] < src.shape[a] {
                    break;
                }
                idx[a] = 0;
            }
        }
        Ok(())
    }
}

enum Scratch {
    I8(Vec<i8>),
    I16(Vec<i16>),
}

/// The kernel of one partition.
pub struct EdgeFirstKernel {
    steps: Vec<Step>,
    inputs: Vec<(Elem, Vec<usize>)>,
    outputs: Vec<(Elem, Vec<usize>)>,
    /// Owns the scratch memory `scratch_spans` points into; never resized.
    _scratch: Vec<Scratch>,
    scratch_spans: Vec<Span>,
    attention: Vec<AttentionScratch>,
    copy8: LutI8Plan,
    threads: Arc<Threads>,
    ins: Vec<Span>,
    outs: Vec<Span>,
}

impl std::fmt::Debug for EdgeFirstKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeFirstKernel")
            .field("steps", &self.steps.len())
            .field("scratch", &self.scratch_spans.len())
            .finish_non_exhaustive()
    }
}

impl EdgeFirstKernel {
    pub(crate) fn new(
        p: Program,
        features: CpuFeatures,
        threads: Arc<Threads>,
    ) -> Result<Self, BoxError> {
        // Every index is below 256.
        #[allow(clippy::cast_possible_truncation)]
        let identity: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut scratch: Vec<Scratch> = p
            .scratch
            .into_iter()
            .map(|(e, n)| match e {
                Elem::I16 => Scratch::I16(vec![0; n]),
                _ => Scratch::I8(vec![0; n]),
            })
            .collect();
        let scratch_spans = scratch
            .iter_mut()
            .map(|s| match s {
                Scratch::I8(v) => Span::of_mut(v),
                Scratch::I16(v) => Span::of_mut(v),
            })
            .collect();
        Ok(Self {
            steps: p.steps,
            ins: Vec::with_capacity(p.inputs.len()),
            outs: Vec::with_capacity(p.outputs.len()),
            inputs: p.inputs,
            outputs: p.outputs,
            _scratch: scratch,
            scratch_spans,
            attention: p.attention,
            copy8: LutI8Plan::from_table_with_features(DType::I8, identity, features)?,
            threads,
        })
    }

    fn capture(&mut self, io: &mut KernelIo<'_>) -> Result<(), BoxError> {
        self.ins.clear();
        self.outs.clear();
        for (k, (elem, _)) in self.inputs.iter().enumerate() {
            let t = io.input(k)?;
            let span = match elem {
                Elem::I8 => Span::of(t.as_slice::<i8>()?),
                Elem::I16 => Span::of(t.as_slice::<i16>()?),
                Elem::I32 => Span::of(t.as_slice::<i32>()?),
                Elem::Other => Span::default(),
            };
            self.ins.push(span);
        }
        for (k, (elem, _)) in self.outputs.iter().enumerate() {
            let mut t = io.output(k)?;
            let span = match elem {
                Elem::I8 => Span::of_mut(t.as_mut_slice::<i8>()?),
                Elem::I16 => Span::of_mut(t.as_mut_slice::<i16>()?),
                _ => return Err(err("unsupported output type")),
            };
            self.outs.push(span);
        }
        Ok(())
    }
}

/// The location a step writes and the locations it reads.
fn operands(step: &Step) -> (&Loc, Vec<&Loc>) {
    match step {
        Step::Softmax8 { src, dst, .. }
        | Step::Softmax16 { src, dst, .. }
        | Step::SoftmaxDot { src, dst, .. }
        | Step::Lut { src, dst, .. }
        | Step::Copy { src, dst, .. } => (dst, vec![src]),
        Step::Binary { lhs, rhs, dst, .. } | Step::MatMul { lhs, rhs, dst, .. } => {
            (dst, vec![lhs, rhs])
        }
        Step::Attention { q, k, v, dst, .. } => (dst, vec![q, k, v]),
    }
}

fn exec(
    step: &Step,
    mem: &Mem<'_>,
    attention: &mut [AttentionScratch],
    copy8: &LutI8Plan,
    pool: Option<&dyn Pool>,
) -> Result<(), BoxError> {
    debug_assert!(
        {
            let (dst, srcs) = operands(step);
            mem.disjoint(dst, &srcs)
        },
        "a step writes memory it reads"
    );
    match step {
        Step::Softmax8 { plan, src, dst } => plan.run(
            TensorRef::I8(mem.view(src)?),
            TensorMut::I8(mem.view_mut(dst)?),
            pool,
        )?,
        Step::Softmax16 { plan, src, dst } => plan.run(
            TensorRef::I16(mem.view(src)?),
            TensorMut::I16(mem.view_mut(dst)?),
            pool,
        )?,
        Step::SoftmaxDot { plan, src, dst } => plan.run(
            TensorRef::I8(mem.view(src)?),
            TensorMut::I8(mem.view_mut(dst)?),
            pool,
        )?,
        Step::Lut { plan, src, dst } => plan.run(
            TensorRef::I8(mem.view(src)?),
            TensorMut::I8(mem.view_mut(dst)?),
            pool,
        )?,
        Step::Binary {
            plan,
            lhs,
            rhs,
            dst,
        } => plan.run(
            TensorRef::I8(mem.view(lhs)?),
            TensorRef::I8(mem.view(rhs)?),
            TensorMut::I8(mem.view_mut(dst)?),
            pool,
        )?,
        Step::MatMul {
            plan,
            lhs,
            rhs,
            dst,
        } => plan.run(
            TensorRef::I8(mem.view(lhs)?),
            TensorRef::I8(mem.view(rhs)?),
            TensorMut::I8(mem.view_mut(dst)?),
            pool,
        )?,
        Step::Attention {
            plan,
            scratch,
            q,
            k,
            v,
            dst,
        } => {
            let s = attention
                .get_mut(*scratch)
                .ok_or_else(|| err("attention scratch"))?;
            plan.run(
                TensorRef::I8(mem.view(q)?),
                TensorRef::I8(mem.view(k)?),
                TensorRef::I8(mem.view(v)?),
                TensorMut::I8(mem.view_mut(dst)?),
                s,
                pool,
            )?;
        }
        Step::Copy {
            elem: Elem::I8,
            src,
            dst,
        } => copy8.run(
            TensorRef::I8(mem.view(src)?),
            TensorMut::I8(mem.view_mut(dst)?),
            pool,
        )?,
        Step::Copy {
            elem: Elem::I16,
            src,
            dst,
        } => mem.copy::<i16>(src, dst)?,
        Step::Copy { .. } => return Err(err("unsupported copy")),
    }
    Ok(())
}

impl PartitionKernel for EdgeFirstKernel {
    fn prepare(&mut self, io: &mut KernelIo<'_>) -> Result<(), BoxError> {
        if io.num_inputs() != self.inputs.len() || io.num_outputs() != self.outputs.len() {
            return Err(err("partition boundary changed since planning"));
        }
        for (k, (_, shape)) in self.inputs.iter().enumerate() {
            if io.input(k)?.info().shape()? != *shape {
                return Err(err("partition input shape changed since planning"));
            }
        }
        for (k, (_, shape)) in self.outputs.iter().enumerate() {
            if io.output(k)?.info().shape()? != *shape {
                return Err(err("partition output shape changed since planning"));
            }
        }
        Ok(())
    }

    fn invoke(&mut self, io: &mut KernelIo<'_>) -> Result<(), BoxError> {
        self.capture(io)?;
        let threads = io
            .recommended_num_threads()
            .map_or(1, std::num::NonZeroUsize::get);
        let workers = self.threads.get(threads);
        let pool = workers.as_ref().map(|w| w as &dyn Pool);
        let mem = Mem {
            ins: &self.ins,
            outs: &self.outs,
            scratch: &self.scratch_spans,
        };
        let ran = self
            .steps
            .iter()
            .try_for_each(|step| exec(step, &mem, &mut self.attention, &self.copy8, pool));
        if let Some(w) = workers.as_ref() {
            w.rest();
        }
        ran
    }
}

/// Runs `p` on plain buffers: the host-side test harness.
#[cfg(test)]
pub(crate) fn run_buffers(
    p: Program,
    ins: &[&[i8]],
    outs: &mut [&mut [i8]],
) -> Result<(), BoxError> {
    let mut k = EdgeFirstKernel::new(p, CpuFeatures::default(), Arc::default())?;
    k.ins = ins.iter().map(|s| Span::of(s)).collect();
    k.outs = outs.iter_mut().map(|s| Span::of_mut(s)).collect();
    let mem = Mem {
        ins: &k.ins,
        outs: &k.outs,
        scratch: &k.scratch_spans,
    };
    let workers = k.threads.get(3);
    let pool = workers.as_ref().map(|w| w as &dyn Pool);
    for step in &k.steps {
        exec(step, &mem, &mut k.attention, &k.copy8, pool)?;
    }
    Ok(())
}
