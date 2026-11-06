// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! The `EdgeFirst` `TFLite` delegate.
//!
//! Runs the parts of a model that NPU delegates leave on the CPU with the NEON kernels of
//! [`edgefirst_kernels`], in place of `TFLite`'s builtin CPU kernels. It claims connected
//! regions of:
//!
//! * `SOFTMAX` (int8, int16), `LOGISTIC`, `ADD`, `SUB`, `MUL` (int8), `BATCH_MATMUL` (int8)
//!   and 1×1 `CONV_2D` (int8), each reproducing `TFLite`'s int8 result: bit for bit on the
//!   scalar kernel tier, and on NEON too except int8 `SOFTMAX` (within ±1 LSB);
//! * `TRANSPOSE`, `RESHAPE`, `SLICE` and `CONCATENATION`, which it does not run at all: they
//!   become strided views and part lists that the compute kernels read and write directly;
//! * attention blocks (`[MUL] → BATCH_MATMUL → SOFTMAX → BATCH_MATMUL`, as in `YOLO11` and
//!   `YOLO26`), fused into one kernel that reads Q, K and V where they are, writes the result
//!   where it is needed, and keeps every one of `TFLite`'s int8 roundings;
//! * distribution-focal-loss heads (`SOFTMAX → [RESHAPE…] → CONV_2D` with one 1×1 output
//!   channel, as in `YOLOv8` and `YOLO11`), fused into one kernel that skips the 8-bit
//!   probabilities between the two: its results sit closer to the float model than `TFLite`'s,
//!   within a few LSB of them.
//!
//! It claims a node only when it reproduces it (the fused heads aside), and leaves everything
//! else to `TFLite` or to the delegates added after it. Kernels read and write tensors in the
//! memory `TFLite` gives them, and the delegate never sets buffer handles.
//!
//! Kernels use as many threads as the interpreter was given
//! ([`InterpreterBuilder::num_threads`](edgefirst_tflite::InterpreterBuilder::num_threads),
//! `TfLiteInterpreterOptionsSetNumThreads` or `Interpreter::SetNumThreads`), the calling
//! thread included, up to the CPUs the process may use, and only that one when none was
//! given, as `TFLite`'s own CPU kernels do.
//! The count is read on every invoke. The delegate keeps its worker threads across invokes;
//! they poll between the kernels of an invoke and sleep as soon as it ends.
//!
//! Add it after the NPU delegate to take over the CPU fallback, or before it with
//! [`Scope::Attention`] to keep only the attention blocks off the NPU:
//!
//! ```no_run
//! use edgefirst_tflite::{Delegate, Interpreter, Library, Model};
//! use edgefirst_tflite_delegate::EdgeFirstDelegate;
//!
//! let lib = Library::new()?;
//! let model = Model::from_file(&lib, "model.imx95.tflite")?;
//! let edgefirst = EdgeFirstDelegate::new()?;
//! let stats = edgefirst.stats();
//! let mut interpreter = Interpreter::builder(&lib)?
//!     .delegate(Delegate::load("/usr/lib/libneutron_delegate.so")?)
//!     .delegate(edgefirst.into_delegate(&lib)?)
//!     .build(&model)?;
//! println!("EdgeFirst delegate claimed {} nodes", stats.claimed_nodes());
//! interpreter.invoke()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Each kernel run emits a `kernels.<op>` `tracing` span.

// Shape and index arithmetic reads best with the usual one-letter names (`m`, `k`, `n`, `h`), and
// the planners are single matches over the supported operators.
#![allow(clippy::many_single_char_names, clippy::too_many_lines)]

mod attention;
mod exec;
mod graph;
mod plan;
mod select;
mod softmax_dot;
mod views;
mod workers;

use std::ffi::CStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub use edgefirst_kernels::{CpuFeatures, KernelError, TfliteKernels, Tier};
use edgefirst_tflite::delegate::custom::{BoxError, CustomDelegate, GraphView, NodeId, Partition};
use edgefirst_tflite::{Delegate, Library};

pub use exec::EdgeFirstKernel;
use graph::Graph;
use plan::PlanOptions;

/// Which nodes the delegate may claim.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Scope {
    /// Every supported node in regions that do some computation. Use after an NPU delegate,
    /// or alone.
    #[default]
    Everything,
    /// Only attention blocks. Use before an NPU delegate that compiles the rest.
    Attention,
}

/// Delegate options: [`Options::default`], changed with [`Options::scope`] and
/// [`Options::kernels`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Options {
    pub scope: Scope,
    /// The `TFLite` kernels whose int8 rounding to reproduce: those of the runtime the delegate
    /// is loaded into.
    pub kernels: TfliteKernels,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            scope: Scope::Everything,
            kernels: if cfg!(target_arch = "aarch64") {
                TfliteKernels::NeonRuy
            } else if cfg!(target_feature = "sse4.1") {
                TfliteKernels::NeonGemmlowp
            } else {
                TfliteKernels::Ruy
            },
        }
    }
}

impl Options {
    /// These options with `scope`.
    #[must_use]
    pub fn scope(mut self, scope: Scope) -> Self {
        self.scope = scope;
        self
    }

    /// These options with `kernels`.
    #[must_use]
    pub fn kernels(mut self, kernels: TfliteKernels) -> Self {
        self.kernels = kernels;
        self
    }
}

/// The `EdgeFirst` delegate, before it is wrapped for an interpreter.
///
/// Use one instance per interpreter.
#[derive(Debug)]
pub struct EdgeFirstDelegate {
    features: CpuFeatures,
    options: Options,
    stats: Stats,
    threads: Arc<workers::Threads>,
}

impl EdgeFirstDelegate {
    /// A delegate with default [`Options`] whose kernels target the CPU this process runs on.
    ///
    /// Honours `EDGEFIRST_KERNELS_FORCE_TIER`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::ForcedTierUnavailable`] when
    /// `EDGEFIRST_KERNELS_FORCE_TIER` names a tier this CPU lacks, or an unknown one.
    pub fn new() -> Result<Self, KernelError> {
        Ok(Self::with_cpu_features(CpuFeatures::from_env_or_probe()?))
    }

    /// A delegate with default [`Options`] whose kernels target `features`; reads no
    /// environment.
    #[must_use]
    pub fn with_cpu_features(features: CpuFeatures) -> Self {
        Self {
            features,
            options: Options::default(),
            stats: Stats::default(),
            threads: Arc::default(),
        }
    }

    /// The delegate with `options`.
    #[must_use]
    pub fn with_options(mut self, options: Options) -> Self {
        self.options = options;
        self
    }

    /// The CPU tier the kernels are built for.
    #[must_use]
    pub fn tier(&self) -> Tier {
        self.features.tier()
    }

    /// A handle to the delegate's counters, which stays valid after
    /// [`into_delegate`](Self::into_delegate).
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.stats.clone()
    }

    /// Wrap the delegate for [`InterpreterBuilder::delegate`](edgefirst_tflite::InterpreterBuilder::delegate).
    ///
    /// # Errors
    ///
    /// Fails if the `TFLite` runtime does not export the opaque delegate API
    /// (`TFLite` 2.17 and later do).
    pub fn into_delegate(self, lib: &Library) -> Result<Delegate, edgefirst_tflite::Error> {
        Delegate::from_custom(lib, self)
    }

    fn plan_options(&self) -> PlanOptions {
        PlanOptions {
            features: self.features,
            kernels: self.options.kernels,
        }
    }
}

/// Counters shared between an [`EdgeFirstDelegate`] and its handles.
#[derive(Debug, Clone, Default)]
pub struct Stats(Arc<Counters>);

#[derive(Debug, Default)]
struct Counters {
    claimed_nodes: AtomicUsize,
    partitions: AtomicUsize,
    fused_attention: AtomicUsize,
    fused_softmax_dot: AtomicUsize,
}

impl Stats {
    /// Nodes claimed by the latest node selection.
    ///
    /// `TFLite` selects nodes when the interpreter is built, and again when it re-applies
    /// delegates (after an input is resized).
    #[must_use]
    pub fn claimed_nodes(&self) -> usize {
        self.0.claimed_nodes.load(Ordering::Relaxed)
    }

    /// Partition kernels created so far.
    #[must_use]
    pub fn partitions(&self) -> usize {
        self.0.partitions.load(Ordering::Relaxed)
    }

    /// Attention blocks running as one fused kernel, over the partitions created so far.
    #[must_use]
    pub fn fused_attention(&self) -> usize {
        self.0.fused_attention.load(Ordering::Relaxed)
    }

    /// Softmax-then-projection heads (YOLO DFL) running as one fused kernel, over the
    /// partitions created so far.
    #[must_use]
    pub fn fused_softmax_dot(&self) -> usize {
        self.0.fused_softmax_dot.load(Ordering::Relaxed)
    }
}

impl CustomDelegate for EdgeFirstDelegate {
    type Kernel = EdgeFirstKernel;

    fn name(&self) -> &CStr {
        c"EdgeFirstDelegate"
    }

    fn select_nodes(&self, view: &GraphView<'_>) -> Result<Vec<NodeId>, BoxError> {
        let g = Graph::from_view(view)?;
        let picked = select::select(&g, self.features, self.options.kernels, self.options.scope);
        self.stats
            .0
            .claimed_nodes
            .store(picked.len(), Ordering::Relaxed);
        Ok(picked
            .into_iter()
            .filter_map(|i| g.ids.get(i).copied())
            .collect())
    }

    fn create_kernel(&self, partition: &Partition<'_>) -> Result<EdgeFirstKernel, BoxError> {
        let g = Graph::from_partition(
            partition.graph(),
            partition.nodes(),
            partition.output_tensors(),
        )?;
        let nodes: Vec<usize> = (0..g.nodes.len()).collect();
        let inputs: Vec<usize> = partition
            .input_tensors()
            .iter()
            .map(|t| t.index())
            .collect();
        let outputs: Vec<usize> = partition
            .output_tensors()
            .iter()
            .map(|t| t.index())
            .collect();
        let program = plan::plan(&g, &nodes, &inputs, &outputs, self.plan_options())?;
        let (attention, dfl) = (program.fused_attention, program.fused_softmax_dot);
        let kernel = EdgeFirstKernel::new(program, self.features, Arc::clone(&self.threads))?;
        let s = &self.stats.0;
        s.partitions.fetch_add(1, Ordering::Relaxed);
        s.fused_attention.fetch_add(attention, Ordering::Relaxed);
        s.fused_softmax_dot.fetch_add(dfl, Ordering::Relaxed);
        Ok(kernel)
    }
}
