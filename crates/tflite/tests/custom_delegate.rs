// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Integration tests for in-process custom delegates.
//!
//! The test delegate claims the minimal model's single ADD node and runs it
//! in Rust, so every test checks both that `TFLite` handed the node over and
//! that the Rust kernel's result reached the output tensor. All tests skip
//! at runtime when no `TFLite` shared library is available.

mod common;

use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use edgefirst_tflite::delegate::custom::{
    BoxError, BuiltinOperator, CustomDelegate, GraphView, KernelIo, NodeId, Partition,
    PartitionKernel,
};
use edgefirst_tflite::{Delegate, Interpreter, Library, Profiler, TensorType};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    ClaimNothing,
    CreateFails,
    InvokeFails,
    InvokePanics,
    ClaimOutputTwice,
}

#[derive(Default)]
struct Probe {
    invokes: AtomicUsize,
    prepares: AtomicUsize,
    kernels_created: AtomicUsize,
    kernels_dropped: AtomicUsize,
    delegate_dropped: AtomicBool,
    selected_shapes: Mutex<Vec<Vec<usize>>>,
    partition_nodes: Mutex<Vec<usize>>,
}

struct AddDelegate {
    probe: Arc<Probe>,
    fault: Fault,
}

impl Drop for AddDelegate {
    fn drop(&mut self) {
        self.probe.delegate_dropped.store(true, Ordering::SeqCst);
    }
}

struct AddKernel {
    probe: Arc<Probe>,
    fault: Fault,
    /// Position of the non-constant operand among the partition inputs.
    dynamic_input: usize,
    constant: Vec<f32>,
}

impl Drop for AddKernel {
    fn drop(&mut self) {
        self.probe.kernels_dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl CustomDelegate for AddDelegate {
    type Kernel = AddKernel;

    fn name(&self) -> &CStr {
        c"TestAddDelegate"
    }

    fn select_nodes(&self, graph: &GraphView<'_>) -> Result<Vec<NodeId>, BoxError> {
        if self.fault == Fault::ClaimNothing {
            return Ok(Vec::new());
        }
        let mut claimed = Vec::new();
        for id in graph.execution_plan()? {
            let node = graph.node(id)?;
            if node.builtin_op() != BuiltinOperator::ADD {
                continue;
            }
            let mut shapes = self.probe.selected_shapes.lock().unwrap();
            for tensor in node.inputs().iter().flatten() {
                let info = graph.tensor(*tensor)?;
                assert_eq!(info.tensor_type(), TensorType::Float32);
                shapes.push(info.shape()?);
            }
            claimed.push(id);
        }
        Ok(claimed)
    }

    fn create_kernel(&self, partition: &Partition<'_>) -> Result<AddKernel, BoxError> {
        if self.fault == Fault::CreateFails {
            return Err("kernel creation refused by test".into());
        }
        *self.probe.partition_nodes.lock().unwrap() =
            partition.nodes().iter().map(|n| n.index()).collect();

        let graph = partition.graph();
        let mut dynamic_input = None;
        let mut constant = None;
        for (pos, tensor) in partition.input_tensors().iter().enumerate() {
            let info = graph.tensor(*tensor)?;
            match info.constant_data::<f32>()? {
                Some(values) => constant = Some(values.to_vec()),
                None => dynamic_input = Some(pos),
            }
        }
        self.probe.kernels_created.fetch_add(1, Ordering::SeqCst);
        Ok(AddKernel {
            probe: Arc::clone(&self.probe),
            fault: self.fault,
            dynamic_input: dynamic_input.ok_or("ADD has no dynamic input")?,
            constant: constant.ok_or("ADD has no constant input")?,
        })
    }
}

impl PartitionKernel for AddKernel {
    fn prepare(&mut self, io: &mut KernelIo<'_>) -> Result<(), BoxError> {
        self.probe.prepares.fetch_add(1, Ordering::SeqCst);
        if io.num_inputs() == 0 || io.num_outputs() != 1 {
            return Err("unexpected partition arity".into());
        }
        Ok(())
    }

    fn invoke(&mut self, io: &mut KernelIo<'_>) -> Result<(), BoxError> {
        self.probe.invokes.fetch_add(1, Ordering::SeqCst);
        match self.fault {
            Fault::InvokeFails => return Err("invoke refused by test".into()),
            Fault::InvokePanics => panic!("invoke panicked in test"),
            _ => {}
        }
        let input = io.input(self.dynamic_input)?;
        let x = input.as_slice::<f32>()?;
        let mut output = io.output(0)?;
        if self.fault == Fault::ClaimOutputTwice {
            io.output(0)?;
        }
        let y = output.as_mut_slice::<f32>()?;
        for ((y, x), c) in y.iter_mut().zip(x).zip(&self.constant) {
            *y = x + c;
        }
        Ok(())
    }
}

/// Skip the calling test when the runtime lacks the opaque delegate API (the
/// vendored host library is a `CMake` build that leaves it out).
macro_rules! require_opaque_delegates {
    ($lib:expr) => {
        match Delegate::from_custom(
            &$lib,
            AddDelegate {
                probe: Arc::new(Probe::default()),
                fault: Fault::ClaimNothing,
            },
        ) {
            Ok(_) => {}
            Err(e) if e.is_unsupported() => {
                common::skip(format_args!(
                    "TFLite runtime lacks the opaque delegate API ({:?} missing)",
                    e.unsupported_api()
                ));
                return;
            }
            Err(e) => panic!("custom delegate creation failed: {e}"),
        }
    };
}

fn build_with<'lib>(
    lib: &'lib Library,
    model: &edgefirst_tflite::Model<'lib>,
    fault: Fault,
    probe: &Arc<Probe>,
) -> edgefirst_tflite::error::Result<Interpreter<'lib>> {
    let delegate = Delegate::from_custom(
        lib,
        AddDelegate {
            probe: Arc::clone(probe),
            fault,
        },
    )?;
    Interpreter::builder(lib)?
        .num_threads(1)
        .delegate(delegate)
        .build(model)
}

fn run(interp: &mut Interpreter<'_>) -> edgefirst_tflite::error::Result<Vec<f32>> {
    {
        let mut inputs = interp.inputs_mut()?;
        inputs[0]
            .as_mut_slice::<f32>()?
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    }
    interp.invoke()?;
    let outputs = interp.outputs()?;
    Ok(outputs[0].as_slice::<f32>()?.to_vec())
}

#[test]
fn custom_delegate_runs_claimed_add() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_opaque_delegates!(lib);
    let model = common::load_model(&lib);
    let probe = Arc::new(Probe::default());

    let mut interp = build_with(&lib, &model, Fault::None, &probe).unwrap();
    assert_eq!(run(&mut interp).unwrap(), [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(run(&mut interp).unwrap(), [2.0, 3.0, 4.0, 5.0]);

    assert_eq!(probe.invokes.load(Ordering::SeqCst), 2);
    assert_eq!(probe.kernels_created.load(Ordering::SeqCst), 1);
    assert!(probe.prepares.load(Ordering::SeqCst) >= 1);
    assert_eq!(probe.partition_nodes.lock().unwrap().len(), 1);
    assert!(probe
        .selected_shapes
        .lock()
        .unwrap()
        .iter()
        .any(|s| s == &[1, 4]));
}

#[test]
fn custom_delegate_claiming_nothing_keeps_builtin_kernels() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_opaque_delegates!(lib);
    let model = common::load_model(&lib);
    let probe = Arc::new(Probe::default());

    let mut interp = build_with(&lib, &model, Fault::ClaimNothing, &probe).unwrap();
    assert_eq!(run(&mut interp).unwrap(), [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(probe.kernels_created.load(Ordering::SeqCst), 0);
    assert_eq!(probe.invokes.load(Ordering::SeqCst), 0);
}

#[test]
fn custom_delegate_kernel_creation_error_fails_build() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_opaque_delegates!(lib);
    let model = common::load_model(&lib);
    let probe = Arc::new(Probe::default());

    assert!(build_with(&lib, &model, Fault::CreateFails, &probe).is_err());
}

#[test]
fn custom_delegate_invoke_error_is_reported() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_opaque_delegates!(lib);
    let model = common::load_model(&lib);
    let probe = Arc::new(Probe::default());

    let mut interp = build_with(&lib, &model, Fault::InvokeFails, &probe).unwrap();
    assert!(run(&mut interp).is_err());
}

#[test]
fn custom_delegate_invoke_panic_is_contained() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_opaque_delegates!(lib);
    let model = common::load_model(&lib);
    let probe = Arc::new(Probe::default());

    let mut interp = build_with(&lib, &model, Fault::InvokePanics, &probe).unwrap();
    assert!(run(&mut interp).is_err());
}

#[test]
fn custom_delegate_rejects_second_borrow_of_an_output() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_opaque_delegates!(lib);
    let model = common::load_model(&lib);
    let probe = Arc::new(Probe::default());

    let mut interp = build_with(&lib, &model, Fault::ClaimOutputTwice, &probe).unwrap();
    assert!(run(&mut interp).is_err());
}

#[test]
fn custom_delegate_state_outlives_interpreter() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_opaque_delegates!(lib);
    let model = common::load_model(&lib);
    let probe = Arc::new(Probe::default());

    let mut interp = build_with(&lib, &model, Fault::None, &probe).unwrap();
    run(&mut interp).unwrap();
    assert!(!probe.delegate_dropped.load(Ordering::SeqCst));
    assert_eq!(probe.kernels_dropped.load(Ordering::SeqCst), 0);

    drop(interp);
    assert_eq!(probe.kernels_dropped.load(Ordering::SeqCst), 1);
    assert!(probe.delegate_dropped.load(Ordering::SeqCst));
}

/// `TFLite` 2.19 names opaque-delegate partitions after their builtin code in
/// profiler events; the delegate's own name only appears in error messages.
#[test]
fn custom_delegate_partition_is_reported_to_profiler() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_opaque_delegates!(lib);
    let model = common::load_model(&lib);
    let probe = Arc::new(Probe::default());
    let profiler = Profiler::new();

    let delegate = Delegate::from_custom(
        &lib,
        AddDelegate {
            probe: Arc::clone(&probe),
            fault: Fault::None,
        },
    )
    .unwrap();
    let Ok(builder) = Interpreter::builder(&lib).unwrap().profiler(&profiler) else {
        common::skip("telemetry profiler not exported by this TFLite build");
        return;
    };
    let mut interp = builder.delegate(delegate).build(&model).unwrap();
    profiler.clear();
    run(&mut interp).unwrap();

    let names: Vec<String> = profiler.events().into_iter().map(|e| e.op_name).collect();
    assert_eq!(
        names,
        ["DELEGATE"],
        "one event for the single claimed partition"
    );
    assert_eq!(probe.invokes.load(Ordering::SeqCst), 1);
}

#[test]
fn custom_delegate_names_missing_runtime_support() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let result = Delegate::from_custom(
        &lib,
        AddDelegate {
            probe: Arc::new(Probe::default()),
            fault: Fault::None,
        },
    );
    match result {
        Ok(delegate) => assert!(!delegate.as_ptr().is_null()),
        Err(e) => {
            assert!(e.is_unsupported(), "unexpected error: {e}");
            assert!(e.unsupported_api().is_some_and(|s| s.starts_with("TfLite")));
        }
    }
}

/// Order in which the NPU delegate and the custom delegate are applied.
#[derive(Clone, Copy, Debug)]
enum Stacking {
    /// i.MX 95: the custom delegate takes what Neutron leaves on the CPU.
    NpuFirst,
    /// i.MX 8M Plus: the custom delegate claims its nodes before VX sees them.
    CustomFirst,
}

/// Build with the board's NPU delegate (`TFLITE_TEST_NPU_DELEGATE`) and the
/// test delegate in the given order, and check the model still computes
/// x + const. Skips when the variable is unset.
fn stack_with_npu(stacking: Stacking) {
    common::require_tflite!();
    let Ok(npu_path) = std::env::var("TFLITE_TEST_NPU_DELEGATE") else {
        common::skip("TFLITE_TEST_NPU_DELEGATE not set");
        return;
    };
    let lib = common::load_library().unwrap();
    require_opaque_delegates!(lib);
    let model = common::load_model(&lib);
    let probe = Arc::new(Probe::default());

    let npu = Delegate::load(&npu_path)
        .unwrap_or_else(|e| panic!("loading NPU delegate {npu_path}: {e}"));
    let custom = Delegate::from_custom(
        &lib,
        AddDelegate {
            probe: Arc::clone(&probe),
            fault: Fault::None,
        },
    )
    .unwrap();
    let (first, second) = match stacking {
        Stacking::NpuFirst => (npu, custom),
        Stacking::CustomFirst => (custom, npu),
    };
    let mut interp = Interpreter::builder(&lib)
        .unwrap()
        .num_threads(1)
        .delegate(first)
        .delegate(second)
        .build(&model)
        .unwrap_or_else(|e| panic!("{stacking:?}: build failed: {e}"));

    assert_eq!(run(&mut interp).unwrap(), [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(run(&mut interp).unwrap(), [2.0, 3.0, 4.0, 5.0]);

    let created = probe.kernels_created.load(Ordering::SeqCst);
    match stacking {
        // The ADD is still in the plan when the custom delegate runs first.
        Stacking::CustomFirst => assert_eq!(created, 1, "custom delegate did not claim ADD"),
        // The NPU delegate may or may not have taken the float ADD.
        Stacking::NpuFirst => assert!(created <= 1),
    }
    assert_eq!(probe.invokes.load(Ordering::SeqCst), 2 * created);
}

#[test]
fn custom_delegate_runs_after_npu_delegate() {
    stack_with_npu(Stacking::NpuFirst);
}

#[test]
fn custom_delegate_runs_before_npu_delegate() {
    stack_with_npu(Stacking::CustomFirst);
}
