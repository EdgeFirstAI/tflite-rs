// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! The `EdgeFirst` delegate against `TFLite`'s own kernels.
//!
//! Each test runs a model with and without the delegate on the same input and compares every
//! output. On the scalar kernel tier the delegate must match `TFLite` bit for bit; on NEON,
//! int8 `SOFTMAX` may differ by 1 LSB.
//!
//! The runtime needs the opaque delegate API, which the vendored host library lacks: these
//! tests skip there and run in the board CI lane. Set `TFLITE_TEST_NPU_DELEGATE` to also stack
//! the delegate with an NPU delegate.

mod common;

use edgefirst_tflite::{Delegate, Interpreter, Library, Model};
use edgefirst_tflite_delegate::{CpuFeatures, EdgeFirstDelegate, Options, Scope, Stats};

const SOFTMAX_INT8: &str = "../../testdata/softmax_int8.tflite";
const SOFTMAX_INT16: &str = "../../testdata/softmax_int16.tflite";
const YOLOV8: &str = "../../testdata/yolov8n-t-b86_quant-u8-i8_combined.tflite";

// The model zoo's exports (AGPL-3.0, so not in the repository; the board lane downloads them,
// see TESTING.md). The `.imx95` ones are compiled with Neutron 3.2.3 and leave their attention
// blocks and decode tails on the CPU.
const YOLOV8_NEUTRON: &str = "../../testdata/yolov8n-det-int8.imx95.tflite";
const YOLO11_NEUTRON: &str = "../../testdata/yolo11n-det-int8.imx95.tflite";
const YOLO26_NEUTRON: &str = "../../testdata/yolo26n-det-int8.imx95.tflite";
const YOLO11: &str = "../../testdata/yolo11n-det-int8.tflite";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernels {
    /// The scalar tier, which must match `TFLite` bit for bit.
    Scalar,
    /// The best tier this CPU has.
    Best,
}

impl Kernels {
    fn features(self) -> CpuFeatures {
        match self {
            Self::Scalar => CpuFeatures::from_forced_tier(Some("scalar")).unwrap(),
            Self::Best => CpuFeatures::probe(),
        }
    }

    fn delegate(self) -> EdgeFirstDelegate {
        EdgeFirstDelegate::with_cpu_features(self.features())
    }

    /// Largest output difference allowed against `TFLite`, in LSB.
    fn tolerance(self) -> u32 {
        match self {
            Self::Scalar => 0,
            Self::Best => 1,
        }
    }
}

/// Deterministic pseudo-random bytes (xorshift32).
fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s.to_le_bytes()[0]
        })
        .collect()
}

/// Fill every input with `pattern`, invoke, and return every output's bytes.
fn run(interp: &mut Interpreter<'_>, seed: u32) -> Vec<Vec<u8>> {
    for (i, mut input) in interp.inputs_mut().unwrap().into_iter().enumerate() {
        let bytes = input.as_bytes_mut().unwrap();
        let fill = pattern(bytes.len(), seed + u32::try_from(i).unwrap());
        bytes.copy_from_slice(&fill);
    }
    interp.invoke().unwrap();
    interp
        .outputs()
        .unwrap()
        .iter()
        .map(|t| t.as_bytes().unwrap().to_vec())
        .collect()
}

/// Largest elementwise difference between two outputs of element width `width`.
fn max_diff(a: &[u8], b: &[u8], width: usize) -> u32 {
    assert_eq!(a.len(), b.len());
    match width {
        1 => a
            .iter()
            .zip(b)
            .map(|(&x, &y)| u32::from(x.cast_signed().abs_diff(y.cast_signed())))
            .max()
            .unwrap_or(0),
        2 => a
            .as_chunks::<2>()
            .0
            .iter()
            .zip(b.as_chunks::<2>().0)
            .map(|(&x, &y)| u32::from(i16::from_le_bytes(x).abs_diff(i16::from_le_bytes(y))))
            .max()
            .unwrap_or(0),
        _ => panic!("unexpected element width {width}"),
    }
}

/// Assert output `i` of `got` is within `tolerances[i]` LSB of `want` (the last entry for
/// the outputs past the end).
fn assert_matches(
    want: &[Vec<u8>],
    got: &[Vec<u8>],
    width: usize,
    tolerances: &[u32],
    context: &str,
) {
    assert_eq!(want.len(), got.len(), "{context}: output count");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let diff = max_diff(w, g, width);
        let tolerance = tolerances[i.min(tolerances.len() - 1)];
        assert!(
            diff <= tolerance,
            "{context}: output {i} differs by {diff} LSB (tolerance {tolerance})"
        );
    }
}

/// Wrap `delegate`, or report a skip when the runtime lacks the opaque API.
fn wrap(lib: &Library, delegate: EdgeFirstDelegate) -> Option<Delegate> {
    match delegate.into_delegate(lib) {
        Ok(d) => Some(d),
        Err(e) if e.is_unsupported() => {
            common::skip(format_args!(
                "TFLite runtime lacks the opaque delegate API: {e}"
            ));
            None
        }
        Err(e) => panic!("creating the EdgeFirst delegate: {e}"),
    }
}

fn build<'lib>(
    lib: &'lib Library,
    model: &Model<'lib>,
    delegates: Vec<Delegate>,
) -> Interpreter<'lib> {
    let mut builder = Interpreter::builder(lib).unwrap().num_threads(1);
    for d in delegates {
        builder = builder.delegate(d);
    }
    builder.build(model).unwrap()
}

fn load_model<'lib>(lib: &'lib Library, path: &str) -> Model<'lib> {
    Model::from_file(lib, path).unwrap_or_else(|e| panic!("{path}: {e} (see TESTING.md)"))
}

/// Compare `model` run by `TFLite` alone and with the delegate added last.
/// Returns the delegate's stats, or `None` if the test was skipped.
fn compare(path: &str, kernels: Kernels, width: usize, tolerance: u32) -> Option<Stats> {
    let lib = common::require_library()?;
    let model = load_model(&lib, path);
    let delegate = kernels.delegate();
    let stats = delegate.stats();
    let edgefirst = wrap(&lib, delegate)?;

    let mut reference = build(&lib, &model, Vec::new());
    let mut delegated = build(&lib, &model, vec![edgefirst]);
    for seed in [1, 2] {
        assert_matches(
            &run(&mut reference, seed),
            &run(&mut delegated, seed),
            width,
            &[tolerance],
            &format!("{path} {kernels:?} seed {seed}"),
        );
    }
    Some(stats)
}

#[test]
fn int8_region_scalar_matches_tflite() {
    let Some(stats) = compare(SOFTMAX_INT8, Kernels::Scalar, 1, 0) else {
        return;
    };
    // SOFTMAX, MUL by a scalar, SOFTMAX and ADD: one region.
    assert_eq!(stats.claimed_nodes(), 4);
    assert_eq!(stats.partitions(), 1);
}

#[test]
fn int8_region_best_tier_within_one_lsb() {
    let Some(stats) = compare(SOFTMAX_INT8, Kernels::Best, 1, 1) else {
        return;
    };
    assert_eq!(stats.claimed_nodes(), 4);
}

#[test]
fn int16_softmax_matches_tflite_on_every_tier() {
    for kernels in [Kernels::Scalar, Kernels::Best] {
        let Some(stats) = compare(SOFTMAX_INT16, kernels, 2, kernels.tolerance()) else {
            return;
        };
        // int16 MUL and ADD stay with TFLite.
        assert_eq!(stats.claimed_nodes(), 2, "{kernels:?}");
        assert_eq!(stats.partitions(), 2, "{kernels:?}");
    }
}

#[test]
fn yolov8_cpu_scalar_matches_tflite() {
    let Some(stats) = compare(YOLOV8, Kernels::Scalar, 1, 0) else {
        return;
    };
    // The backbone's SiLU LOGISTIC and MUL, and the decode tail.
    assert!(stats.claimed_nodes() > 100, "{}", stats.claimed_nodes());
}

/// NEON int8 `SOFTMAX` is within 1 LSB; the decode layers after it widen that to 2.
#[test]
fn yolov8_cpu_best_tier_within_two_lsb() {
    let Some(stats) = compare(YOLOV8, Kernels::Best, 1, 2) else {
        return;
    };
    assert!(stats.claimed_nodes() > 100, "{}", stats.claimed_nodes());
}

/// The NPU delegate under test, or `None` after reporting a skip.
fn npu_delegate_path() -> Option<String> {
    let path = std::env::var("TFLITE_TEST_NPU_DELEGATE").ok();
    if path.is_none() {
        common::skip("TFLITE_TEST_NPU_DELEGATE not set");
    }
    path
}

fn is_neutron(npu: &str) -> bool {
    npu.contains("neutron")
}

/// The YOLO exports' box output, which a DFL head feeds.
const BOXES: usize = 0;

/// A fused DFL head skips `TFLite`'s 8-bit probabilities, so the boxes it feeds sit within a
/// few LSB of `TFLite`'s (up to 5 on the model zoo's `YOLOv8n` and `YOLO11n`).
const DFL_BOXES_LSB: u32 = 8;

/// After the NPU delegate, the delegate takes the CPU fallback and must change no result by
/// more than `tolerance` LSB, besides the boxes of a fused DFL head. Returns its stats.
fn after_npu(npu: &str, path: &str, delegate: EdgeFirstDelegate, tolerance: u32) -> Option<Stats> {
    let lib = common::require_library()?;
    let model = load_model(&lib, path);
    let stats = delegate.stats();
    let edgefirst = wrap(&lib, delegate)?;
    let mut reference = build(&lib, &model, vec![Delegate::load(npu).unwrap()]);
    let mut stacked = build(&lib, &model, vec![Delegate::load(npu).unwrap(), edgefirst]);
    let mut tolerances = vec![tolerance; BOXES + 2];
    if stats.fused_softmax_dot() > 0 {
        tolerances[BOXES] = tolerance.max(DFL_BOXES_LSB);
    }
    for seed in [1, 2] {
        let want = run(&mut reference, seed);
        assert_eq!(
            run(&mut reference, seed),
            want,
            "{path}: the NPU alone is not deterministic"
        );
        assert_matches(
            &want,
            &run(&mut stacked, seed),
            1,
            &tolerances,
            &format!("{path} seed {seed}"),
        );
    }
    Some(stats)
}

#[test]
fn yolov8_after_npu_matches_npu_alone() {
    let Some(npu) = npu_delegate_path() else {
        return;
    };
    let path = if is_neutron(&npu) {
        YOLOV8_NEUTRON
    } else {
        YOLOV8
    };
    let Some(stats) = after_npu(&npu, path, Kernels::Scalar.delegate(), 0) else {
        return;
    };
    if is_neutron(&npu) {
        assert!(
            stats.claimed_nodes() >= 20,
            "the decode tail: {}",
            stats.claimed_nodes()
        );
    }
}

/// `YOLO11` and `YOLO26` after Neutron: the attention blocks run fused and bit for bit (the
/// boxes of `YOLO11`'s fused DFL head aside).
#[test]
fn attention_after_neutron_matches_neutron_alone() {
    let Some(npu) = npu_delegate_path() else {
        return;
    };
    if !is_neutron(&npu) {
        // VX compiles these models whole; `attention_before_npu_is_fused` covers it.
        return;
    }
    // (model, attention blocks, DFL heads)
    for (path, blocks, dfl) in [(YOLO11_NEUTRON, 1, 1), (YOLO26_NEUTRON, 2, 0)] {
        let Some(stats) = after_npu(&npu, path, Kernels::Scalar.delegate(), 0) else {
            return;
        };
        assert_eq!(stats.fused_attention(), blocks, "{path}");
        assert_eq!(stats.fused_softmax_dot(), dfl, "{path}");
        // NEON int8 SOFTMAX is within 1 LSB; the layers after it can widen that.
        let Some(stats) = after_npu(&npu, path, Kernels::Best.delegate(), 4) else {
            return;
        };
        assert_eq!(stats.fused_attention(), blocks, "{path}");
    }
}

/// Before the NPU delegate, with [`Scope::Attention`], the delegate takes only the attention
/// block and the NPU compiles the rest.
#[test]
fn attention_before_npu_is_fused() {
    let Some(npu) = npu_delegate_path() else {
        return;
    };
    let Some(lib) = common::require_library() else {
        return;
    };
    // Neutron needs models converted offline, whose `NeutronGraph` operators only resolve
    // when Neutron goes first; it takes nothing of the original model.
    let model = load_model(&lib, YOLO11);
    let delegate = Kernels::Best
        .delegate()
        .with_options(Options::default().scope(Scope::Attention));
    let stats = delegate.stats();
    let Some(edgefirst) = wrap(&lib, delegate) else {
        return;
    };
    let mut stacked = build(&lib, &model, vec![edgefirst, Delegate::load(&npu).unwrap()]);
    let first = run(&mut stacked, 1);
    assert_matches(&first, &run(&mut stacked, 1), 1, &[0], "repeat run");
    assert_eq!(
        stats.claimed_nodes(),
        4,
        "MUL, BATCH_MATMUL, SOFTMAX, BATCH_MATMUL"
    );
    assert_eq!(stats.fused_attention(), 1);
}
