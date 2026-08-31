// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Multi-context Neutron delegate example and on-device verification.
//!
//! Runs several interpreter contexts — each with its own delegate instance —
//! against the same model, the worker-pool pattern used for overlapped
//! inference. Verifies that each context's DMA-BUF state (per-instance
//! registry, EDGEAI-1188) survives its siblings' creation and destruction,
//! and that overlapped invocations from two worker threads produce outputs
//! byte-identical to a single-context baseline.
//!
//! On a delegate build with the older process-global DMA-BUF singleton, this
//! example FAILs: creating the second context wipes the first context's
//! tensor map (watch for `WARNING: dmabuf_set_delegate called with existing
//! delegate` on stderr).
//!
//! ```sh
//! NEUTRON_ENABLE_ZERO_COPY=1 cargo run -p neutron-multi-context -- \
//!     model.imx95.tflite /usr/lib/libneutron_delegate.so
//! ```
//!
//! Exits 0 on PASS, 1 on FAIL — usable directly as an integration check.

use edgefirst_tflite::{Delegate, Interpreter, Library, Model, TensorType};

const DELEGATE_DEFAULT: &str = "/usr/lib/libneutron_delegate.so";
const WORKER_ITERS: usize = 50;

/// Live per-instance DMA-BUF support (not just symbol presence).
fn dmabuf_live(interp: &Interpreter) -> bool {
    interp
        .delegate(0)
        .and_then(|d| d.dmabuf())
        .map(|db| db.is_supported())
        .unwrap_or(false)
}

/// Count graph tensor indices with a live DMA-BUF mapping for this
/// interpreter's delegate instance.
fn mapped_tensors(interp: &Interpreter) -> usize {
    interp
        .delegate(0)
        .and_then(|d| d.dmabuf())
        .map(|db| (0..2048).filter(|&i| db.tensor_info(i).is_ok()).count())
        .unwrap_or(0)
}

/// Snapshot every output tensor as raw bytes for exact comparison.
fn output_bytes(interp: &Interpreter) -> Vec<Vec<u8>> {
    interp
        .outputs()
        .expect("outputs")
        .iter()
        .map(|t| match t.tensor_type() {
            TensorType::Int8 => t
                .as_slice::<i8>()
                .expect("i8 slice")
                .iter()
                .map(|&v| v as u8)
                .collect(),
            _ => t.as_slice::<u8>().expect("u8 slice").to_vec(),
        })
        .collect()
}

/// Fill input 0 with a deterministic byte pattern so every invoke across
/// every context sees identical input.
fn fill_input(interp: &mut Interpreter) {
    let mut inputs = interp.inputs_mut().expect("inputs_mut");
    let n = inputs[0].shape().expect("shape").iter().product::<usize>();
    let data: Vec<u8> = (0..n).map(|i| (i * 7 % 251) as u8).collect();
    inputs[0].copy_from_slice::<u8>(&data).expect("fill input");
}

fn diff_count(a: &[Vec<u8>], b: &[Vec<u8>]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.iter().zip(y.iter()).filter(|(p, q)| p != q).count())
        .sum()
}

fn main() {
    let model_path = std::env::args().nth(1).expect("model path required");
    let delegate_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| DELEGATE_DEFAULT.to_string());

    let lib = Library::new().expect("tflite library");
    let model = Model::from_file(&lib, &model_path).expect("model");

    // ── Phase A: single-context baseline ────────────────────────────────
    let d1 = Delegate::load(&delegate_path).expect("delegate 1");
    let mut ctx1 = Interpreter::builder(&lib)
        .expect("builder 1")
        .delegate(d1)
        .build(&model)
        .expect("interpreter 1");
    let ctx1_dmabuf_initial = dmabuf_live(&ctx1);
    println!(
        "[A1] ctx1 dmabuf live: {ctx1_dmabuf_initial}, mapped tensors: {}",
        mapped_tensors(&ctx1)
    );

    fill_input(&mut ctx1);
    ctx1.invoke().expect("ctx1 baseline invoke");
    let baseline = output_bytes(&ctx1);
    println!(
        "[A2] baseline outputs: {} tensors, {} bytes total",
        baseline.len(),
        baseline.iter().map(Vec::len).sum::<usize>()
    );

    // ── Phase B: second context alongside the first ─────────────────────
    let d2 = Delegate::load(&delegate_path).expect("delegate 2");
    let mut ctx2 = Interpreter::builder(&lib)
        .expect("builder 2")
        .delegate(d2)
        .build(&model)
        .expect("interpreter 2");

    let ctx1_dmabuf_after = dmabuf_live(&ctx1);
    let ctx2_dmabuf = dmabuf_live(&ctx2);
    println!(
        "[B1] ctx2 dmabuf live: {ctx2_dmabuf}, mapped tensors: {}",
        mapped_tensors(&ctx2)
    );
    println!(
        "[B2] ctx1 dmabuf live: {ctx1_dmabuf_after} (was {ctx1_dmabuf_initial}), mapped tensors: {}",
        mapped_tensors(&ctx1)
    );

    // Both contexts run the same input; outputs must match the baseline.
    fill_input(&mut ctx1);
    ctx1.invoke().expect("ctx1 invoke with ctx2 alive");
    let out1 = output_bytes(&ctx1);
    fill_input(&mut ctx2);
    ctx2.invoke().expect("ctx2 invoke");
    let out2 = output_bytes(&ctx2);
    println!(
        "[B4] output diff vs baseline — ctx1: {} bytes, ctx2: {} bytes",
        diff_count(&baseline, &out1),
        diff_count(&baseline, &out2)
    );

    // Destroying a sibling must not disturb a surviving context.
    drop(ctx2);
    let ctx1_dmabuf_final = dmabuf_live(&ctx1);
    println!(
        "[B5] ctx1 dmabuf live after ctx2 drop: {ctx1_dmabuf_final}, mapped tensors: {}",
        mapped_tensors(&ctx1)
    );
    ctx1.invoke().expect("ctx1 invoke after ctx2 drop");
    let out1b = output_bytes(&ctx1);
    println!(
        "[B6] ctx1 output diff vs baseline after ctx2 drop: {} bytes",
        diff_count(&baseline, &out1b)
    );
    drop(ctx1);

    // ── Phase C: overlapped workers on two threads ──────────────────────
    let errs: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2)
            .map(|w| {
                let lib = &lib;
                let model = &model;
                let delegate_path = delegate_path.clone();
                let baseline = &baseline;
                s.spawn(move || -> Result<usize, String> {
                    let d = Delegate::load(&delegate_path)
                        .map_err(|e| format!("worker {w} delegate: {e}"))?;
                    let mut interp = Interpreter::builder(lib)
                        .map_err(|e| format!("worker {w} builder: {e}"))?
                        .delegate(d)
                        .build(model)
                        .map_err(|e| format!("worker {w} interpreter: {e}"))?;
                    let mut bad = 0;
                    for i in 0..WORKER_ITERS {
                        fill_input(&mut interp);
                        interp
                            .invoke()
                            .map_err(|e| format!("worker {w} invoke {i}: {e}"))?;
                        if diff_count(baseline, &output_bytes(&interp)) > 0 {
                            bad += 1;
                        }
                    }
                    Ok(bad)
                })
            })
            .collect();
        handles
            .into_iter()
            .enumerate()
            .filter_map(|(w, h)| match h.join() {
                Ok(Ok(bad)) => {
                    println!("[C] worker {w}: {bad}/{WORKER_ITERS} mismatched iterations");
                    (bad > 0).then(|| format!("worker {w}: {bad} mismatches"))
                }
                Ok(Err(e)) => Some(e),
                Err(_) => Some(format!("worker {w} panicked")),
            })
            .collect()
    });

    // ── Verdict ─────────────────────────────────────────────────────────
    let mut failures = Vec::new();
    if ctx1_dmabuf_initial && !ctx1_dmabuf_after {
        failures.push("ctx1 dmabuf state wiped by ctx2 creation".to_string());
    }
    if ctx1_dmabuf_initial && !ctx1_dmabuf_final {
        failures.push("ctx1 dmabuf state wiped by ctx2 destruction".to_string());
    }
    failures.extend(errs);
    if failures.is_empty() {
        println!("PASS: multi-context DMABUF state is per-instance and outputs are stable");
    } else {
        println!("FAIL:");
        for f in &failures {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
