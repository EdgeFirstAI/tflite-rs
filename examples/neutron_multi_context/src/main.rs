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

use edgefirst_tflite::{Delegate, Interpreter, Library, Model, Tensor};

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

/// Element width in bytes, derived from the tensor's own byte size.
///
/// `as_slice::<T>` and `copy_from_slice::<T>` are measured in *elements*
/// (`volume`), not bytes, so touching a tensor as `u8` reaches only the
/// first `volume` bytes of a wider buffer. Selecting an unsigned integer of
/// the tensor's own element width covers the full `byte_size()` span for any
/// fixed-width element type.
fn elem_width(byte_size: usize, volume: usize) -> usize {
    byte_size.checked_div(volume).unwrap_or(0)
}

/// Snapshot a tensor's complete data buffer as raw bytes.
fn tensor_bytes(t: &Tensor<'_>) -> Vec<u8> {
    let volume = t.volume().expect("volume");
    match elem_width(t.byte_size(), volume) {
        0 => Vec::new(),
        1 => t.as_slice::<u8>().expect("u8 slice").to_vec(),
        2 => bytes_of(t.as_slice::<u16>().expect("u16 slice")),
        4 => bytes_of(t.as_slice::<u32>().expect("u32 slice")),
        8 => bytes_of(t.as_slice::<u64>().expect("u64 slice")),
        w => panic!("unsupported element width: {w} bytes"),
    }
}

/// Flatten a slice of unsigned integers into their native-endian bytes.
///
/// Comparing floats through their bit patterns is deliberate: the check is
/// for byte-identical buffers, and `NaN != NaN` would otherwise make an
/// unchanged output look like a mismatch.
fn bytes_of<T: Copy + IntoNeBytes>(values: &[T]) -> Vec<u8> {
    values.iter().flat_map(|v| v.into_ne_bytes()).collect()
}

/// Native-endian byte expansion for the unsigned integers used as
/// width-matched stand-ins for the real element type.
trait IntoNeBytes {
    type Bytes: IntoIterator<Item = u8>;
    fn into_ne_bytes(self) -> Self::Bytes;
}

macro_rules! impl_into_ne_bytes {
    ($($ty:ty),+) => {$(
        impl IntoNeBytes for $ty {
            type Bytes = [u8; std::mem::size_of::<$ty>()];
            fn into_ne_bytes(self) -> Self::Bytes {
                self.to_ne_bytes()
            }
        }
    )+};
}
impl_into_ne_bytes!(u16, u32, u64);

/// Snapshot every output tensor as raw bytes for exact comparison.
fn output_bytes(interp: &Interpreter) -> Vec<Vec<u8>> {
    interp
        .outputs()
        .expect("outputs")
        .iter()
        .map(tensor_bytes)
        .collect()
}

/// Fill input 0 with a deterministic byte pattern so every invoke across
/// every context sees identical input.
///
/// Writes the tensor's entire byte buffer; filling only the first `volume`
/// bytes of a wider tensor would leave the remainder holding stale memory,
/// and the "identical input" premise with it.
fn fill_input(interp: &mut Interpreter) {
    let mut inputs = interp.inputs_mut().expect("inputs_mut");
    let t = &mut inputs[0];
    let volume = t.volume().expect("volume");
    let byte = |i: usize| u8::try_from(i * 7 % 251).expect("pattern byte");

    macro_rules! fill {
        ($ty:ty, $w:expr) => {{
            let data: Vec<$ty> = (0..volume)
                .map(|e| <$ty>::from_ne_bytes(std::array::from_fn(|b| byte(e * $w + b))))
                .collect();
            t.copy_from_slice::<$ty>(&data).expect("fill input");
        }};
    }

    match elem_width(t.byte_size(), volume) {
        0 => {}
        1 => fill!(u8, 1),
        2 => fill!(u16, 2),
        4 => fill!(u32, 4),
        8 => fill!(u64, 8),
        w => panic!("unsupported element width: {w} bytes"),
    }
}

/// Total differing bytes between two output snapshots.
///
/// Length mismatches count as differences instead of being truncated away by
/// `zip`: a changed tensor count, or a tensor whose byte size moved, is a
/// difference, not something to pass over.
fn diff_count(a: &[Vec<u8>], b: &[Vec<u8>]) -> usize {
    let paired: usize = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let differing = x.iter().zip(y.iter()).filter(|(p, q)| p != q).count();
            differing + x.len().abs_diff(y.len())
        })
        .sum();
    let unpaired: usize = a
        .iter()
        .skip(b.len())
        .chain(b.iter().skip(a.len()))
        .map(Vec::len)
        .sum();
    paired + unpaired
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
    let ctx1_diff = diff_count(&baseline, &out1);
    let ctx2_diff = diff_count(&baseline, &out2);
    println!("[B4] output diff vs baseline — ctx1: {ctx1_diff} bytes, ctx2: {ctx2_diff} bytes");

    // Destroying a sibling must not disturb a surviving context.
    drop(ctx2);
    let ctx1_dmabuf_final = dmabuf_live(&ctx1);
    println!(
        "[B5] ctx1 dmabuf live after ctx2 drop: {ctx1_dmabuf_final}, mapped tensors: {}",
        mapped_tensors(&ctx1)
    );
    ctx1.invoke().expect("ctx1 invoke after ctx2 drop");
    let out1b = output_bytes(&ctx1);
    let ctx1_diff_after_drop = diff_count(&baseline, &out1b);
    println!("[B6] ctx1 output diff vs baseline after ctx2 drop: {ctx1_diff_after_drop} bytes");
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
    // Byte-identical outputs are part of the contract, not just diagnostics.
    for (what, diff) in [
        (
            "ctx1 output differs from baseline with ctx2 alive",
            ctx1_diff,
        ),
        ("ctx2 output differs from baseline", ctx2_diff),
        (
            "ctx1 output differs from baseline after ctx2 drop",
            ctx1_diff_after_drop,
        ),
    ] {
        if diff > 0 {
            failures.push(format!("{what}: {diff} bytes"));
        }
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
