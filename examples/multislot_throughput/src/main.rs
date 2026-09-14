// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Multi-context throughput benchmark.
//!
//! Measures inference throughput for a delegate under three sharing models,
//! to answer whether concurrent multi-context invocation gains throughput on
//! a single NPU (e.g. VX / Vivante on i.MX 8M Plus) or merely serializes:
//!
//!   * **S**  — sequential baseline: one interpreter, one thread, back-to-back
//!     invokes.
//!   * **M(N)** — multi-context: N interpreters built from **one shared
//!     `Model`** (shared flatbuffer weights; each interpreter keeps its own
//!     tensor arena and its own delegate instance), N worker threads invoking
//!     concurrently.
//!
//! Single-interpreter multi-thread (concurrent `Invoke` on ONE interpreter) is
//! deliberately NOT benchmarked: TFLite forbids it and the crate encodes that
//! rule as `Interpreter: Send + !Sync`. `Model` is `Send + Sync`, so the
//! supported "shared weights" form is the M(N) configuration above.
//!
//! Graph compilation and warmup are excluded from the measured window; timing
//! is duration-based (fixed wall-clock window, count completed invokes) to
//! avoid per-thread tail skew.
//!
//! ```sh
//! # VX (imx8mp):     model.tflite       /usr/lib/libvx_delegate.so
//! # Neutron (imx95): model.imx95.tflite /usr/lib/libneutron_delegate.so
//! multislot-throughput <model> <delegate> [window_secs] [warmup_iters]
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;
use std::time::{Duration, Instant};

use edgefirst_tflite::{Delegate, Interpreter, Library, Model, Tensor};

const DEFAULT_WINDOW_SECS: u64 = 4;
const DEFAULT_WARMUP: usize = 5;
const SLOT_COUNTS: [usize; 3] = [1, 2, 4];

/// Element width in bytes, derived from the tensor's own byte size, so a fill
/// touches the whole buffer regardless of element type (`as_slice`/
/// `copy_from_slice` count elements, not bytes).
fn elem_width(byte_size: usize, volume: usize) -> usize {
    byte_size.checked_div(volume).unwrap_or(0)
}

/// Fill input 0 with a deterministic byte pattern (whole buffer).
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

/// Report whether the crate resolved a live DMA-BUF path for this interpreter.
fn dmabuf_live(interp: &Interpreter) -> bool {
    interp
        .delegate(0)
        .and_then(|d| d.dmabuf())
        .map(|db| db.is_supported())
        .unwrap_or(false)
}

/// Build one interpreter for `model` with a fresh delegate instance.
fn build_interp<'lib>(
    lib: &'lib Library,
    model: &Model<'lib>,
    delegate_path: &str,
) -> Interpreter<'lib> {
    let d = Delegate::load(delegate_path).expect("delegate load");
    Interpreter::builder(lib)
        .expect("builder")
        .delegate(d)
        .build(model)
        .expect("interpreter build")
}

/// One benchmarked configuration's result.
struct SlotResult {
    slots: usize,
    build_secs: f64,
    invokes: u64,
    window_secs: f64,
    /// Aggregate throughput across all slots.
    fps: f64,
    /// Mean per-invoke latency observed by a worker (window / its own count).
    mean_latency_ms: f64,
    dmabuf: bool,
}

/// Run one configuration of `n` concurrent interpreters (all from the shared
/// `model`) for a fixed wall-clock window; return aggregate throughput.
fn run_config(
    lib: &Library,
    model: &Model,
    delegate_path: &str,
    n: usize,
    warmup: usize,
    window: Duration,
) -> SlotResult {
    // Build + warm every interpreter first (graph compile is excluded from the
    // measured window). Each owns its own delegate instance; all share `model`.
    let t_build = Instant::now();
    let mut interps: Vec<Interpreter> = (0..n)
        .map(|_| {
            let mut it = build_interp(lib, model, delegate_path);
            fill_input(&mut it);
            for _ in 0..warmup {
                it.invoke().expect("warmup invoke");
            }
            it
        })
        .collect();
    let build_secs = t_build.elapsed().as_secs_f64();
    let dmabuf = interps.first().map(dmabuf_live).unwrap_or(false);

    let start = Barrier::new(n + 1);
    let stop = AtomicBool::new(false);

    let (counts, elapsed) = std::thread::scope(|s| {
        let start = &start;
        let stop = &stop;
        let handles: Vec<_> = interps
            .iter_mut()
            .map(|it| {
                s.spawn(move || -> u64 {
                    start.wait();
                    let mut count = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        it.invoke().expect("invoke");
                        count += 1;
                    }
                    count
                })
            })
            .collect();

        start.wait();
        let t0 = Instant::now();
        std::thread::sleep(window);
        stop.store(true, Ordering::Relaxed);
        let counts: Vec<u64> = handles
            .into_iter()
            .map(|h| h.join().expect("join"))
            .collect();
        (counts, t0.elapsed().as_secs_f64())
    });

    let invokes: u64 = counts.iter().sum();
    let fps = invokes as f64 / elapsed;
    // Mean of each worker's own per-invoke latency (window / its count).
    let mean_latency_ms = {
        let per: Vec<f64> = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| elapsed * 1000.0 / c as f64)
            .collect();
        if per.is_empty() {
            0.0
        } else {
            per.iter().sum::<f64>() / per.len() as f64
        }
    };

    SlotResult {
        slots: n,
        build_secs,
        invokes,
        window_secs: elapsed,
        fps,
        mean_latency_ms,
        dmabuf,
    }
}

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .expect("usage: <model> <delegate> [secs] [warmup]");
    let delegate_path = std::env::args().nth(2).expect("delegate path required");
    let window = Duration::from_secs(
        std::env::args()
            .nth(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_WINDOW_SECS),
    );
    let warmup = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WARMUP);

    let lib = Library::new().expect("tflite library");
    let model = Model::from_file(&lib, &model_path).expect("model");

    // Describe the input so the reader knows what one invoke costs.
    {
        let probe = build_interp(&lib, &model, &delegate_path);
        let inputs = probe.inputs().expect("inputs");
        let t: &Tensor = &inputs[0];
        println!(
            "model: {model_path}\ndelegate: {delegate_path}\ninput[0]: {} bytes ({} elems)\nwindow: {:?}, warmup: {warmup}\n",
            t.byte_size(),
            t.volume().unwrap_or(0),
            window,
        );
    }

    println!("Sharing model matrix:");
    println!("  M(N): N interpreters share ONE Model (weights) — supported (Model: Send+Sync)");
    println!("  single-interpreter concurrent invoke — UNSUPPORTED (Interpreter: Send, !Sync)\n");

    let mut results = Vec::new();
    for n in SLOT_COUNTS {
        let r = run_config(&lib, &model, &delegate_path, n, warmup, window);
        println!(
            "slots={:>1}  build={:>6.2}s  dmabuf={:<5}  invokes={:>6}  window={:>5.2}s  fps={:>8.2}  lat/invoke={:>7.2}ms",
            r.slots, r.build_secs, r.dmabuf, r.invokes, r.window_secs, r.fps, r.mean_latency_ms
        );
        results.push(r);
    }

    // Speedup relative to the sequential (slots=1) baseline.
    let base = results
        .iter()
        .find(|r| r.slots == 1)
        .map(|r| r.fps)
        .unwrap_or(0.0);
    println!("\nSpeedup vs sequential (slots=1):");
    for r in &results {
        let speedup = if base > 0.0 { r.fps / base } else { 0.0 };
        println!("  slots={}: {:.2}x", r.slots, speedup);
    }
    println!(
        "\nInterpretation: speedup ~1.0x => single NPU serializes concurrent contexts (multi-slot buys nothing);\n\
         speedup > 1 => concurrent contexts overlap on the device."
    );
}
