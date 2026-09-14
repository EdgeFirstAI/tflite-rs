// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Multi-context throughput benchmark.
//!
//! Measures inference throughput for a delegate under two sharing models,
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
//!
//! **VX prerequisite:** `N > 1` creates one delegate instance per slot, which
//! needs a multi-context-safe VxDelegate (EdgeFirst `tflite-vx-delegate-imx`
//! with the EDGEAI-1435 instance-resolver fix). Older VxDelegate builds
//! double-free or corrupt under concurrent contexts (see `ARCHITECTURE.md`), so
//! this benchmark aborts instead of reporting results on them. Neutron and CPU
//! delegates have no such prerequisite.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;
use std::time::{Duration, Instant};

use edgefirst_tflite::{Delegate, Interpreter, Library, Model, Tensor};

const DEFAULT_WINDOW_SECS: u64 = 4;
const DEFAULT_WARMUP: usize = 5;
const SLOT_COUNTS: [usize; 3] = [1, 2, 4];

/// Fill input 0 with a deterministic byte pattern spanning the whole
/// allocation. Writing raw bytes across `byte_size()` covers every fixed- or
/// packed-width element type (`Int4`, `Complex128`, …) without an element-width
/// table — the content is irrelevant to throughput; only that the buffer is
/// initialized and identical every invoke.
fn fill_input(interp: &mut Interpreter) {
    let mut inputs = interp.inputs_mut().expect("inputs_mut");
    let t = &mut inputs[0];
    let data: Vec<u8> = (0..t.byte_size())
        .map(|i| u8::try_from(i * 7 % 251).expect("pattern byte"))
        .collect();
    t.copy_from_bytes(&data).expect("fill input");
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
///
/// Thread count is fixed at 1 so that raising the slot count changes only the
/// number of concurrent contexts, not the amount of CPU parallelism — the
/// reported speedup then reflects multi-context overlap alone, not extra
/// per-context worker threads.
fn build_interp<'lib>(
    lib: &'lib Library,
    model: &Model<'lib>,
    delegate_path: &str,
) -> Interpreter<'lib> {
    let d = Delegate::load(delegate_path).expect("delegate load");
    Interpreter::builder(lib)
        .expect("builder")
        .num_threads(1)
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

    // `ready` synchronizes warmup completion; `go` releases the workers only
    // after the clock starts, so no counted invoke precedes `t0`; `stop` ends
    // the window.
    let ready = Barrier::new(n + 1);
    let go = AtomicBool::new(false);
    let stop = AtomicBool::new(false);

    let (counts, elapsed) = std::thread::scope(|s| {
        let ready = &ready;
        let go = &go;
        let stop = &stop;
        let handles: Vec<_> = interps
            .iter_mut()
            .map(|it| {
                s.spawn(move || -> u64 {
                    ready.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    let mut count = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        it.invoke().expect("invoke");
                        count += 1;
                    }
                    count
                })
            })
            .collect();

        ready.wait();
        // Start the clock, THEN release the workers: every counted invoke
        // falls inside the measured window (`elapsed` also covers the in-flight
        // invokes finishing after `stop`, so counts and window stay consistent).
        let t0 = Instant::now();
        go.store(true, Ordering::Release);
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
