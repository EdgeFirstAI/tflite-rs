// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! # Async Pipeline Example
//!
//! Demonstrates pipelined inference with a circular buffer of interpreter
//! slots. Each slot is a separate [`Interpreter`] created from the same
//! model. While one thread runs inference on slot N, the main thread fills
//! inputs for slot N+1 and reads outputs from slot N-1, achieving CPU/NPU
//! overlap.
//!
//! ## Pipeline Stages
//!
//! ```text
//!   ┌──────────┐   ┌──────────┐   ┌──────────┐
//!   │ Fill (CPU)│──▶│Infer(thr)│──▶│Read (CPU)│
//!   └──────────┘   └──────────┘   └──────────┘
//!        slot[i]      slot[i-1]      slot[i-2]
//! ```
//!
//! ## Usage
//!
//! ```text
//! cargo run -p async-pipeline -- model.tflite [iterations] [depth]
//! ```
//!
//! - `iterations`: total inferences to run (default: 100)
//! - `depth`: ring buffer depth / number of interpreter slots (default: 2)
//!
//! An optional `--delegate <path>` flag loads a hardware delegate (e.g.,
//! VxDelegate for NPU acceleration).

use edgefirst_tflite::{Delegate, Interpreter, Library, Model};
use std::{env, process, thread, time::Instant};

/// Simulate CPU preprocessing by writing a pattern into all input tensors.
fn fill_inputs(interp: &mut Interpreter<'_>, frame: usize) {
    let val = (frame % 256) as f32;
    let mut inputs = interp.inputs_mut().expect("failed to get input tensors");
    for tensor in &mut inputs {
        let slice = tensor.as_mut_slice::<f32>().expect("failed to map input");
        for (j, elem) in slice.iter_mut().enumerate() {
            *elem = val + j as f32;
        }
    }
}

/// Read outputs and compute a simple checksum for verification.
fn read_outputs(interp: &Interpreter<'_>) -> f32 {
    let outputs = interp.outputs().expect("failed to get output tensors");
    let mut checksum: f32 = 0.0;
    for tensor in &outputs {
        let slice = tensor.as_slice::<f32>().expect("failed to map output");
        for &v in slice {
            checksum += v;
        }
    }
    checksum
}

/// An in-flight inference request wrapping a scoped thread join handle.
///
/// The interpreter is moved into the thread for the duration of inference
/// and returned when the thread joins, preventing any concurrent access.
struct InferRequest<'lib, 'scope> {
    handle: Option<thread::ScopedJoinHandle<'scope, Interpreter<'lib>>>,
}

impl<'lib, 'scope> InferRequest<'lib, 'scope> {
    /// Submit inference asynchronously — returns immediately.
    fn submit(scope: &'scope thread::Scope<'scope, 'lib>, mut interp: Interpreter<'lib>) -> Self {
        let handle = scope.spawn(move || {
            interp.invoke().expect("invoke failed");
            interp
        });
        Self {
            handle: Some(handle),
        }
    }

    /// Block until inference completes and return the interpreter.
    fn wait(mut self) -> Interpreter<'lib> {
        self.handle
            .take()
            .expect("already waited")
            .join()
            .expect("inference thread panicked")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model.tflite> [iterations] [depth] [--delegate <path>]",
            args[0]
        );
        process::exit(1);
    }

    let model_path = &args[1];
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let depth: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);

    // Parse optional --delegate flag.
    let delegate_path = args
        .windows(2)
        .find(|w| w[0] == "--delegate")
        .map(|w| w[1].clone());

    println!("Model:      {model_path}");
    println!("Iterations: {iterations}");
    println!("Depth:      {depth}");

    // Load the library and model (shared across all interpreter slots).
    let lib = Library::new()?;
    let model = Model::from_file(&lib, model_path)?;

    // Create `depth` interpreter slots, each with its own delegate.
    let mut slots: Vec<Interpreter<'_>> = Vec::with_capacity(depth);
    for _ in 0..depth {
        let mut builder = Interpreter::builder(&lib)?.num_threads(1);
        if let Some(ref path) = delegate_path {
            builder = builder.delegate(Delegate::load(path)?);
        }
        slots.push(builder.build(&model)?);
    }

    println!(
        "Created {} interpreter slot(s): {} inputs, {} outputs",
        slots.len(),
        slots[0].input_count(),
        slots[0].output_count()
    );

    // Warm-up each slot.
    for slot in &mut slots {
        slot.invoke()?;
    }
    println!("Warm-up complete\n");

    // ── Baseline: synchronous ───────────────────────────────────────
    println!("=== Synchronous baseline (fill → invoke → read) x {iterations} ===");
    let start = Instant::now();
    for frame in 0..iterations {
        let slot = &mut slots[0];
        fill_inputs(slot, frame);
        slot.invoke()?;
        let _ = read_outputs(slot);
    }
    let sync_elapsed = start.elapsed();
    println!(
        "  Total: {:.2}ms, Avg: {:.2}ms/iter, Throughput: {:.1} fps",
        sync_elapsed.as_secs_f64() * 1000.0,
        sync_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        iterations as f64 / sync_elapsed.as_secs_f64(),
    );

    // ── Pipelined: circular buffer ──────────────────────────────────
    println!("\n=== Pipelined async (depth={depth}) x {iterations} ===");
    let start = Instant::now();

    // Wrap interpreters in Option so we can take/return them.
    let mut slot_interps: Vec<Option<Interpreter<'_>>> = slots.into_iter().map(Some).collect();

    thread::scope(|scope| {
        // In-flight request ring: one Option<InferRequest> per slot.
        let mut inflight: Vec<Option<InferRequest<'_, '_>>> = (0..depth).map(|_| None).collect();

        for frame in 0..iterations + depth {
            let slot_idx = frame % depth;

            // Wait for the previous request on this slot (if any).
            if let Some(req) = inflight[slot_idx].take() {
                let interp = req.wait();
                let _ = read_outputs(&interp);
                // Return the interpreter to its slot.
                slot_interps[slot_idx] = Some(interp);
            }

            // If there are still frames to submit, fill and submit.
            if frame < iterations {
                let mut interp = slot_interps[slot_idx]
                    .take()
                    .expect("slot should have an interpreter");
                fill_inputs(&mut interp, frame);
                inflight[slot_idx] = Some(InferRequest::submit(scope, interp));
            }
        }
    });

    let pipeline_elapsed = start.elapsed();
    println!(
        "  Total: {:.2}ms, Avg: {:.2}ms/iter, Throughput: {:.1} fps",
        pipeline_elapsed.as_secs_f64() * 1000.0,
        pipeline_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        iterations as f64 / pipeline_elapsed.as_secs_f64(),
    );

    // ── Summary ─────────────────────────────────────────────────────
    let speedup = sync_elapsed.as_secs_f64() / pipeline_elapsed.as_secs_f64();
    println!(
        "\nSpeedup: {:.2}x (sync {:.2}ms vs pipeline {:.2}ms)",
        speedup,
        sync_elapsed.as_secs_f64() * 1000.0,
        pipeline_elapsed.as_secs_f64() * 1000.0,
    );

    Ok(())
}
