// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Benchmark a model with and without the `EdgeFirst` delegate.
//!
//! Builds two interpreters for the model, one as `TFLite` (and the NPU
//! delegate, if given) runs it and one with the `EdgeFirst` delegate added,
//! and reports:
//!
//! * the nodes the `EdgeFirst` delegate claimed and its kernel tier;
//! * how far the two interpreters' outputs are apart on the same input;
//! * invoke latency, measured in alternating blocks so drift (thermal, DVFS)
//!   affects both alike;
//! * per-operator time from the `TFLite` telemetry profiler, in a separate
//!   pass so profiling does not touch the latency numbers. The `EdgeFirst`
//!   delegate's partitions appear as `DELEGATE`.
//!
//! ```sh
//! # i.MX 95: EdgeFirst after Neutron
//! delegate-bench model.imx95.tflite --npu /usr/lib/libneutron_delegate.so
//! # i.MX 8M Plus: EdgeFirst before VX
//! delegate-bench model.tflite --npu /usr/lib/libvx_delegate.so --before
//! ```
//!
//! Options:
//!
//! * `--iters N` (default 200) and `--warmup N` (20; outputs are compared after at least one
//!   invoke);
//! * `--threads N`: interpreter threads, which `TFLite` and the `EdgeFirst` delegate both use
//!   (default 1);
//! * `--before`: attention blocks only, before the NPU delegate;
//! * `--scope attention|everything`: what the `EdgeFirst` delegate claims (default
//!   `everything`, or `attention` with `--before`);
//! * `--xnnpack`: add XNNPACK last to both interpreters, for the operators the other
//!   delegates leave, as stock `TFLite` applies it by default;
//! * `--image PATH`: raw interleaved 8-bit RGB for every input of that byte size (int8
//!   inputs take it shifted by 128), so the outputs compare on a real image rather than the
//!   default fill;
//! * `--lib PATH`: the `TFLite` library (else auto-discovery).
//!
//! `EDGEFIRST_KERNELS_FORCE_TIER` selects the kernel tier.

use std::collections::HashMap;
use std::time::Instant;

use edgefirst_tflite::{Delegate, Interpreter, Library, Model, Profiler, TensorType};
use edgefirst_tflite_delegate::{EdgeFirstDelegate, Options, Scope, Stats};

/// Blocks of invokes per interpreter in the latency pass.
const ROUNDS: usize = 10;

struct Args {
    model: String,
    npu: Option<String>,
    lib: Option<String>,
    iters: usize,
    warmup: usize,
    threads: i32,
    before: bool,
    /// `Scope::Attention` with `--before` unless `--scope` says otherwise.
    scope: Option<Scope>,
    /// XNNPACK last in both interpreters.
    xnnpack: bool,
    /// Raw interleaved 8-bit RGB pixels for every input of that byte size.
    image: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut parsed = Args {
        model: String::new(),
        npu: None,
        lib: None,
        iters: 200,
        warmup: 20,
        threads: 1,
        before: false,
        scope: None,
        xnnpack: false,
        image: None,
    };
    let number = |v: Option<String>, flag: &str| -> Result<usize, String> {
        v.ok_or(format!("{flag} needs a value"))?
            .parse()
            .map_err(|e| format!("{flag}: {e}"))
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--npu" => parsed.npu = Some(args.next().ok_or("--npu needs a path")?),
            "--lib" => parsed.lib = Some(args.next().ok_or("--lib needs a path")?),
            "--iters" => parsed.iters = number(args.next(), "--iters")?.max(ROUNDS),
            "--warmup" => parsed.warmup = number(args.next(), "--warmup")?,
            "--threads" => {
                parsed.threads = i32::try_from(number(args.next(), "--threads")?)
                    .map_err(|e| format!("--threads: {e}"))?;
            }
            "--before" => parsed.before = true,
            "--scope" => {
                parsed.scope = Some(match args.next().as_deref() {
                    Some("attention") => Scope::Attention,
                    Some("everything") => Scope::Everything,
                    _ => return Err("--scope needs attention or everything".into()),
                });
            }
            "--xnnpack" => parsed.xnnpack = true,
            "--image" => parsed.image = Some(args.next().ok_or("--image needs a path")?),
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}")),
            _ if parsed.model.is_empty() => parsed.model = arg,
            _ => return Err(format!("unexpected argument {arg}")),
        }
    }
    if parsed.model.is_empty() {
        return Err("usage: delegate-bench <model> [--npu PATH] [--before] \
                    [--scope attention|everything] [--xnnpack] [--iters N] [--warmup N] \
                    [--threads N] [--image PATH] [--lib PATH]"
            .into());
    }
    Ok(parsed)
}

/// Build an interpreter: the NPU delegate if any, plus the `EdgeFirst`
/// delegate before or after it when `edgefirst` is set, and XNNPACK last with
/// `--xnnpack`.
fn build<'lib>(
    lib: &'lib Library,
    model: &Model<'lib>,
    args: &Args,
    edgefirst: bool,
    profiler: Option<&Profiler>,
) -> Result<(Interpreter<'lib>, Option<Stats>), Box<dyn std::error::Error>> {
    let mut delegates = Vec::new();
    if let Some(npu) = &args.npu {
        delegates.push(Delegate::load(npu)?);
    }
    let mut stats = None;
    if edgefirst {
        let delegate =
            EdgeFirstDelegate::new()?.with_options(Options::default().scope(scope(args)));
        stats = Some(delegate.stats());
        let delegate = delegate.into_delegate(lib)?;
        if args.before {
            delegates.insert(0, delegate);
        } else {
            delegates.push(delegate);
        }
    }
    if args.xnnpack {
        delegates.push(Delegate::xnnpack(lib, args.threads)?);
    }
    let mut builder = Interpreter::builder(lib)?.num_threads(args.threads);
    if let Some(profiler) = profiler {
        builder = builder.profiler(profiler)?;
    }
    for d in delegates {
        builder = builder.delegate(d);
    }
    Ok((builder.build(model)?, stats))
}

fn scope(args: &Args) -> Scope {
    args.scope.unwrap_or(if args.before {
        Scope::Attention
    } else {
        Scope::Everything
    })
}

/// Fill every input with the same deterministic bytes (xorshift32), then copy `rgb` into each
/// input of its byte size (shifted by 128 for int8 inputs).
fn fill_inputs(
    interp: &mut Interpreter<'_>,
    rgb: Option<&[u8]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut s = 0x9e37_79b9u32;
    for mut input in interp.inputs_mut()? {
        let int8 = input.tensor_type() == TensorType::Int8;
        let bytes = input.as_bytes_mut()?;
        for b in bytes.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *b = s.to_le_bytes()[0];
        }
        if let Some(rgb) = rgb.filter(|rgb| rgb.len() == bytes.len()) {
            for (d, &s) in bytes.iter_mut().zip(rgb) {
                *d = if int8 { s ^ 0x80 } else { s };
            }
        }
    }
    Ok(())
}

/// Largest elementwise difference between two outputs, in LSB or float units. A float NaN
/// against a number counts as an infinite difference.
fn max_diff(ty: TensorType, a: &[u8], b: &[u8]) -> f64 {
    let pairs: Box<dyn Iterator<Item = f64>> = match ty {
        TensorType::Int8 => Box::new(
            a.iter()
                .zip(b)
                .map(|(&x, &y)| f64::from(x.cast_signed().abs_diff(y.cast_signed()))),
        ),
        TensorType::Int16 => Box::new(
            a.as_chunks::<2>()
                .0
                .iter()
                .zip(b.as_chunks::<2>().0)
                .map(|(&x, &y)| f64::from(i16::from_le_bytes(x).abs_diff(i16::from_le_bytes(y)))),
        ),
        TensorType::Float32 => Box::new(a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0).map(
            |(&x, &y)| {
                let (x, y) = (f32::from_le_bytes(x), f32::from_le_bytes(y));
                match (x.is_nan(), y.is_nan()) {
                    (false, false) => f64::from((x - y).abs()),
                    (true, true) => 0.0,
                    _ => f64::INFINITY,
                }
            },
        )),
        _ => Box::new(a.iter().zip(b).map(|(&x, &y)| f64::from(x.abs_diff(y)))),
    };
    pairs.fold(0.0, f64::max)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let lib = match &args.lib {
        Some(path) => Library::from_path(path)?,
        None => Library::new()?,
    };
    let model = Model::from_file(&lib, &args.model)?;
    let names = ["stock", "edgefirst"];

    // Latency pass, no profiler.
    let (mut stock, _) = build(&lib, &model, &args, false, None)?;
    let (mut ef, stats) = build(&lib, &model, &args, true, None)?;
    let stats = stats.expect("EdgeFirst interpreter has stats");
    println!("model:      {}", args.model);
    println!(
        "stack:      {}{}, EdgeFirst scope {}{}",
        args.npu.as_deref().unwrap_or("CPU only"),
        if args.before {
            " (EdgeFirst before it)"
        } else {
            " (EdgeFirst after it)"
        },
        if scope(&args) == Scope::Attention {
            "attention"
        } else {
            "everything"
        },
        if args.xnnpack { ", XNNPACK last" } else { "" }
    );
    println!(
        "edgefirst:  {} nodes claimed in {} partitions, {} fused attention, {} fused DFL, tier {}",
        stats.claimed_nodes(),
        stats.partitions(),
        stats.fused_attention(),
        stats.fused_softmax_dot(),
        EdgeFirstDelegate::new()?.tier().as_str()
    );

    let rgb = args.image.as_deref().map(std::fs::read).transpose()?;
    let interps = [&mut stock, &mut ef];
    for interp in interps {
        fill_inputs(interp, rgb.as_deref())?;
        // At least one invoke, so the outputs compared below are computed.
        for _ in 0..args.warmup.max(1) {
            interp.invoke()?;
        }
    }
    let outputs: Vec<Vec<(TensorType, Vec<u8>)>> = [&stock, &ef]
        .iter()
        .map(|i| {
            i.outputs().map(|ts| {
                ts.iter()
                    .map(|t| (t.tensor_type(), t.as_bytes().unwrap_or_default().to_vec()))
                    .collect()
            })
        })
        .collect::<Result<_, _>>()?;
    for (k, ((ty, a), (_, b))) in outputs[0].iter().zip(&outputs[1]).enumerate() {
        let diff = max_diff(*ty, a, b);
        let differing = a.iter().zip(b).filter(|(x, y)| x != y).count();
        println!(
            "output {k}:   {ty:?}, max diff {diff}, {differing} of {} bytes differ",
            a.len()
        );
    }

    let mut latencies = [Vec::new(), Vec::new()];
    let per_round = args.iters / ROUNDS;
    for _ in 0..ROUNDS {
        for (interp, lat) in [&mut stock, &mut ef].into_iter().zip(&mut latencies) {
            for _ in 0..per_round {
                let t = Instant::now();
                interp.invoke()?;
                lat.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    println!("\nlatency (ms, {} invokes each):", per_round * ROUNDS);
    println!(
        "  {:<10} {:>8} {:>8} {:>8} {:>8}",
        "", "median", "p90", "mean", "min"
    );
    let mut medians = [0.0; 2];
    for ((name, lat), median) in names.iter().zip(&mut latencies).zip(&mut medians) {
        lat.sort_by(f64::total_cmp);
        #[allow(clippy::cast_precision_loss)]
        let mean = lat.iter().sum::<f64>() / lat.len() as f64;
        *median = percentile(lat, 0.5);
        println!(
            "  {name:<10} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            *median,
            percentile(lat, 0.9),
            mean,
            lat[0]
        );
    }
    println!(
        "  saved {:.2} ms per invoke at the median ({:.1}%)",
        medians[0] - medians[1],
        100.0 * (medians[0] - medians[1]) / medians[0]
    );
    drop((stock, ef));

    // Profile pass.
    for (name, edgefirst) in names.iter().zip([false, true]) {
        let profiler = Profiler::new();
        let (mut interp, _) = build(&lib, &model, &args, edgefirst, Some(&profiler))?;
        fill_inputs(&mut interp, rgb.as_deref())?;
        for _ in 0..args.warmup {
            interp.invoke()?;
        }
        profiler.clear();
        let runs = args.iters.min(100);
        for _ in 0..runs {
            interp.invoke()?;
        }
        let mut per_op: HashMap<String, (u64, usize)> = HashMap::new();
        for e in profiler.drain_events() {
            let entry = per_op.entry(e.op_name).or_default();
            entry.0 += e.duration_us;
            entry.1 += 1;
        }
        let mut rows: Vec<_> = per_op.into_iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1 .0));
        #[allow(clippy::cast_precision_loss)]
        let total: f64 = rows.iter().map(|r| r.1 .0 as f64).sum::<f64>() / runs as f64 / 1e3;
        println!("\nper-operator time, {name} (ms per invoke, {runs} invokes, total {total:.2}):");
        for (op, (us, count)) in rows {
            #[allow(clippy::cast_precision_loss)]
            let ms = us as f64 / runs as f64 / 1e3;
            println!("  {op:<24} {ms:>8.3}  x{}", count / runs);
        }
    }
    Ok(())
}
