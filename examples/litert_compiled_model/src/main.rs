// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! `LiteRT` `CompiledModel` inference example.
//!
//! Loads a model through the `LiteRT` Next path when `LiteRt*` symbols are
//! present in the shared library. On classic TensorFlow Lite hosts the example
//! reports which symbol was missing and exits, instead of panicking.
//!
//! ```sh
//! cargo run -p litert-compiled-model -- model.tflite
//!
//! # Against a specific runtime (e.g. the Android LiteRT library):
//! TFLITE_LIBRARY_PATH=./libLiteRt.so cargo run -p litert-compiled-model -- model.tflite
//! ```

use edgefirst_tflite::litert::{
    accelerators, CompiledModel, Environment, HwAccelerators, Model, Options,
};
use edgefirst_tflite::Library;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model.tflite".to_string());

    let lib = Library::new()?;
    if !lib.has_litert() {
        // `litert_missing_symbol` separates "classic TFLite" from "partial
        // LiteRT build" — two situations with very different fixes.
        let missing = lib.litert_missing_symbol().unwrap_or("<unknown>");
        eprintln!("LiteRT unavailable: {missing} could not be resolved.");
        if missing == "LiteRtCreateEnvironment" {
            eprintln!(
                "This library exports no LiteRT surface at all — it is a classic \
                 TensorFlow Lite build. The Interpreter API remains available; set \
                 TFLITE_LIBRARY_PATH to a LiteRT-capable library to use CompiledModel."
            );
        } else {
            eprintln!(
                "This library exports some LiteRT symbols but not the full set, \
                 which usually means a version skew between the deployed runtime \
                 and these bindings."
            );
        }
        std::process::exit(2);
    }

    let env = Environment::new(&lib)?;
    println!("LiteRT environment ready");

    let accels = accelerators(&env)?;
    println!("Accelerators ({}):", accels.len());
    for accel in &accels {
        println!("  {accel}");
    }

    // Request the NPU when one is registered, always permitting CPU fallback.
    let has_npu = accels
        .iter()
        .any(|a| a.hardware.contains(HwAccelerators::NPU));
    let requested = if has_npu {
        HwAccelerators::NPU | HwAccelerators::CPU
    } else {
        HwAccelerators::CPU
    };
    println!("Requesting: {requested}");

    let model = Model::from_file(&env, &path)?;
    let opts = Options::new(&lib)?.hardware_accelerators(requested)?;
    let mut compiled = CompiledModel::create(&env, &model, &opts)?;

    println!("Model: {path}");
    println!("Signatures: {}", model.num_signatures()?);
    println!("Inputs:  {}", model.num_inputs(0)?);
    println!("Outputs: {}", model.num_outputs(0)?);

    // `false` means at least one operator fell back off the requested
    // accelerators — the usual explanation for disappointing NPU numbers.
    let fully = compiled.is_fully_accelerated()?;
    println!("Fully accelerated: {fully}");
    if !fully {
        println!("  (some operators fell back; expect reduced throughput)");
    }

    // Allocate buffers the compiled model can actually reach: `create_*_buffer`
    // queries the runtime's placement requirements rather than assuming host
    // memory is usable by the accelerator.
    let mut inputs = vec![compiled.create_input_buffer(0, 0)?];
    let mut outputs = vec![compiled.create_output_buffer(0, 0)?];
    println!(
        "Input buffer: {} bytes, output buffer: {} bytes",
        inputs[0].size(),
        outputs[0].size()
    );

    // Zero-filled input is enough for a smoke run.
    let zeros = vec![0u8; inputs[0].size()];
    inputs[0].write_bytes(&zeros)?;

    compiled.run_default(&mut inputs, &mut outputs)?;

    let result = outputs[0].read_bytes()?;
    println!("Sync run completed (read {} output bytes)", result.len());

    Ok(())
}
