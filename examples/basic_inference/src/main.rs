// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Basic TFLite inference example.
//!
//! Loads a model, populates input tensors, runs inference, and reads outputs.
//!
//! ```sh
//! cargo run -p basic-inference -- model.tflite
//! ```

use edgefirst_tflite::{Interpreter, Library, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model.tflite".to_string());

    // Discover and load the TFLite shared library.
    let lib = Library::new()?;

    // Load the model from a file.
    let model = Model::from_file(&lib, &path)?;

    // Build an interpreter with 4 threads.
    let mut interpreter = Interpreter::builder(&lib)?.num_threads(4).build(&model)?;

    println!("Model: {path}");
    println!("Inputs:  {}", interpreter.input_count());
    println!("Outputs: {}", interpreter.output_count());

    // Print input tensor info.
    for (i, tensor) in interpreter.inputs()?.iter().enumerate() {
        println!("  input[{i}]: {tensor}");
    }

    // Run inference (inputs are zero-filled by default).
    interpreter.invoke()?;

    // Print output tensor info.
    for (i, tensor) in interpreter.outputs()?.iter().enumerate() {
        println!("  output[{i}]: {tensor}");
    }

    Ok(())
}
