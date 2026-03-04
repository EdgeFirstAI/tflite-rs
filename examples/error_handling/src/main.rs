// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Error handling example for TFLite inference.
//!
//! Demonstrates error classification, graceful fallback from delegate to
//! CPU-only inference, and `std::error::Error::source()` chain traversal.
//!
//! ```sh
//! cargo run -p error-handling -- model.tflite [delegate.so]
//! ```

use edgefirst_tflite::{Delegate, Error, Interpreter, Library, Model};

/// Classify an [`Error`] using the inspection helpers and print diagnostics.
fn classify_error(err: &Error) {
    println!("  Error: {err}");

    if err.is_library_error() {
        println!("  Classification: library/loading error");
    } else if err.is_delegate_error() {
        println!("  Classification: delegate error");
    } else if err.is_null_pointer() {
        println!("  Classification: null pointer from C API");
    }

    if let Some(code) = err.status_code() {
        println!("  Status code: {code}");
    }

    // Walk the std::error::Error::source() chain.
    let mut source = std::error::Error::source(err);
    while let Some(cause) = source {
        println!("  Caused by: {cause}");
        source = cause.source();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model.tflite".to_string());
    let delegate_path = std::env::args().nth(2);

    // 1. Try to load the TFLite shared library, handling library errors
    //    gracefully.
    let lib = match Library::new() {
        Ok(lib) => lib,
        Err(err) => {
            eprintln!("Failed to load TFLite library.");
            classify_error(&err);
            if err.is_library_error() {
                eprintln!("Hint: ensure libtensorflow-lite is installed.");
            }
            return Err(err.into());
        }
    };

    // 2. Load the model.
    let model = Model::from_file(&lib, &model_path)?;
    println!("Model loaded: {model_path}");

    // 3. Optionally try to load a delegate, falling back to CPU on failure.
    let delegate = if let Some(ref path) = delegate_path {
        match Delegate::load(path) {
            Ok(d) => {
                println!("Delegate loaded: {path}");
                Some(d)
            }
            Err(err) => {
                eprintln!("Delegate failed to load, falling back to CPU.");
                classify_error(&err);
                None // fall back to CPU-only inference
            }
        }
    } else {
        println!("No delegate path provided, using CPU-only inference.");
        None
    };

    // 4. Build the interpreter, attaching the delegate if available.
    let mut builder = Interpreter::builder(&lib)?.num_threads(4);
    if let Some(d) = delegate {
        builder = builder.delegate(d);
    }
    let mut interpreter = builder.build(&model)?;

    println!("Interpreter ready.");
    println!("  Inputs:  {}", interpreter.input_count());
    println!("  Outputs: {}", interpreter.output_count());

    // 5. Run inference (inputs are zero-filled by default).
    interpreter.invoke()?;
    println!("Inference complete.");

    // Print output tensor info.
    for (i, tensor) in interpreter.outputs()?.iter().enumerate() {
        println!("  output[{i}]: {tensor}");
    }

    Ok(())
}
