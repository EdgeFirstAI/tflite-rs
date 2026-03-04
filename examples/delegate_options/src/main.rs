// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Delegate options example.
//!
//! Demonstrates how to configure a delegate with key-value options using the
//! [`DelegateOptions`] builder, probe for optional feature support, and
//! transfer delegate ownership into the interpreter.
//!
//! ```sh
//! cargo run -p delegate-options -- model.tflite libvx_delegate.so
//! ```

use edgefirst_tflite::{Delegate, DelegateOptions, Interpreter, Library, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model.tflite".to_string());
    let delegate_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "libvx_delegate.so".to_string());

    // Load the TFLite library and model.
    let lib = Library::new()?;
    let model = Model::from_file(&lib, &model_path)?;

    // Build delegate options with key-value pairs.
    let options = DelegateOptions::new()
        .option("cache_file_path", "/tmp/vx_cache")
        .option("device_id", "0");

    println!("Options: {options:?}");

    // Compare: load with default options vs. load with custom options.
    // Delegate::load() uses DelegateOptions::default() internally.
    //   let _default = Delegate::load(&delegate_path)?;
    let delegate = Delegate::load_with_options(&delegate_path, &options)?;

    // Probe for optional feature support.
    let has_dmabuf = delegate.has_dmabuf();
    let has_camera = delegate.has_camera_adaptor();
    println!("DMA-BUF supported:         {has_dmabuf}");
    println!("CameraAdaptor supported:   {has_camera}");

    // Transfer delegate ownership into the interpreter via the builder.
    let mut interpreter = Interpreter::builder(&lib)?
        .delegate(delegate)
        .num_threads(4)
        .build(&model)?;

    println!("Model: {model_path}");
    println!("Inputs:  {}", interpreter.input_count());
    println!("Outputs: {}", interpreter.output_count());

    // Access the delegate after ownership transfer.
    let delegate_ref = interpreter.delegate(0).expect("delegate not found");
    println!("Delegate: {delegate_ref:?}");

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

    println!("Inference complete.");

    Ok(())
}
