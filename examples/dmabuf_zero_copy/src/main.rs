// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! DMA-BUF zero-copy inference example.
//!
//! Demonstrates loading a delegate, querying DMA-BUF tensor metadata via the
//! HAL Delegate API, and running inference with zero-copy cache synchronization.
//!
//! ```sh
//! cargo run -p dmabuf-zero-copy -- model.tflite libvx_delegate.so
//! ```

use edgefirst_tflite::{Delegate, Interpreter, Library, Model};

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

    // Load the delegate for NPU acceleration.
    let delegate = Delegate::load(&delegate_path)?;

    // Check for DMA-BUF support.
    let has_dmabuf = delegate.has_dmabuf();
    println!("DMA-BUF supported: {has_dmabuf}");

    if !has_dmabuf {
        eprintln!("This delegate does not support DMA-BUF zero-copy.");
        std::process::exit(1);
    }

    // Build the interpreter with the delegate.
    let mut interpreter = Interpreter::builder(&lib)?
        .delegate(delegate)
        .build(&model)?;

    // Access the delegate's DMA-BUF interface.
    let delegate_ref = interpreter.delegate(0).expect("delegate not found");
    let dmabuf = delegate_ref.dmabuf().expect("DMA-BUF not available");

    println!("DMA-BUF is_supported: {}", dmabuf.is_supported());

    // Query tensor DMA-BUF metadata via the HAL Delegate API.
    match dmabuf.tensor_info(0) {
        Ok(info) => {
            println!("Input tensor[0] DMA-BUF info:");
            println!("  fd:     {}", info.fd);
            println!("  size:   {} bytes", info.size);
            println!("  offset: {}", info.offset);
            println!("  shape:  {:?}", info.shape);
            println!("  dtype:  {}", info.dtype);
        }
        Err(e) => {
            println!("tensor_info not available: {e}");
        }
    }

    // In a real application, you would get a DMA-BUF fd from V4L2, DRM, etc.
    // Here we show the HAL API workflow:
    //
    //   // Query input tensor DMA-BUF info
    //   let info = dmabuf.tensor_info(0)?;
    //   // ... write data into the DMA-BUF via info.fd ...
    //
    //   // For each frame:
    //   dmabuf.sync_for_device(0)?;    // flush CPU caches for NPU
    //   interpreter.invoke()?;
    //   dmabuf.sync_for_cpu(0)?;       // invalidate caches for CPU
    //   // ... read output tensors ...

    // Run inference.
    interpreter.invoke()?;
    println!("Inference complete.");

    // Print output tensor info.
    for (i, tensor) in interpreter.outputs()?.iter().enumerate() {
        println!("  output[{i}]: {tensor}");
    }

    Ok(())
}
