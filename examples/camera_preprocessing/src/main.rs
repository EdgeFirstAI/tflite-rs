// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Camera preprocessing example using NPU-accelerated format conversion.
//!
//! Demonstrates the full `CameraAdaptor` API: probing format support, querying
//! channel counts, converting FourCC codes, and configuring NPU-side format
//! conversion before running inference.
//!
//! ```sh
//! cargo run -p camera-preprocessing -- model.tflite libvx_delegate.so
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

    // Load the VxDelegate for NPU acceleration.
    let delegate = Delegate::load(&delegate_path)?;

    // Check for CameraAdaptor support.
    let has_camera_adaptor = delegate.has_camera_adaptor();
    println!("CameraAdaptor supported: {has_camera_adaptor}");

    if !has_camera_adaptor {
        eprintln!("This delegate does not support CameraAdaptor preprocessing.");
        std::process::exit(1);
    }

    let adaptor = delegate
        .camera_adaptor()
        .expect("CameraAdaptor not available");

    // -----------------------------------------------------------------------
    // Probe: query format support for common camera formats.
    // -----------------------------------------------------------------------
    let formats = ["rgba", "rgb", "nv12", "bgr", "gray"];
    println!("\nFormat support:");
    for fmt in &formats {
        println!("  {fmt:>4}: supported={}", adaptor.is_supported(fmt));
    }

    // -----------------------------------------------------------------------
    // Channel counts: input (camera-side) and output (model-side) channels.
    // -----------------------------------------------------------------------
    println!("\nChannel counts:");
    for fmt in &formats {
        println!(
            "  {fmt:>4}: input_channels={}, output_channels={}",
            adaptor.input_channels(fmt),
            adaptor.output_channels(fmt),
        );
    }

    // -----------------------------------------------------------------------
    // FourCC conversions: format string <-> V4L2 FourCC code.
    // -----------------------------------------------------------------------
    println!("\nFourCC conversions:");
    for fmt in &formats {
        let fourcc = adaptor.fourcc(fmt).unwrap_or_else(|| "N/A".to_string());
        println!("  {fmt:>4} -> fourcc: {fourcc}");
    }

    // Reverse lookup: FourCC code back to format string.
    let fourcc_codes = ["RGBP", "RGB3", "NV12", "BGR3"];
    println!("\nReverse FourCC lookup:");
    for code in &fourcc_codes {
        let name = adaptor
            .from_fourcc(code)
            .unwrap_or_else(|| "N/A".to_string());
        println!("  {code} -> format: {name}");
    }

    // -----------------------------------------------------------------------
    // Configure: set_format - simple RGBA -> RGB conversion on tensor 0.
    // -----------------------------------------------------------------------
    println!("\nConfiguring RGBA -> RGB conversion on tensor 0...");
    adaptor.set_format(0, "rgba")?;
    println!("  set_format(0, \"rgba\") OK");

    if let Some(current) = adaptor.format(0) {
        println!("  current format for tensor 0: {current}");
    }

    // -----------------------------------------------------------------------
    // Configure: set_format_ex - resize to 640x480 with letterboxing.
    // -----------------------------------------------------------------------
    println!("\nConfiguring with resize and letterbox options...");
    adaptor.set_format_ex(
        0,      // tensor index
        "rgba", // camera format
        640,    // width
        480,    // height
        true,   // letterbox enabled
        0,      // letterbox fill color (black)
    )?;
    println!("  set_format_ex(0, \"rgba\", 640, 480, letterbox=true, color=0) OK");

    // -----------------------------------------------------------------------
    // Configure: set_formats - explicit camera/model format pair.
    // -----------------------------------------------------------------------
    println!("\nConfiguring explicit camera/model format pair...");
    adaptor.set_formats(0, "nv12", "rgb")?;
    println!("  set_formats(0, camera=\"nv12\", model=\"rgb\") OK");

    // -----------------------------------------------------------------------
    // Build: create interpreter with the delegate and run inference.
    // -----------------------------------------------------------------------
    println!("\nBuilding interpreter with delegate...");
    let mut interpreter = Interpreter::builder(&lib)?
        .delegate(delegate)
        .build(&model)?;

    println!("Model: {model_path}");
    println!("Inputs:  {}", interpreter.input_count());
    println!("Outputs: {}", interpreter.output_count());

    // Print input tensor info.
    for (i, tensor) in interpreter.inputs()?.iter().enumerate() {
        println!("  input[{i}]: {tensor}");
    }

    // Run inference.
    interpreter.invoke()?;
    println!("Inference complete.");

    // Print output tensor info.
    for (i, tensor) in interpreter.outputs()?.iter().enumerate() {
        println!("  output[{i}]: {tensor}");
    }

    Ok(())
}
