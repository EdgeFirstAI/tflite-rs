// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Quantized model inference example.
//!
//! Loads a quantized `TFLite` model, inspects quantization parameters on input
//! and output tensors, applies quantize/dequantize arithmetic, and runs
//! inference with sample data.
//!
//! ```sh
//! cargo run -p quantized-inference -- model.tflite
//! ```

use edgefirst_tflite::{Interpreter, Library, Model, QuantizationParams, TensorType};

/// Quantize a floating-point value to an unsigned 8-bit integer.
///
/// Uses the affine quantization formula:
///   `quantized_value = round(real_value / scale) + zero_point`
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn quantize_u8(value: f32, params: QuantizationParams) -> u8 {
    let quantized = (value / params.scale) + params.zero_point as f32;
    quantized.round().clamp(0.0, 255.0) as u8
}

/// Dequantize an unsigned 8-bit integer back to a floating-point value.
///
/// Uses the affine dequantization formula:
///   `real_value = scale * (quantized_value - zero_point)`
#[allow(clippy::cast_precision_loss)]
fn dequantize_u8(value: u8, params: QuantizationParams) -> f32 {
    params.scale * (f32::from(value) - params.zero_point as f32)
}

/// Quantize a floating-point value to a signed 8-bit integer.
///
/// Uses the affine quantization formula:
///   `quantized_value = round(real_value / scale) + zero_point`
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn quantize_i8(value: f32, params: QuantizationParams) -> i8 {
    let quantized = (value / params.scale) + params.zero_point as f32;
    quantized.round().clamp(-128.0, 127.0) as i8
}

/// Dequantize a signed 8-bit integer back to a floating-point value.
///
/// Uses the affine dequantization formula:
///   `real_value = scale * (quantized_value - zero_point)`
#[allow(clippy::cast_precision_loss)]
fn dequantize_i8(value: i8, params: QuantizationParams) -> f32 {
    params.scale * (f32::from(value) - params.zero_point as f32)
}

/// Populate input tensors with sample quantized data.
fn fill_inputs(interpreter: &mut Interpreter<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let mut inputs = interpreter.inputs_mut()?;
    for input in &mut inputs {
        let params = input.quantization_params();
        let volume = input.volume()?;

        match input.tensor_type() {
            TensorType::UInt8 => {
                // Quantize a sample value of 1.0 for each element.
                let sample = quantize_u8(1.0, params);
                let data = vec![sample; volume];
                input.copy_from_slice(&data)?;
                println!(
                    "  Filled {} with {volume} uint8 values (1.0 -> quantized {sample})",
                    input.name(),
                );
            }
            TensorType::Int8 => {
                // Quantize a sample value of 1.0 for each element.
                let sample = quantize_i8(1.0, params);
                let data = vec![sample; volume];
                input.copy_from_slice(&data)?;
                println!(
                    "  Filled {} with {volume} int8 values (1.0 -> quantized {sample})",
                    input.name(),
                );
            }
            TensorType::Float32 => {
                // Non-quantized input: fill with 1.0 directly.
                let data = vec![1.0_f32; volume];
                input.copy_from_slice(&data)?;
                println!(
                    "  Filled {} with {volume} float32 values (1.0)",
                    input.name()
                );
            }
            other => {
                println!(
                    "  Skipping {} with unsupported type {other:?}",
                    input.name()
                );
            }
        }
    }
    Ok(())
}

/// Read and print dequantized output tensors.
fn print_outputs(interpreter: &Interpreter<'_>) -> Result<(), Box<dyn std::error::Error>> {
    for (i, tensor) in interpreter.outputs()?.iter().enumerate() {
        let params = tensor.quantization_params();

        match tensor.tensor_type() {
            TensorType::UInt8 => {
                let data = tensor.as_slice::<u8>()?;
                let dequantized: Vec<f32> =
                    data.iter().map(|&v| dequantize_u8(v, params)).collect();
                let preview: Vec<_> = dequantized.iter().take(10).collect();
                println!("  output[{i}]: {} uint8 values", data.len());
                println!(
                    "    raw (first 10):         {:?}",
                    &data[..data.len().min(10)]
                );
                println!("    dequantized (first 10): {preview:.4?}");
            }
            TensorType::Int8 => {
                let data = tensor.as_slice::<i8>()?;
                let dequantized: Vec<f32> =
                    data.iter().map(|&v| dequantize_i8(v, params)).collect();
                let preview: Vec<_> = dequantized.iter().take(10).collect();
                println!("  output[{i}]: {} int8 values", data.len());
                println!(
                    "    raw (first 10):         {:?}",
                    &data[..data.len().min(10)]
                );
                println!("    dequantized (first 10): {preview:.4?}");
            }
            TensorType::Float32 => {
                let data = tensor.as_slice::<f32>()?;
                let preview: Vec<_> = data.iter().take(10).collect();
                println!("  output[{i}]: {} float32 values", data.len());
                println!("    values (first 10): {preview:.6?}");
            }
            other => {
                println!("  output[{i}]: unsupported type {other:?}");
            }
        }
    }
    Ok(())
}

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

    // Print input tensor info including quantization parameters.
    for (i, tensor) in interpreter.inputs()?.iter().enumerate() {
        let params = tensor.quantization_params();
        println!(
            "  input[{i}]: {tensor} (scale={}, zero_point={})",
            params.scale, params.zero_point,
        );
    }

    // Print output tensor info including quantization parameters.
    for (i, tensor) in interpreter.outputs()?.iter().enumerate() {
        let params = tensor.quantization_params();
        println!(
            "  output[{i}]: {tensor} (scale={}, zero_point={})",
            params.scale, params.zero_point,
        );
    }

    // Populate input tensors with sample quantized data.
    fill_inputs(&mut interpreter)?;

    // Run inference.
    interpreter.invoke()?;
    println!("Inference complete.");

    // Read and dequantize output tensors.
    print_outputs(&interpreter)?;

    Ok(())
}
