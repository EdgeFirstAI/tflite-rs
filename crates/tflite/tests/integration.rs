// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Integration tests for `edgefirst-tflite`.
//!
//! All tests are gated with `require_tflite!()` -- they compile everywhere but
//! skip at runtime when no `TFLite` shared library is available.

mod common;

use edgefirst_tflite::TensorType;

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

#[test]
fn library_new_succeeds() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let debug = format!("{lib:?}");
    assert!(debug.contains("Library"));
}

#[test]
fn library_from_path_bad_path_errors() {
    common::require_tflite!();
    let err = edgefirst_tflite::Library::from_path("/__nonexistent_lib__.so").unwrap_err();
    assert!(err.is_library_error());
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[test]
fn model_from_bytes_valid() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    assert!(!model.data().is_empty());
}

#[test]
fn model_from_bytes_invalid() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let result = edgefirst_tflite::Model::from_bytes(&lib, [0xFF; 4]);
    assert!(result.is_err());
}

#[test]
fn model_data_roundtrip() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    assert_eq!(model.data(), common::MINIMAL_MODEL);
}

#[test]
fn model_from_file_succeeds() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = edgefirst_tflite::Model::from_file(&lib, "../../testdata/minimal.tflite")
        .expect("failed to load model from file");
    assert_eq!(model.data(), common::MINIMAL_MODEL);
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

#[test]
fn interpreter_builder_creates() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let builder = edgefirst_tflite::Interpreter::builder(&lib).unwrap();
    let debug = format!("{builder:?}");
    assert!(debug.contains("InterpreterBuilder"));
}

#[test]
fn interpreter_num_threads_works() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    // Should not panic with different thread counts.
    let _interp = edgefirst_tflite::Interpreter::builder(&lib)
        .unwrap()
        .num_threads(2)
        .build(&model)
        .unwrap();
}

#[test]
fn interpreter_build_succeeds() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    let debug = format!("{interp:?}");
    assert!(debug.contains("Interpreter"));
}

#[test]
fn interpreter_invoke_succeeds() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let mut interp = common::build_interpreter(&lib, &model);
    interp.invoke().expect("invoke should succeed");
}

#[test]
fn interpreter_input_output_count() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    assert_eq!(interp.input_count(), 1);
    assert_eq!(interp.output_count(), 1);
}

#[test]
fn interpreter_inputs_returns_tensors() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    let inputs = interp.inputs().unwrap();
    assert_eq!(inputs.len(), 1);
}

#[test]
fn interpreter_outputs_returns_tensors() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    let outputs = interp.outputs().unwrap();
    assert_eq!(outputs.len(), 1);
}

#[test]
fn interpreter_delegates_empty_without_delegates() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    assert!(interp.delegates().is_empty());
}

// ---------------------------------------------------------------------------
// Tensor (immutable)
// ---------------------------------------------------------------------------

#[test]
fn tensor_type_is_float32() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    let inputs = interp.inputs().unwrap();
    assert_eq!(inputs[0].tensor_type(), TensorType::Float32);
}

#[test]
fn tensor_shape_matches_model() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    let inputs = interp.inputs().unwrap();
    let shape = inputs[0].shape().unwrap();
    assert_eq!(shape, vec![1, 4]);
}

#[test]
fn tensor_byte_size_and_volume_consistent() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    let inputs = interp.inputs().unwrap();
    let volume = inputs[0].volume().unwrap();
    let byte_size = inputs[0].byte_size();
    // float32 = 4 bytes per element
    assert_eq!(byte_size, volume * std::mem::size_of::<f32>());
}

#[test]
fn tensor_as_slice_f32_correct_length() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    let inputs = interp.inputs().unwrap();
    let slice = inputs[0].as_slice::<f32>().unwrap();
    assert_eq!(slice.len(), 4);
}

#[test]
fn tensor_display_format() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    let inputs = interp.inputs().unwrap();
    let display = format!("{}", inputs[0]);
    assert!(display.contains("Float32"));
    assert!(display.contains("1x4") || display.contains('4'));
}

// ---------------------------------------------------------------------------
// TensorMut (mutable)
// ---------------------------------------------------------------------------

#[test]
fn tensor_mut_copy_from_slice_roundtrip() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let mut interp = common::build_interpreter(&lib, &model);

    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    {
        let mut inputs = interp.inputs_mut().unwrap();
        inputs[0].copy_from_slice(&data).unwrap();
        let readback = inputs[0].as_slice::<f32>().unwrap();
        assert_eq!(readback, &data);
    }
}

#[test]
fn tensor_mut_as_mut_slice_write() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let mut interp = common::build_interpreter(&lib, &model);

    {
        let mut inputs = interp.inputs_mut().unwrap();
        let slice = inputs[0].as_mut_slice::<f32>().unwrap();
        slice[0] = 42.0;
        slice[1] = 43.0;
    }

    let inputs = interp.inputs().unwrap();
    let readback = inputs[0].as_slice::<f32>().unwrap();
    assert!((readback[0] - 42.0).abs() < f32::EPSILON);
    assert!((readback[1] - 43.0).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// Full pipeline: load -> write input -> invoke -> read output
// ---------------------------------------------------------------------------

#[test]
fn full_pipeline_end_to_end() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let mut interp = common::build_interpreter(&lib, &model);

    // Write input: [1.0, 2.0, 3.0, 4.0]
    let input_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    {
        let mut inputs = interp.inputs_mut().unwrap();
        inputs[0].copy_from_slice(&input_data).unwrap();
    }

    // Run inference.
    interp.invoke().expect("invoke should succeed");

    // Read output: model adds [1,1,1,1] so expect [2,3,4,5].
    let outputs = interp.outputs().unwrap();
    let output_data = outputs[0].as_slice::<f32>().unwrap();
    assert_eq!(output_data.len(), 4);

    let expected = [2.0f32, 3.0, 4.0, 5.0];
    for (got, want) in output_data.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-5, "expected {want}, got {got}");
    }
}
