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
// XNNPACK delegate
// ---------------------------------------------------------------------------

#[test]
fn xnnpack_delegate_does_not_panic() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    // The result depends on whether the library includes XNNPACK symbols.
    // We verify it does not panic regardless.
    let _result = edgefirst_tflite::Delegate::xnnpack(&lib, 4);
}

#[test]
fn xnnpack_delegate_invoke_succeeds() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();

    let delegate = match edgefirst_tflite::Delegate::xnnpack(&lib, 2) {
        Ok(d) => d,
        Err(e) if e.is_invalid_argument() => {
            eprintln!("SKIPPED: XNNPACK not available in this TFLite build");
            return;
        }
        Err(e) => panic!("Delegate::xnnpack failed with unexpected error: {e}"),
    };

    let model = common::load_model(&lib);
    let mut interp = edgefirst_tflite::Interpreter::builder(&lib)
        .unwrap()
        .delegate(delegate)
        .num_threads(2)
        .build(&model)
        .unwrap();

    let input_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    {
        let mut inputs = interp.inputs_mut().unwrap();
        inputs[0].copy_from_slice(&input_data).unwrap();
    }

    interp.invoke().expect("invoke with XNNPACK should succeed");

    let outputs = interp.outputs().unwrap();
    let output_data = outputs[0].as_slice::<f32>().unwrap();
    let expected = [2.0f32, 3.0, 4.0, 5.0];
    assert_eq!(output_data.len(), expected.len());
    for (got, want) in output_data.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-5, "expected {want}, got {got}");
    }
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

// ---------------------------------------------------------------------------
// Multi-interpreter: concurrent inference from shared model
// ---------------------------------------------------------------------------

#[test]
fn multi_interpreter_same_model() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);

    // Create two interpreters from the same model.
    let mut interp_a = common::build_interpreter(&lib, &model);
    let mut interp_b = common::build_interpreter(&lib, &model);

    // Write different inputs to each.
    {
        let mut inputs = interp_a.inputs_mut().unwrap();
        inputs[0].copy_from_slice(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    }
    {
        let mut inputs = interp_b.inputs_mut().unwrap();
        inputs[0]
            .copy_from_slice(&[10.0f32, 20.0, 30.0, 40.0])
            .unwrap();
    }

    // Invoke both — they should not interfere.
    interp_a.invoke().unwrap();
    interp_b.invoke().unwrap();

    // Model adds [1,1,1,1].
    let out_a = interp_a.outputs().unwrap();
    let out_b = interp_b.outputs().unwrap();
    let data_a = out_a[0].as_slice::<f32>().unwrap();
    let data_b = out_b[0].as_slice::<f32>().unwrap();

    let expected_a = [2.0f32, 3.0, 4.0, 5.0];
    let expected_b = [11.0f32, 21.0, 31.0, 41.0];

    for (got, want) in data_a.iter().zip(expected_a.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "interp_a: expected {want}, got {got}"
        );
    }
    for (got, want) in data_b.iter().zip(expected_b.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "interp_b: expected {want}, got {got}"
        );
    }
}

#[test]
fn multi_interpreter_threaded() {
    const NUM_THREADS: usize = 4;
    const ITERATIONS: usize = 50;

    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);

    // Create one interpreter per thread.
    let interpreters: Vec<_> = (0..NUM_THREADS)
        .map(|_| common::build_interpreter(&lib, &model))
        .collect();

    // Move each interpreter into its own thread.
    std::thread::scope(|s| {
        let handles: Vec<_> = interpreters
            .into_iter()
            .enumerate()
            .map(|(thread_id, mut interp)| {
                s.spawn(move || {
                    for i in 0..ITERATIONS {
                        let base = f32::from((thread_id * 100 + i) as u16);
                        let input = [base, base + 1.0, base + 2.0, base + 3.0];
                        {
                            let mut inputs = interp.inputs_mut().unwrap();
                            inputs[0].copy_from_slice(&input).unwrap();
                        }

                        interp.invoke().unwrap();

                        let outputs = interp.outputs().unwrap();
                        let data = outputs[0].as_slice::<f32>().unwrap();

                        // Model adds [1,1,1,1].
                        let expected = [base + 1.0, base + 2.0, base + 3.0, base + 4.0];
                        for (got, want) in data.iter().zip(expected.iter()) {
                            assert!(
                                (got - want).abs() < 1e-5,
                                "thread {thread_id} iter {i}: expected {want}, got {got}"
                            );
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("worker thread panicked");
        }
    });
}
