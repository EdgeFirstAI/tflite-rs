// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Integration tests for `edgefirst-tflite`.
//!
//! All tests are gated with `require_tflite!()` -- they compile everywhere but
//! skip at runtime when no `TFLite` shared library is available.

mod common;

use edgefirst_tflite::{litert, TensorType};

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
// Raw-byte accessors + element-type guard (float32 I/O)
//
// minimal.tflite has a Float32 input and output (4 elements, +1.0 each).
// A raw-byte accessor must span the FULL byte_size (16 bytes), not the
// element count (4). Before the byte accessors existed, callers reached for
// `as_slice::<u8>()`, which returned a 4-byte (quarter) view and silently
// truncated every float32 copy -- the on-device "Input tensor N lacks data"
// crash. These tests pin the correct lengths and the loud rejection.
// ---------------------------------------------------------------------------

#[test]
fn tensor_as_bytes_spans_full_byte_size() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    let inputs = interp.inputs().unwrap();
    let byte_size = inputs[0].byte_size();
    assert_eq!(byte_size, 16, "4 float32 elements");
    // The bug: a u8 view must be all 16 bytes, not the 4-element volume.
    assert_eq!(inputs[0].as_bytes().unwrap().len(), byte_size);
}

#[test]
fn tensor_as_slice_u8_on_float32_is_rejected() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = common::build_interpreter(&lib, &model);
    let inputs = interp.inputs().unwrap();
    // Previously returned a 4-byte (quarter) slice; now a typed error.
    let err = inputs[0].as_slice::<u8>().unwrap_err();
    assert!(err.is_invalid_argument(), "{err}");
    assert!(err.to_string().contains("width mismatch"), "{err}");
}

#[test]
fn tensor_mut_copy_from_bytes_roundtrip() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let mut interp = common::build_interpreter(&lib, &model);

    // Stage four f32 values as raw native-endian bytes -- the shape the
    // profiler's preprocess produces -- through the byte API, then invoke
    // and confirm the model saw the whole buffer (adds 1.0 to each).
    // Native, not little: copy_from_bytes writes straight into the tensor
    // allocation that TFLite reads as host-order f32, so a fixed
    // little-endian encoding would decode as garbage on a big-endian host.
    let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut src = Vec::new();
    for v in values {
        src.extend_from_slice(&v.to_ne_bytes());
    }
    {
        let mut inputs = interp.inputs_mut().unwrap();
        assert_eq!(src.len(), inputs[0].byte_size());
        inputs[0].copy_from_bytes(&src).unwrap();
    }
    interp.invoke().unwrap();

    // Read the output back through the byte accessor and reinterpret.
    let outputs = interp.outputs().unwrap();
    let out_bytes = outputs[0].as_bytes().unwrap();
    assert_eq!(out_bytes.len(), 16);
    let (chunks, _) = out_bytes.as_chunks::<4>();
    let out: Vec<f32> = chunks.iter().map(|c| f32::from_ne_bytes(*c)).collect();
    for (got, want) in out.iter().zip([2.0f32, 3.0, 4.0, 5.0]) {
        assert!((got - want).abs() < f32::EPSILON, "got {got}, want {want}");
    }
}

#[test]
fn tensor_mut_copy_from_bytes_wrong_length_rejected() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let mut interp = common::build_interpreter(&lib, &model);
    let mut inputs = interp.inputs_mut().unwrap();
    // A quarter-sized buffer -- the exact truncation the old code produced --
    // is rejected rather than silently partially copied.
    let err = inputs[0].copy_from_bytes(&[0u8; 4]).unwrap_err();
    assert!(err.is_invalid_argument(), "{err}");
    assert!(err.to_string().contains("byte size"), "{err}");
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
                        #[allow(clippy::cast_possible_truncation)]
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

// ---------------------------------------------------------------------------
// LiteRT (soft-optional)
// ---------------------------------------------------------------------------

#[test]
fn litert_probe_matches_has_litert() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    assert_eq!(lib.has_litert(), lib.litert().is_some());
    // The availability flag and the diagnostic must never disagree.
    assert_eq!(lib.has_litert(), lib.litert_missing_symbol().is_none());
}

#[test]
fn litert_unavailable_when_symbols_missing() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    if lib.has_litert() {
        eprintln!("SKIPPED: library has LiteRT; unavailable path not exercised");
        return;
    }
    let err = litert::Environment::new(&lib).unwrap_err();
    assert!(err.is_litert_unavailable());
    // The error must name the unresolved symbol and agree with the probe.
    assert!(err.litert_missing_symbol().is_some());
    assert_eq!(err.litert_missing_symbol(), lib.litert_missing_symbol());
}

#[test]
fn litert_environment_and_accelerators() {
    common::require_litert!();
    let lib = common::load_library().unwrap();
    let env = litert::Environment::new(&lib).expect("Environment::new");
    let accels = litert::accelerators(&env).expect("accelerators");

    // Environment creation always registers the CPU accelerator.
    assert!(
        accels
            .iter()
            .any(|a| a.hardware.contains(litert::HwAccelerators::CPU)),
        "expected a CPU accelerator, got {accels:?}"
    );
    for accel in &accels {
        assert!(!accel.to_string().is_empty());
    }
}

#[test]
fn litert_options_round_trip_accelerators() {
    common::require_litert!();
    let lib = common::load_library().unwrap();
    let requested = litert::HwAccelerators::CPU | litert::HwAccelerators::GPU;
    let opts = litert::Options::new(&lib)
        .unwrap()
        .hardware_accelerators(requested)
        .unwrap();
    assert_eq!(opts.hardware_accelerator_set().unwrap(), requested);
}

#[test]
fn litert_compiled_model_sync_run() {
    common::require_litert!();
    let lib = common::load_library().unwrap();
    let env = litert::Environment::new(&lib).unwrap();
    let model = litert::Model::from_buffer(&env, common::MINIMAL_MODEL).unwrap();
    assert_eq!(model.num_signatures().unwrap(), 1);
    assert_eq!(model.num_inputs(0).unwrap(), 1);
    assert_eq!(model.num_outputs(0).unwrap(), 1);

    let opts = litert::Options::new(&lib)
        .unwrap()
        .hardware_accelerators(litert::HwAccelerators::CPU)
        .unwrap();
    let mut compiled = litert::CompiledModel::create(&env, &model, &opts).unwrap();

    // The minimal model is a single Add, which XNNPACK claims in full.
    assert!(compiled.is_fully_accelerated().unwrap());

    // Requirements report the logical size; the allocation may be padded
    // beyond it but never below.
    let reqs = compiled.input_buffer_requirements(0, 0).unwrap();
    assert_eq!(reqs.size(), 16);

    let mut inputs = vec![compiled.create_input_buffer(0, 0).unwrap()];
    let mut outputs = vec![compiled.create_output_buffer(0, 0).unwrap()];
    assert!(inputs[0].size() >= reqs.size());
    inputs[0].tensor_type().expect("input tensor type");

    let input_data = [1.0f32, 2.0, 3.0, 4.0];
    let mut bytes = Vec::with_capacity(16);
    for v in input_data {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    inputs[0].write_bytes(&bytes).unwrap();

    compiled.run_default(&mut inputs, &mut outputs).unwrap();

    let out = outputs[0].read_bytes().unwrap();
    assert_eq!(out.len(), outputs[0].size());

    // Read back only the logical tensor, skipping any alignment padding.
    let mut logical = vec![0u8; 16];
    outputs[0].read_bytes_into(&mut logical).unwrap();
    assert_eq!(&out[..16], &logical[..]);

    // A second run must reuse the internal handle arrays without corruption.
    compiled.run_default(&mut inputs, &mut outputs).unwrap();
    assert_eq!(
        out,
        outputs[0].read_bytes().unwrap(),
        "repeat inference must be deterministic"
    );
}

#[test]
fn litert_rejects_out_of_bounds_buffer_access() {
    common::require_litert!();
    let lib = common::load_library().unwrap();
    let env = litert::Environment::new(&lib).unwrap();
    let model = litert::Model::from_buffer(&env, common::MINIMAL_MODEL).unwrap();
    let opts = litert::Options::new(&lib)
        .unwrap()
        .hardware_accelerators(litert::HwAccelerators::CPU)
        .unwrap();
    let compiled = litert::CompiledModel::create(&env, &model, &opts).unwrap();
    let mut buffer = compiled.create_input_buffer(0, 0).unwrap();

    let too_big = vec![0u8; buffer.size() + 1];
    let err = buffer.write_bytes(&too_big).unwrap_err();
    assert!(err.is_invalid_argument(), "{err}");

    let mut dst = vec![0u8; buffer.size() + 1];
    let err = buffer.read_bytes_into(&mut dst).unwrap_err();
    assert!(err.is_invalid_argument(), "{err}");

    // A repeated write must still succeed — the failed attempts must not have
    // left the buffer locked.
    buffer.write_bytes(&vec![0u8; buffer.size()]).unwrap();
}

#[test]
fn litert_model_rejects_invalid_buffer() {
    common::require_litert!();
    let lib = common::load_library().unwrap();
    let env = litert::Environment::new(&lib).unwrap();
    let err = litert::Model::from_buffer(&env, vec![0u8; 32]).unwrap_err();
    assert!(err.litert_status_code().is_some(), "{err}");
}

// ---------------------------------------------------------------------------
// Custom allocation (experimental C API)
// ---------------------------------------------------------------------------

/// A heap buffer whose usable region starts on a `kDefaultTensorAlignment`
/// boundary, which is what
/// [`Interpreter::set_custom_allocation_for_input`] requires.
///
/// Over-allocates and offsets rather than using a custom `Layout` so the
/// storage is still an ordinary `Vec` with ordinary drop behaviour. The heap
/// block does not move when the `AlignedBuffer` value moves, so the address
/// handed to the runtime stays valid as long as this value is alive.
struct AlignedBuffer {
    storage: Vec<u8>,
    offset: usize,
    len: usize,
}

impl AlignedBuffer {
    const ALIGNMENT: usize = 64;

    fn new(len: usize) -> Self {
        let storage = vec![0u8; len + Self::ALIGNMENT];
        let offset = storage.as_ptr().align_offset(Self::ALIGNMENT);
        assert!(offset + len <= storage.len());
        Self {
            storage,
            offset,
            len,
        }
    }

    fn ptr(&mut self) -> std::ptr::NonNull<u8> {
        // SAFETY: `offset + len <= storage.len()`, asserted in `new`.
        let p = unsafe { self.storage.as_mut_ptr().add(self.offset) };
        std::ptr::NonNull::new(p).expect("Vec never allocates at null")
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.storage[self.offset..self.offset + self.len]
    }
}

/// Skip unless the runtime exports the experimental C API.
macro_rules! require_custom_allocation {
    ($lib:expr) => {
        if !$lib.has_custom_allocation() {
            eprintln!(
                "SKIPPED: this TFLite build does not export \
                 TfLiteInterpreterSetCustomAllocationForTensor."
            );
            return;
        }
    };
}

#[test]
fn custom_allocation_backs_the_input_tensor() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_custom_allocation!(lib);
    let model = common::load_model(&lib);

    // Declared before the interpreter so it outlives every read of the
    // pointer handed to the runtime.
    let mut buffer = AlignedBuffer::new(4 * std::mem::size_of::<f32>());
    for (i, value) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
        buffer.as_mut_slice()[i * 4..(i + 1) * 4].copy_from_slice(&value.to_ne_bytes());
    }

    let mut interp = common::build_interpreter(&lib, &model);
    let bytes = interp.inputs().unwrap()[0].byte_size();
    let ptr = buffer.ptr();

    // SAFETY: `buffer` outlives `interp`, its heap block is not relocated
    // while it is alive, and nothing else writes it during `invoke`.
    unsafe {
        interp
            .set_custom_allocation_for_input(0, ptr, bytes)
            .unwrap();
    }
    interp.allocate_tensors().unwrap();
    interp.invoke().unwrap();

    // minimal.tflite adds 1.0 elementwise, so the model read *our* buffer
    // rather than the arena iff the output is the input plus one.
    let outputs = interp.outputs().unwrap();
    let out = outputs[0].as_slice::<f32>().unwrap();
    assert_eq!(out, &[2.0f32, 3.0, 4.0, 5.0]);
}

#[test]
fn custom_allocation_rejects_undersized_buffer() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_custom_allocation!(lib);
    let model = common::load_model(&lib);

    let mut buffer = AlignedBuffer::new(64);
    let mut interp = common::build_interpreter(&lib, &model);
    let ptr = buffer.ptr();

    // SAFETY: the call is rejected before the pointer is ever dereferenced.
    let err = unsafe { interp.set_custom_allocation_for_input(0, ptr, 4) }.unwrap_err();
    assert!(err.is_invalid_argument(), "{err}");
    assert!(err.to_string().contains("smaller"), "{err}");
}

#[test]
fn custom_allocation_rejects_misaligned_buffer() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_custom_allocation!(lib);
    let model = common::load_model(&lib);

    let mut buffer = AlignedBuffer::new(128);
    let aligned = buffer.ptr();
    // One byte past the aligned start: valid memory, wrong alignment.
    // SAFETY: `aligned + 1` is still inside the 128-byte region.
    let misaligned = std::ptr::NonNull::new(unsafe { aligned.as_ptr().add(1) }).unwrap();

    let mut interp = common::build_interpreter(&lib, &model);
    // SAFETY: the call is rejected before the pointer is ever dereferenced.
    let err = unsafe { interp.set_custom_allocation_for_input(0, misaligned, 64) }.unwrap_err();
    assert!(err.is_invalid_argument(), "{err}");
    assert!(err.to_string().contains("aligned"), "{err}");
}

#[test]
fn custom_allocation_rejects_out_of_range_input() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    require_custom_allocation!(lib);
    let model = common::load_model(&lib);

    let mut buffer = AlignedBuffer::new(64);
    let mut interp = common::build_interpreter(&lib, &model);
    let inputs = interp.input_count();
    let ptr = buffer.ptr();

    // SAFETY: the call is rejected before the pointer is ever dereferenced.
    let err = unsafe { interp.set_custom_allocation_for_input(inputs, ptr, 64) }.unwrap_err();
    assert!(err.is_invalid_argument(), "{err}");
    assert!(err.to_string().contains("out of range"), "{err}");
}

#[test]
fn custom_allocation_probe_matches_symbol_presence() {
    common::require_tflite!();
    let lib = common::load_library().unwrap();
    // Whatever the probe decided, the error path must agree with it: an
    // unsupported runtime reports the API by name rather than a bare status.
    if lib.has_custom_allocation() {
        return;
    }
    let model = common::load_model(&lib);
    let mut buffer = AlignedBuffer::new(64);
    let mut interp = common::build_interpreter(&lib, &model);
    let ptr = buffer.ptr();
    // SAFETY: the call is rejected before the pointer is ever dereferenced.
    let err = unsafe { interp.set_custom_allocation_for_input(0, ptr, 64) }.unwrap_err();
    assert!(err.is_unsupported(), "{err}");
    assert_eq!(
        err.unsupported_api(),
        Some("TfLiteInterpreterSetCustomAllocationForTensor")
    );
}
