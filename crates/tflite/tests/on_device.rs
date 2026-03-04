// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! On-device integration tests for DMA-BUF and `CameraAdaptor`.
//!
//! These tests require an i.MX device with:
//! - `TFLite` shared library (`libtensorflowlite_c.so`)
//! - `VxDelegate` (`libvx_delegate.so`)
//! - NPU driver loaded
//! - DMA heap available (`/dev/dma_heap/linux,cma`)
//!
//! All tests are gated with `require_tflite!()` and `require_delegate!()` so
//! they compile everywhere but skip at runtime when the hardware is missing.
//!
//! # Running on device
//!
//! ```sh
//! # Cross-compile
//! cargo zigbuild --workspace --all-features --target aarch64-unknown-linux-gnu --tests
//!
//! # Deploy and run on device
//! scp target/aarch64-unknown-linux-gnu/debug/deps/on_device-* root@device:/tmp/
//! scp testdata/minimal.tflite root@device:/tmp/testdata/
//! ssh root@device 'cd /tmp && TFLITE_TEST_LIB=/usr/lib/libtensorflowlite_c.so \
//!     VX_DELEGATE_LIB=/usr/lib/libvx_delegate.so ./on_device-*'
//! ```

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Default `VxDelegate` library path.
const DEFAULT_DELEGATE_LIB: &str = "libvx_delegate.so";

/// Returns the `VxDelegate` library path from `VX_DELEGATE_LIB` env var or default.
fn delegate_lib_path() -> String {
    std::env::var("VX_DELEGATE_LIB").unwrap_or_else(|_| DEFAULT_DELEGATE_LIB.to_string())
}

/// Returns `true` if the `VxDelegate` shared library can be loaded.
fn delegate_available() -> bool {
    edgefirst_tflite::Delegate::load(delegate_lib_path()).is_ok()
}

/// Skip the calling test if no `VxDelegate` is available.
macro_rules! require_delegate {
    () => {
        common::require_tflite!();
        if !crate::delegate_available() {
            eprintln!(
                "SKIPPED: VxDelegate not available. \
                 Set VX_DELEGATE_LIB=/path/to/libvx_delegate.so to enable."
            );
            return;
        }
    };
}

/// Load delegate with default options.
fn load_delegate() -> edgefirst_tflite::Delegate {
    edgefirst_tflite::Delegate::load(delegate_lib_path()).expect("failed to load delegate")
}

/// Build interpreter with `VxDelegate` for the minimal test model.
fn build_delegated_interpreter<'lib>(
    lib: &'lib edgefirst_tflite::Library,
    model: &edgefirst_tflite::Model<'lib>,
) -> edgefirst_tflite::Interpreter<'lib> {
    let delegate = load_delegate();
    edgefirst_tflite::Interpreter::builder(lib)
        .expect("failed to create builder")
        .num_threads(1)
        .delegate(delegate)
        .build(model)
        .expect("failed to build interpreter")
}

/// Returns the path from `TFLITE_TEST_MODEL` env var, if set.
fn test_model_path() -> Option<String> {
    std::env::var("TFLITE_TEST_MODEL").ok()
}

/// Skip the calling test if `TFLITE_TEST_MODEL` is not set.
macro_rules! require_test_model {
    () => {
        require_delegate!();
        if crate::test_model_path().is_none() {
            eprintln!(
                "SKIPPED: TFLITE_TEST_MODEL not set. \
                 Set TFLITE_TEST_MODEL=/path/to/model.tflite to enable."
            );
            return;
        }
    };
}

// ===========================================================================
// Category A: Delegate Loading & Probing
// ===========================================================================

#[test]
fn delegate_load_succeeds() {
    require_delegate!();
    let delegate = load_delegate();
    let debug = format!("{delegate:?}");
    assert!(debug.contains("Delegate"));
}

#[test]
fn delegate_load_with_options_succeeds() {
    require_delegate!();
    let opts = edgefirst_tflite::DelegateOptions::new().option("device_id", "0");
    let delegate = edgefirst_tflite::Delegate::load_with_options(delegate_lib_path(), &opts)
        .expect("failed to load delegate with options");
    let debug = format!("{delegate:?}");
    assert!(debug.contains("Delegate"));
}

#[test]
fn delegate_load_bad_path_errors() {
    common::require_tflite!();
    let err = edgefirst_tflite::Delegate::load("/__nonexistent_delegate__.so").unwrap_err();
    assert!(err.is_library_error());
}

#[cfg(feature = "dmabuf")]
#[test]
fn delegate_has_dmabuf() {
    require_delegate!();
    let delegate = load_delegate();
    assert!(delegate.has_dmabuf(), "VxDelegate should support DMA-BUF");
}

#[cfg(feature = "dmabuf")]
#[test]
fn delegate_dmabuf_is_supported() {
    require_delegate!();
    let delegate = load_delegate();
    let dmabuf = delegate.dmabuf().expect("dmabuf() returned None");
    assert!(
        dmabuf.is_supported(),
        "DMA-BUF should be supported on i.MX with NPU"
    );
}

#[cfg(feature = "camera_adaptor")]
#[test]
fn delegate_has_camera_adaptor() {
    require_delegate!();
    let delegate = load_delegate();
    assert!(
        delegate.has_camera_adaptor(),
        "VxDelegate should support CameraAdaptor"
    );
}

// ===========================================================================
// Category B: Delegated Inference (basic NPU test)
// ===========================================================================

#[test]
fn delegated_inference_succeeds() {
    require_delegate!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let mut interp = build_delegated_interpreter(&lib, &model);

    // Write input: [1.0, 2.0, 3.0, 4.0]
    let input_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    {
        let mut inputs = interp.inputs_mut().unwrap();
        inputs[0].copy_from_slice(&input_data).unwrap();
    }

    interp.invoke().expect("delegated invoke should succeed");

    // Model adds [1,1,1,1] so expect [2,3,4,5].
    let outputs = interp.outputs().unwrap();
    let output_data = outputs[0].as_slice::<f32>().unwrap();
    let expected = [2.0f32, 3.0, 4.0, 5.0];
    for (got, want) in output_data.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-3,
            "expected {want}, got {got} (NPU may have small rounding)"
        );
    }
}

#[test]
fn delegated_interpreter_has_delegate() {
    require_delegate!();
    let lib = common::load_library().unwrap();
    let model = common::load_model(&lib);
    let interp = build_delegated_interpreter(&lib, &model);
    assert_eq!(interp.delegates().len(), 1);
    assert!(interp.delegate(0).is_some());
}

// ===========================================================================
// Category C: DMA-BUF Export Mode (delegate allocates buffers)
// ===========================================================================

#[cfg(feature = "dmabuf")]
mod dmabuf_export {
    use super::*;
    use edgefirst_tflite::dmabuf::Ownership;

    #[test]
    fn request_output_dmabuf() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        // Get output tensor size (float32[1,4] = 16 bytes)
        let outputs = interp.outputs().unwrap();
        let out_size = outputs[0].byte_size();
        assert!(out_size > 0);

        let (handle, desc) = dmabuf
            .request(0, Ownership::Delegate, out_size)
            .expect("request output dmabuf failed");

        assert!(desc.fd >= 0, "expected valid fd, got {}", desc.fd);
        assert!(desc.size >= out_size, "buffer too small");
        assert_ne!(
            handle.raw(),
            -1,
            "handle should not be kTfLiteNullBufferHandle"
        );

        // Verify fd matches via get_fd
        let queried_fd = dmabuf.fd(handle).expect("get_fd failed");
        assert_eq!(queried_fd, desc.fd);
    }

    #[test]
    fn request_input_dmabuf() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        let inputs = interp.inputs().unwrap();
        let in_size = inputs[0].byte_size();

        let (handle, desc) = dmabuf
            .request(0, Ownership::Delegate, in_size)
            .expect("request input dmabuf failed");

        assert!(desc.fd >= 0);
        assert!(desc.size >= in_size);

        // Bind to input tensor
        dmabuf
            .bind_to_tensor(handle, 0)
            .expect("bind_to_tensor failed");
    }

    #[test]
    fn request_and_release() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        let outputs = interp.outputs().unwrap();
        let out_size = outputs[0].byte_size();

        let (handle, _desc) = dmabuf
            .request(0, Ownership::Delegate, out_size)
            .expect("request failed");

        dmabuf.release(handle).expect("release failed");
    }

    #[test]
    fn request_with_zero_size_fails() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        // Size 0 should be rejected by the C API.
        let result = dmabuf.request(0, Ownership::Delegate, 0);
        assert!(result.is_err(), "request with size=0 should fail");
    }
}

// ===========================================================================
// Category D: DMA-BUF Import Mode (client allocates buffers)
// ===========================================================================

#[cfg(feature = "dmabuf")]
mod dmabuf_import {
    use super::*;
    use edgefirst_tflite::dmabuf::{Ownership, SyncMode};

    /// Allocate a DMA-BUF by requesting from the delegate (export mode),
    /// then use the fd for import-mode testing. This avoids needing direct
    /// `dma_heap` ioctls in Rust test code.
    fn allocate_dmabuf_fd(
        dmabuf: &edgefirst_tflite::dmabuf::DmaBuf<'_>,
        size: usize,
    ) -> (edgefirst_tflite::dmabuf::BufferHandle, i32) {
        let (handle, desc) = dmabuf
            .request(-1, Ownership::Delegate, size)
            .expect("export alloc for import test failed");
        (handle, desc.fd)
    }

    #[test]
    fn register_and_unregister() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        let inputs = interp.inputs().unwrap();
        let in_size = inputs[0].byte_size();

        // Use export mode to get a valid fd, then test import-mode registration
        let (export_handle, fd) = allocate_dmabuf_fd(&dmabuf, in_size);

        let import_handle = dmabuf
            .register(fd, in_size, SyncMode::None)
            .expect("register failed");

        assert_ne!(import_handle.raw(), -1);

        dmabuf.unregister(import_handle).expect("unregister failed");
        dmabuf.release(export_handle).expect("release failed");
    }

    #[test]
    fn register_bind_and_invoke() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let mut interp = build_delegated_interpreter(&lib, &model);

        let inputs = interp.inputs().unwrap();
        let in_size = inputs[0].byte_size();
        drop(inputs);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        // Allocate + register
        let (export_handle, fd) = allocate_dmabuf_fd(&dmabuf, in_size);
        let handle = dmabuf
            .register(fd, in_size, SyncMode::None)
            .expect("register failed");

        // Bind to input tensor 0
        dmabuf.bind_to_tensor(handle, 0).expect("bind failed");

        // Write input data via the tensor (CPU path for simplicity)
        {
            let mut inputs = interp.inputs_mut().unwrap();
            inputs[0].copy_from_slice(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        }

        // Invoke
        interp.invoke().expect("invoke with dmabuf should succeed");

        // Read output
        let outputs = interp.outputs().unwrap();
        let output_data = outputs[0].as_slice::<f32>().unwrap();
        assert_eq!(output_data.len(), 4);
        drop(outputs);

        // Cleanup
        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().unwrap();
        dmabuf.unregister(handle).expect("unregister failed");
        dmabuf.release(export_handle).expect("release failed");
    }
}

// ===========================================================================
// Category E: Cache Synchronization
// ===========================================================================

#[cfg(feature = "dmabuf")]
mod dmabuf_sync {
    use super::*;
    use edgefirst_tflite::dmabuf::{Ownership, SyncMode};

    #[test]
    fn begin_end_cpu_access() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        let inputs = interp.inputs().unwrap();
        let in_size = inputs[0].byte_size();

        let (handle, _desc) = dmabuf
            .request(0, Ownership::Delegate, in_size)
            .expect("request failed");

        // Begin CPU write access
        dmabuf
            .begin_cpu_access(handle, SyncMode::Write)
            .expect("begin_cpu_access failed");

        // End CPU write access (flush caches)
        dmabuf
            .end_cpu_access(handle, SyncMode::Write)
            .expect("end_cpu_access failed");

        // Begin CPU read access
        dmabuf
            .begin_cpu_access(handle, SyncMode::Read)
            .expect("begin_cpu_access read failed");

        // End CPU read access
        dmabuf
            .end_cpu_access(handle, SyncMode::Read)
            .expect("end_cpu_access read failed");
    }

    #[test]
    fn sync_for_device_and_cpu() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        let inputs = interp.inputs().unwrap();
        let in_size = inputs[0].byte_size();

        let (handle, _desc) = dmabuf
            .request(0, Ownership::Delegate, in_size)
            .expect("request failed");

        // Sync for device (flush CPU writes before NPU reads)
        dmabuf
            .sync_for_device(handle)
            .expect("sync_for_device failed");

        // Sync for CPU (invalidate cache before CPU reads NPU output)
        dmabuf.sync_for_cpu(handle).expect("sync_for_cpu failed");
    }
}

// ===========================================================================
// Category F: Buffer Cycling
// ===========================================================================

#[cfg(feature = "dmabuf")]
mod dmabuf_cycling {
    use super::*;
    use edgefirst_tflite::dmabuf::Ownership;

    #[test]
    fn set_and_get_active_buffer() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        let inputs = interp.inputs().unwrap();
        let in_size = inputs[0].byte_size();

        // Register two buffers
        let (h1, _) = dmabuf
            .request(0, Ownership::Delegate, in_size)
            .expect("request 1 failed");
        let (h2, _) = dmabuf
            .request(0, Ownership::Delegate, in_size)
            .expect("request 2 failed");

        // Bind first buffer
        dmabuf.bind_to_tensor(h1, 0).expect("bind h1 failed");

        // Set active to h1
        dmabuf.set_active(0, h1).expect("set_active h1 failed");
        let active = dmabuf.active_buffer(0);
        assert_eq!(active, Some(h1));

        // Switch to h2
        dmabuf.set_active(0, h2).expect("set_active h2 failed");
        let active = dmabuf.active_buffer(0);
        assert_eq!(active, Some(h2));
    }

    #[test]
    fn no_active_buffer_initially() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        // No buffer set yet for tensor 0
        assert_eq!(dmabuf.active_buffer(0), None);
    }
}

// ===========================================================================
// Category G: Graph State
// ===========================================================================

#[cfg(feature = "dmabuf")]
mod dmabuf_graph {
    use super::*;

    #[test]
    fn graph_compiled_after_invoke() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let mut interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let _dmabuf = delegate.dmabuf().expect("dmabuf not available");

        // Graph may or may not be compiled before first invoke (implementation-defined).
        // After invoke it should be compiled.
        interp.invoke().expect("invoke failed");

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().unwrap();
        assert!(
            dmabuf.is_graph_compiled(),
            "graph should be compiled after invoke"
        );
    }

    #[test]
    fn invalidate_graph_clears_compiled() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let mut interp = build_delegated_interpreter(&lib, &model);

        // First invoke to compile graph
        interp.invoke().expect("invoke failed");

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().unwrap();
        assert!(dmabuf.is_graph_compiled());

        // Invalidate
        dmabuf.invalidate_graph().expect("invalidate failed");
        assert!(
            !dmabuf.is_graph_compiled(),
            "graph should not be compiled after invalidation"
        );
    }
}

// ===========================================================================
// Category H: CameraAdaptor Queries (no model needed)
// ===========================================================================

#[cfg(feature = "camera_adaptor")]
mod camera_adaptor_queries {
    use super::*;

    #[test]
    fn rgba_is_supported() {
        require_delegate!();
        let delegate = load_delegate();
        let adaptor = delegate
            .camera_adaptor()
            .expect("camera_adaptor not available");
        assert!(adaptor.is_supported("rgba"));
    }

    #[test]
    fn bgra_is_supported() {
        require_delegate!();
        let delegate = load_delegate();
        let adaptor = delegate.camera_adaptor().unwrap();
        assert!(adaptor.is_supported("bgra"));
    }

    #[test]
    fn rgbx_is_supported() {
        require_delegate!();
        let delegate = load_delegate();
        let adaptor = delegate.camera_adaptor().unwrap();
        assert!(adaptor.is_supported("rgbx"));
    }

    #[test]
    fn rgb_is_supported() {
        require_delegate!();
        let delegate = load_delegate();
        let adaptor = delegate.camera_adaptor().unwrap();
        assert!(adaptor.is_supported("rgb"));
    }

    #[test]
    fn unsupported_format_returns_false() {
        require_delegate!();
        let delegate = load_delegate();
        let adaptor = delegate.camera_adaptor().unwrap();
        // YUV formats are not yet implemented
        assert!(!adaptor.is_supported("nv12"));
        assert!(!adaptor.is_supported("yuyv"));
    }

    #[test]
    fn rgba_input_channels_is_4() {
        require_delegate!();
        let delegate = load_delegate();
        let adaptor = delegate.camera_adaptor().unwrap();
        assert_eq!(adaptor.input_channels("rgba"), 4);
    }

    #[test]
    fn rgba_output_channels_is_3() {
        require_delegate!();
        let delegate = load_delegate();
        let adaptor = delegate.camera_adaptor().unwrap();
        assert_eq!(adaptor.output_channels("rgba"), 3);
    }

    #[test]
    fn rgb_channels_are_3() {
        require_delegate!();
        let delegate = load_delegate();
        let adaptor = delegate.camera_adaptor().unwrap();
        assert_eq!(adaptor.input_channels("rgb"), 3);
        assert_eq!(adaptor.output_channels("rgb"), 3);
    }

    #[test]
    fn fourcc_roundtrip() {
        require_delegate!();
        let delegate = load_delegate();
        let adaptor = delegate.camera_adaptor().unwrap();

        // Get FourCC for rgba, then convert back
        if let Some(fourcc) = adaptor.fourcc("rgba") {
            let back = adaptor.from_fourcc(&fourcc);
            assert!(
                back.is_some(),
                "from_fourcc should return a format for {fourcc}"
            );
        }
    }
}

// ===========================================================================
// Category I: CameraAdaptor with Delegate (graph integration)
// ===========================================================================

#[cfg(all(feature = "dmabuf", feature = "camera_adaptor"))]
mod camera_adaptor_integration {
    use super::*;
    use edgefirst_tflite::dmabuf::Ownership;

    #[test]
    fn set_format_before_build() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);

        let delegate = load_delegate();

        // Set CameraAdaptor format BEFORE building interpreter
        {
            let adaptor = delegate.camera_adaptor().expect("no camera_adaptor");
            adaptor
                .set_format(0, "rgba")
                .expect("set_format should succeed");
        }

        // Build interpreter with this delegate
        let mut interp = edgefirst_tflite::Interpreter::builder(&lib)
            .unwrap()
            .num_threads(1)
            .delegate(delegate)
            .build(&model)
            .expect("build should succeed with CameraAdaptor");

        // Invoke should succeed (graph compiled with Slice op for RGBA→RGB)
        interp
            .invoke()
            .expect("invoke with CameraAdaptor should succeed");
    }

    #[test]
    fn set_formats_rgba_to_rgb() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);

        let delegate = load_delegate();
        {
            let adaptor = delegate.camera_adaptor().expect("no camera_adaptor");
            adaptor
                .set_formats(0, "rgba", "rgb")
                .expect("set_formats should succeed");
        }

        let mut interp = edgefirst_tflite::Interpreter::builder(&lib)
            .unwrap()
            .num_threads(1)
            .delegate(delegate)
            .build(&model)
            .expect("build should succeed");

        interp.invoke().expect("invoke should succeed");
    }

    #[test]
    fn get_format_after_set() {
        require_delegate!();
        let delegate = load_delegate();
        {
            let adaptor = delegate.camera_adaptor().expect("no camera_adaptor");
            adaptor.set_format(0, "rgba").unwrap();
            let format = adaptor.format(0);
            assert!(format.is_some(), "format should be set after set_format");
        }
    }

    #[test]
    fn dmabuf_with_camera_adaptor_pipeline() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);

        // 1. Configure CameraAdaptor (before build)
        let delegate = load_delegate();
        {
            let adaptor = delegate.camera_adaptor().expect("no camera_adaptor");
            adaptor.set_format(0, "rgba").unwrap();
        }

        // 2. Build interpreter
        let mut interp = edgefirst_tflite::Interpreter::builder(&lib)
            .unwrap()
            .num_threads(1)
            .delegate(delegate)
            .build(&model)
            .expect("build failed");

        // 3. Get tensor sizes and request DMA-BUF
        //    With CameraAdaptor active, input buffer should be RGBA (4ch)
        //    but model tensor is RGB (3ch) or float32[1,4].
        //    For our minimal float32 model, the adaptor may not change sizes.
        let inputs = interp.inputs().unwrap();
        let in_size = inputs[0].byte_size();
        drop(inputs);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("no dmabuf");

        // Request input DMA-BUF
        let (in_handle, in_desc) = dmabuf
            .request(0, Ownership::Delegate, in_size)
            .expect("request input failed");
        assert!(in_desc.fd >= 0);

        // Bind to input tensor
        dmabuf
            .bind_to_tensor(in_handle, 0)
            .expect("bind input failed");

        // 4. Write input data (via CPU tensor for simplicity)
        {
            let mut inputs = interp.inputs_mut().unwrap();
            inputs[0].copy_from_slice(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        }

        // 5. Sync and invoke
        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().unwrap();
        dmabuf
            .sync_for_device(in_handle)
            .expect("sync_for_device failed");

        interp.invoke().expect("invoke failed");

        // 6. Read output
        let outputs = interp.outputs().unwrap();
        let output_data = outputs[0].as_slice::<f32>().unwrap();
        assert_eq!(output_data.len(), 4);
    }
}

// ===========================================================================
// Category J: Error Handling
// ===========================================================================

#[cfg(feature = "dmabuf")]
mod dmabuf_errors {
    use super::*;
    use edgefirst_tflite::dmabuf::SyncMode;

    #[test]
    fn register_invalid_fd_returns_handle() {
        require_delegate!();
        let lib = common::load_library().unwrap();
        let model = common::load_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        // fd=-1 is invalid but the delegate may still accept registration
        // (validation may be deferred to bind/invoke). Either success or
        // error is acceptable here.
        let _result = dmabuf.register(-1, 16, SyncMode::None);
        // Not asserting specific behavior - just ensuring no crash.
    }
}

// ===========================================================================
// Category K: Real Model Tests (TFLITE_TEST_MODEL, e.g. YOLOv8n)
// ===========================================================================

mod real_model {
    use super::*;

    /// Load model from `TFLITE_TEST_MODEL` path.
    fn load_real_model(lib: &edgefirst_tflite::Library) -> edgefirst_tflite::Model<'_> {
        let path = test_model_path().expect("TFLITE_TEST_MODEL must be set");
        edgefirst_tflite::Model::from_file(lib, &path)
            .unwrap_or_else(|e| panic!("failed to load model from {path}: {e}"))
    }

    #[test]
    fn real_model_loads() {
        require_test_model!();
        let lib = common::load_library().unwrap();
        let model = load_real_model(&lib);
        let data = model.data();
        assert!(data.len() > 1024, "model should be non-trivial in size");
    }

    #[test]
    fn real_model_delegated_inference() {
        require_test_model!();
        let lib = common::load_library().unwrap();
        let model = load_real_model(&lib);
        let mut interp = build_delegated_interpreter(&lib, &model);

        let input_count = interp.input_count();
        let output_count = interp.output_count();
        assert!(input_count >= 1, "model should have at least 1 input");
        assert!(output_count >= 1, "model should have at least 1 output");

        // Print tensor info
        {
            let inputs = interp.inputs().unwrap();
            for (i, t) in inputs.iter().enumerate() {
                eprintln!("  input[{i}]: {t}");
            }
        }
        {
            let outputs = interp.outputs().unwrap();
            for (i, t) in outputs.iter().enumerate() {
                eprintln!("  output[{i}]: {t}");
            }
        }

        // Zero-fill inputs and invoke
        {
            let mut inputs = interp.inputs_mut().unwrap();
            for input in &mut inputs {
                let byte_size = input.byte_size();
                let zeros = vec![0u8; byte_size];
                input.copy_from_slice(&zeros).unwrap();
            }
        }

        interp
            .invoke()
            .expect("delegated invoke on real model should succeed");

        // Verify outputs are readable
        let outputs = interp.outputs().unwrap();
        for (i, output) in outputs.iter().enumerate() {
            assert!(
                output.byte_size() > 0,
                "output[{i}] should have non-zero size"
            );
        }
    }

    #[cfg(feature = "dmabuf")]
    #[test]
    fn real_model_dmabuf_export_all_tensors() {
        require_test_model!();
        let lib = common::load_library().unwrap();
        let model = load_real_model(&lib);
        let interp = build_delegated_interpreter(&lib, &model);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("dmabuf not available");

        // Request DMA-BUFs for all input tensors
        let inputs = interp.inputs().unwrap();
        for (i, tensor) in inputs.iter().enumerate() {
            let size = tensor.byte_size();
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let tensor_index = i as i32;
            let (handle, desc) = dmabuf
                .request(
                    tensor_index,
                    edgefirst_tflite::dmabuf::Ownership::Delegate,
                    size,
                )
                .unwrap_or_else(|e| panic!("request input[{i}] dmabuf failed: {e}"));
            eprintln!(
                "  input[{i}] dmabuf: fd={}, size={}, handle={}",
                desc.fd,
                desc.size,
                handle.raw()
            );
            assert!(desc.fd >= 0);
            assert!(desc.size >= size);
            dmabuf
                .bind_to_tensor(handle, tensor_index)
                .unwrap_or_else(|e| panic!("bind input[{i}] failed: {e}"));
        }

        // Request DMA-BUFs for all output tensors
        let outputs = interp.outputs().unwrap();
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let output_base = interp.input_count() as i32;
        for (i, tensor) in outputs.iter().enumerate() {
            let size = tensor.byte_size();
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let tensor_idx = output_base + i as i32;
            let (handle, desc) = dmabuf
                .request(
                    tensor_idx,
                    edgefirst_tflite::dmabuf::Ownership::Delegate,
                    size,
                )
                .unwrap_or_else(|e| panic!("request output[{i}] dmabuf failed: {e}"));
            eprintln!(
                "  output[{i}] dmabuf: fd={}, size={}, handle={}",
                desc.fd,
                desc.size,
                handle.raw()
            );
            assert!(desc.fd >= 0);
            assert!(desc.size >= size);
        }
    }

    #[cfg(feature = "camera_adaptor")]
    #[test]
    fn real_model_camera_adaptor_rgba() {
        require_test_model!();
        let lib = common::load_library().unwrap();
        let model = load_real_model(&lib);

        let delegate = load_delegate();
        {
            let adaptor = delegate.camera_adaptor().expect("no camera_adaptor");
            adaptor
                .set_format(0, "rgba")
                .expect("set_format rgba should succeed");

            // With CameraAdaptor, input channels become 4 (RGBA)
            let in_ch = adaptor.input_channels("rgba");
            let out_ch = adaptor.output_channels("rgba");
            eprintln!("  CameraAdaptor rgba: in_ch={in_ch}, out_ch={out_ch}");
            assert_eq!(in_ch, 4);
            assert_eq!(out_ch, 3);
        }

        let mut interp = edgefirst_tflite::Interpreter::builder(&lib)
            .unwrap()
            .num_threads(1)
            .delegate(delegate)
            .build(&model)
            .expect("build with CameraAdaptor+rgba should succeed");

        // Print tensor shapes after CameraAdaptor injection
        {
            let inputs = interp.inputs().unwrap();
            for (i, t) in inputs.iter().enumerate() {
                eprintln!("  input[{i}] after CameraAdaptor: {t}");
            }
        }

        // Zero-fill and invoke
        {
            let mut inputs = interp.inputs_mut().unwrap();
            for input in &mut inputs {
                let zeros = vec![0u8; input.byte_size()];
                input.copy_from_slice(&zeros).unwrap();
            }
        }
        interp
            .invoke()
            .expect("invoke with CameraAdaptor+rgba should succeed");
    }

    #[cfg(all(feature = "dmabuf", feature = "camera_adaptor"))]
    #[test]
    fn real_model_full_pipeline() {
        require_test_model!();
        let lib = common::load_library().unwrap();
        let model = load_real_model(&lib);

        // 1. Configure CameraAdaptor for RGBA input
        let delegate = load_delegate();
        {
            let adaptor = delegate.camera_adaptor().expect("no camera_adaptor");
            adaptor.set_format(0, "rgba").unwrap();
        }

        // 2. Build interpreter
        let mut interp = edgefirst_tflite::Interpreter::builder(&lib)
            .unwrap()
            .num_threads(1)
            .delegate(delegate)
            .build(&model)
            .expect("build failed");

        // 3. Allocate DMA-BUF for input
        let inputs = interp.inputs().unwrap();
        let in_size = inputs[0].byte_size();
        drop(inputs);

        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().expect("no dmabuf");

        let (in_handle, in_desc) = dmabuf
            .request(0, edgefirst_tflite::dmabuf::Ownership::Delegate, in_size)
            .expect("request input dmabuf failed");
        eprintln!(
            "  input dmabuf: fd={}, size={}, handle={}",
            in_desc.fd,
            in_desc.size,
            in_handle.raw()
        );
        dmabuf
            .bind_to_tensor(in_handle, 0)
            .expect("bind input failed");

        // 4. Sync and invoke
        dmabuf
            .sync_for_device(in_handle)
            .expect("sync_for_device failed");

        // Zero-fill input via CPU tensor
        {
            let mut inputs = interp.inputs_mut().unwrap();
            let zeros = vec![0u8; inputs[0].byte_size()];
            inputs[0].copy_from_slice(&zeros).unwrap();
        }

        interp
            .invoke()
            .expect("full pipeline invoke should succeed");

        // 5. Read output
        let delegate = interp.delegate(0).unwrap();
        let dmabuf = delegate.dmabuf().unwrap();
        dmabuf.sync_for_cpu(in_handle).expect("sync_for_cpu failed");

        let outputs = interp.outputs().unwrap();
        for (i, output) in outputs.iter().enumerate() {
            eprintln!("  output[{i}]: {} bytes", output.byte_size());
            assert!(output.byte_size() > 0);
        }
    }
}

// ===========================================================================
// Category L: Metadata from Real Model
// ===========================================================================

#[cfg(feature = "metadata")]
mod real_model_metadata {
    use super::*;

    #[test]
    fn extract_metadata_from_real_model() {
        require_test_model!();
        let lib = common::load_library().unwrap();
        let path = test_model_path().expect("TFLITE_TEST_MODEL must be set");
        let model = edgefirst_tflite::Model::from_file(&lib, &path)
            .unwrap_or_else(|e| panic!("failed to load model from {path}: {e}"));

        let metadata = edgefirst_tflite::metadata::Metadata::from_model_bytes(model.data());
        eprintln!("  Metadata: {metadata}");
        // Just verify it doesn't panic and produces reasonable output.
    }
}
