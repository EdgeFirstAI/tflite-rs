// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Ergonomic Rust API for TensorFlow Lite inference.
//!
//! This crate provides a safe, idiomatic wrapper around the `TFLite` C API
//! with support for hardware-accelerated inference via delegates, DMA-BUF
//! zero-copy, and NPU preprocessing.
//!
//! # Quick Start
//!
//! ```no_run
//! use edgefirst_tflite::{Library, Model, Interpreter};
//!
//! let lib = Library::new()?;
//! let model = Model::from_file(&lib, "model.tflite")?;
//!
//! let mut interpreter = Interpreter::builder(&lib)?
//!     .num_threads(4)
//!     .build(&model)?;
//!
//! // Populate input tensors...
//! interpreter.invoke()?;
//!
//! let outputs = interpreter.outputs()?;
//! # Ok::<(), edgefirst_tflite::Error>(())
//! ```
//!
//! # Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `dmabuf` | DMA-BUF zero-copy inference via HAL Delegate API |
//! | `camera_adaptor` | NPU-accelerated format conversion |
//! | `metadata` | `TFLite` model metadata extraction (`TFLITE_METADATA` buffer) |
//! | `archive` | Embedded ZIP-archive metadata (`edgefirst.json`, `labels.txt`) |
//! | `full` | Enables all optional features |
//!
//! # `LiteRT` Next
//!
//! When the loaded shared library exports `LiteRt*` symbols (for example the
//! Android `LiteRT` runtime), the [`litert`] module provides an ergonomic
//! `CompiledModel` path with explicit accelerator selection and buffer
//! placement. Classic TensorFlow Lite libraries continue to work through
//! [`Interpreter`]; `LiteRT` constructors then return an error for which
//! [`Error::is_litert_unavailable`] is `true`, and
//! [`Error::litert_missing_symbol`] names the symbol that was absent.
//!
//! Branch on [`Library::has_litert`] to choose a path:
//!
//! ```no_run
//! use edgefirst_tflite::{litert, Interpreter, Library, Model};
//!
//! let lib = Library::new()?;
//! if lib.has_litert() {
//!     let env = litert::Environment::new(&lib)?;
//!     let model = litert::Model::from_file(&env, "model.tflite")?;
//!     // ... compile and run via litert::CompiledModel
//! } else {
//!     let model = Model::from_file(&lib, "model.tflite")?;
//!     let interpreter = Interpreter::builder(&lib)?.build(&model)?;
//!     // ... run via the classic Interpreter
//! }
//! # Ok::<(), edgefirst_tflite::Error>(())
//! ```

pub mod delegate;
pub mod error;
pub mod interpreter;
pub mod library;
pub mod litert;
pub mod model;
pub mod profiler;
pub mod tensor;

#[cfg(feature = "dmabuf")]
pub mod dmabuf;

#[cfg(feature = "camera_adaptor")]
pub mod camera_adaptor;

#[cfg(feature = "metadata")]
pub mod metadata;

#[cfg(feature = "archive")]
pub mod archive;

// Public re-exports for convenience.
pub use delegate::{Delegate, DelegateOptions};
pub use error::{Error, LiteRtStatusCode, StatusCode};
pub use interpreter::{Interpreter, InterpreterBuilder};
pub use library::Library;
pub use model::Model;
pub use profiler::{OpEvent, Profiler};
pub use tensor::{QuantizationParams, Tensor, TensorMut, TensorType};

// ---------------------------------------------------------------------------
// Compile-time thread-safety assertions
// ---------------------------------------------------------------------------

const fn assert_send<T: Send>() {}
const fn assert_sync<T: Sync>() {}

const _: () = {
    // Library: loaded function table — safe to share across threads.
    assert_send::<Library>();
    assert_sync::<Library>();

    // Model: read-only after creation — safe to share across threads.
    assert_send::<Model<'_>>();
    assert_sync::<Model<'_>>();

    // Interpreter: mutable state — can be sent but not shared.
    assert_send::<Interpreter<'_>>();
    // Interpreter is intentionally NOT Sync.

    // Delegate: opaque handle — safe to send and share.
    assert_send::<Delegate>();
    assert_sync::<Delegate>();

    // LiteRT handles. Environment / Model / Options are Sync because every
    // operation reachable through a shared reference is a read. CompiledModel
    // and TensorBuffer are Send-only: inference and host mapping mutate runtime
    // state, and both gate that behind `&mut self`. See each type's
    // `# Thread safety` documentation for the full argument.
    assert_send::<litert::Environment<'_>>();
    assert_sync::<litert::Environment<'_>>();
    assert_send::<litert::Model<'_>>();
    assert_sync::<litert::Model<'_>>();
    assert_send::<litert::Options<'_>>();
    assert_sync::<litert::Options<'_>>();
    assert_send::<litert::CompiledModel<'_>>();
    assert_send::<litert::TensorBuffer<'_>>();
};
