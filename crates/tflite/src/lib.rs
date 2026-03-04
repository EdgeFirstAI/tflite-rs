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
//! | `dmabuf` | DMA-BUF zero-copy inference via `VxDelegate` |
//! | `camera_adaptor` | NPU-accelerated format conversion |
//! | `metadata` | `TFLite` model metadata extraction |
//! | `full` | Enables all optional features |

pub mod delegate;
pub mod error;
pub mod interpreter;
pub mod library;
pub mod model;
pub mod tensor;

#[cfg(feature = "dmabuf")]
pub mod dmabuf;

#[cfg(feature = "camera_adaptor")]
pub mod camera_adaptor;

#[cfg(feature = "metadata")]
pub mod metadata;

// Public re-exports for convenience.
pub use delegate::{Delegate, DelegateOptions};
pub use error::{Error, StatusCode};
pub use interpreter::{Interpreter, InterpreterBuilder};
pub use library::Library;
pub use model::Model;
pub use tensor::{QuantizationParams, Tensor, TensorMut, TensorType};
