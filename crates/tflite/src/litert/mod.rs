// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Ergonomic `LiteRT` Next API — the `CompiledModel` inference path.
//!
//! `LiteRT` Next is Google's successor to the classic TensorFlow Lite
//! `Interpreter`. Instead of building an interpreter and mutating its tensors,
//! you *compile* a model for a set of hardware accelerators and then run it
//! against explicitly allocated [`TensorBuffer`]s. That split is what lets the
//! runtime place buffers in memory the accelerator can reach (shared, DMA-able,
//! or device-local) rather than always copying through host memory.
//!
//! This module is **soft-optional**. It is compiled unconditionally, but every
//! constructor first checks that the loaded shared library actually exports the
//! `LiteRt*` symbols. On a classic TensorFlow Lite build they return an error
//! for which [`Error::is_litert_unavailable`] is `true`, and
//! [`Error::litert_missing_symbol`] names the symbol that was absent. Nothing
//! panics, and [`crate::Interpreter`] remains fully usable.
//!
//! # Object graph and lifetimes
//!
//! Handles form a strict ownership chain, and each type's lifetime parameter
//! names what it borrows. The borrow checker enforces the C API's own
//! requirements — several of these objects hold interior pointers into their
//! parent, and destroying a parent early is undefined behaviour:
//!
//! ```text
//! Library                       // owns the dlopen handle
//!  ├── Environment<'lib>        // borrows Library
//!  │    ├── Model<'env>         // borrows Environment
//!  │    │    └── CompiledModel<'env>       // borrows Environment AND Model
//!  │    │         └── BufferRequirements<'cm>  // borrows CompiledModel
//!  │    └── TensorBuffer<'env>  // borrows Environment
//!  └── Options<'lib>            // borrows Library
//! ```
//!
//! Two of these edges exist purely for soundness and are easy to get wrong when
//! calling the C API directly:
//!
//! - **`CompiledModel` borrows its `Model`.** `LiteRtCreateCompiledModel` does
//!   not take ownership; the compiled model reads the model's flatbuffer for
//!   the lifetime of every inference. Destroying the model first segfaults.
//! - **`BufferRequirements` borrows its `CompiledModel`.** The C API documents
//!   the returned requirements as *"still owned by the `LiteRtCompiledModel`
//!   and only valid during \[its\] lifetime"*.
//!
//! # Quick start
//!
//! ```no_run
//! use edgefirst_tflite::{litert, Library};
//!
//! let lib = Library::new()?;
//! let env = litert::Environment::new(&lib)?;
//! let model = litert::Model::from_file(&env, "model.tflite")?;
//! let opts = litert::Options::new(&lib)?
//!     .hardware_accelerators(litert::HwAccelerators::CPU)?;
//! let mut compiled = litert::CompiledModel::create(&env, &model, &opts)?;
//!
//! // Allocate buffers that satisfy the compiled model's requirements.
//! let mut inputs = vec![compiled.create_input_buffer(0, 0)?];
//! let mut outputs = vec![compiled.create_output_buffer(0, 0)?];
//!
//! let input_size = inputs[0].size();
//! inputs[0].write_bytes(&vec![0u8; input_size])?;
//! compiled.run(0, &mut inputs, &mut outputs)?;
//! let result = outputs[0].read_bytes()?;
//!
//! println!("fully accelerated: {}", compiled.is_fully_accelerated()?);
//! # Ok::<(), edgefirst_tflite::Error>(())
//! ```
//!
//! # Detecting availability
//!
//! Branch on [`Library::has_litert`](crate::Library::has_litert) when you want
//! to fall back to the classic path, rather than attempting construction and
//! inspecting the error:
//!
//! ```no_run
//! use edgefirst_tflite::{litert, Library};
//!
//! let lib = Library::new()?;
//! if lib.has_litert() {
//!     let env = litert::Environment::new(&lib)?;
//!     for accel in litert::accelerators(&env)? {
//!         println!("{}: {}", accel.id, accel.name);
//!     }
//! } else {
//!     // Classic TFLite: use Interpreter instead.
//!     println!("LiteRT unavailable: {:?}", lib.litert_missing_symbol());
//! }
//! # Ok::<(), edgefirst_tflite::Error>(())
//! ```
//!
//! # Thread safety
//!
//! `LiteRT` does not document the thread-safety of its C API, so this module
//! takes the conservative position: each type documents exactly what it
//! guarantees under a `# Thread safety` heading, and inference is serialised by
//! `&mut self` on [`CompiledModel::run`]. See the individual types for detail.

mod accelerator;
mod compiled_model;
mod environment;
mod model;
mod options;
mod tensor_buffer;

pub use accelerator::{accelerators, AcceleratorInfo};
pub use compiled_model::CompiledModel;
pub use environment::Environment;
pub use model::Model;
pub use options::{HwAccelerators, Options};
pub use tensor_buffer::{BufferRequirements, TensorBuffer, TensorBufferLock};

/// Element type and layout of a `LiteRT` tensor, as reported by the C API.
///
/// Obtained from [`Model::input_tensor_type`], [`Model::output_tensor_type`],
/// or [`TensorBuffer::tensor_type`], and consumed when allocating buffers.
pub use edgefirst_tflite_sys::litert::RankedTensorType;

use edgefirst_tflite_sys::litert::LiteRtFunctions;

use crate::error::{Error, Result};
use crate::Library;

/// Resolve the `LiteRT` function table or produce a descriptive error.
///
/// Every public constructor in this module funnels through here so that the
/// "not a `LiteRT` build" diagnosis is worded identically everywhere and always
/// names the missing symbol.
pub(crate) fn require_litert(lib: &Library) -> Result<&LiteRtFunctions> {
    lib.litert_result().map_err(Error::litert_unavailable)
}
