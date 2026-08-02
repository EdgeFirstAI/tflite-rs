// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! `LiteRT` environment handle.

use std::ptr::null;

use edgefirst_tflite_sys::litert::{Environment as RawEnv, LiteRtFunctions};

use crate::error::{litert_status_to_result, Error, Result};
use crate::litert::require_litert;
use crate::Library;

/// The `LiteRT` runtime environment — the root of every other `LiteRT` handle.
///
/// Creating an `Environment` initialises the runtime and populates its
/// accelerator registry: the CPU accelerator is always registered, and GPU/NPU
/// accelerators are registered here if their plugin libraries can be loaded.
/// Everything else in this module is created from an `Environment` and borrows
/// it, so it must be the longest-lived `LiteRT` object you hold.
///
/// One environment per process is the intended usage; creating several is
/// legal but repeats plugin discovery.
///
/// # Examples
///
/// ```no_run
/// use edgefirst_tflite::{litert, Library};
///
/// let lib = Library::new()?;
/// let env = litert::Environment::new(&lib)?;
///
/// for accel in litert::accelerators(&env)? {
///     println!("{} (hw: {})", accel.name, accel.hardware);
/// }
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
///
/// # Thread safety
///
/// `Environment` is [`Send`] and [`Sync`]. Every method exposed here is a
/// read: the accelerator registry is populated during
/// `LiteRtCreateEnvironment` and the getters used by [`crate::litert`] do not
/// mutate it.
///
/// Note that `Sync` is load-bearing for the rest of the module —
/// [`Model`](crate::litert::Model) and
/// [`CompiledModel`](crate::litert::CompiledModel) hold `&Environment` and are
/// `Send`, so sending one to a worker thread shares the environment across
/// threads. Upstream `LiteRT` does not *document* thread-safety guarantees; if
/// you allocate tensor buffers from several threads at once and observe
/// corruption, serialise access with a `Mutex` and please report it.
#[derive(Debug)]
pub struct Environment<'lib> {
    ptr: RawEnv,
    fns: &'lib LiteRtFunctions,
}

impl<'lib> Environment<'lib> {
    /// Create a `LiteRT` environment with default options.
    ///
    /// # Errors
    ///
    /// - [`Error::is_litert_unavailable`] when `lib` does not export the
    ///   required `LiteRt*` symbols. This is the expected result on a classic
    ///   TensorFlow Lite library; [`Error::litert_missing_symbol`] says which
    ///   symbol was absent.
    /// - A `LiteRT` status error ([`Error::litert_status_code`]) if the runtime
    ///   fails to initialise.
    pub fn new(lib: &'lib Library) -> Result<Self> {
        let fns = require_litert(lib)?;
        let mut env: RawEnv = std::ptr::null_mut();
        // SAFETY: `create_environment` accepts a zero-length option array with
        // a null pointer, and writes an owned handle through `env`.
        let status = unsafe { (fns.create_environment)(0, null(), &raw mut env) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtCreateEnvironment"))?;
        if env.is_null() {
            return Err(Error::null_pointer("LiteRtCreateEnvironment returned null"));
        }
        Ok(Self { ptr: env, fns })
    }

    pub(crate) fn as_raw(&self) -> RawEnv {
        self.ptr
    }

    /// The `LiteRT` function table backing this environment.
    ///
    /// Returned with the lifetime of `&self` (not `'lib`) so that borrows
    /// derived from an `Environment` are tied to the environment rather than to
    /// the underlying [`Library`], which is what gives the child types a single
    /// lifetime parameter.
    pub(crate) fn functions(&self) -> &LiteRtFunctions {
        self.fns
    }
}

// SAFETY: `Environment` owns a `LiteRtEnvironment`, an opaque heap handle with
// no thread affinity — it is not tied to a TLS slot, an OS thread, or a
// rendering context. Moving it between threads is therefore sound.
unsafe impl Send for Environment<'_> {}

// SAFETY: the only operations reachable through `&Environment` in this crate
// are `as_raw`/`functions` (field reads) and the accelerator registry getters
// in `litert::accelerators`, which read a registry that
// `LiteRtCreateEnvironment` fully populates before returning. No API here
// mutates the environment through a shared reference. See the type-level
// `# Thread safety` note for the caveat about undocumented upstream behaviour.
unsafe impl Sync for Environment<'_> {}

impl Drop for Environment<'_> {
    fn drop(&mut self) {
        // SAFETY: `ptr` was created by `LiteRtCreateEnvironment`, is non-null
        // (checked in `new`), and has not been destroyed — `Drop` runs once and
        // every child handle borrows `self`, so all of them are already gone.
        unsafe {
            (self.fns.destroy_environment)(self.ptr);
        }
    }
}
