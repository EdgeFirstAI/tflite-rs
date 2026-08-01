// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! FFI bindings for the experimental `TFLite` C API (`c_api_experimental.h`).
//!
//! These symbols live in the main `TFLite` shared library but are *not* part
//! of the bindgen surface, and deliberately so: the generated
//! `tensorflowlite_c` loader resolves every symbol eagerly, so one
//! experimental symbol missing from an older runtime would make the entire
//! library fail to load. They are resolved individually here instead, the same
//! way [`crate::xnnpack_ffi`] and [`crate::litert`] treat optional symbols.
//!
//! Upstream marks this API "experimental and subject to change". The two
//! symbols bound here have been stable since TF 2.5, but a signature change on
//! a future runtime would not be caught at compile time the way the
//! bindgen-backed tables are — see `crates/tflite-sys/update.sh`.

use std::ffi::c_void;

use crate::{TfLiteInterpreter, TfLiteStatus};

/// A caller-provided buffer that backs a tensor in place of the arena.
///
/// Maps to `TfLiteCustomAllocation` from `common.h`. The runtime does **not**
/// take ownership: `data` must remain valid, at the same address, for as long
/// as the interpreter can read the tensor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TfLiteCustomAllocation {
    /// Base address of the buffer.
    pub data: *mut c_void,
    /// Size of the buffer in bytes. Must be at least the tensor's byte size.
    pub bytes: usize,
}

/// Default behaviour: the runtime verifies the buffer's alignment.
#[allow(non_upper_case_globals)]
pub const kTfLiteCustomAllocationFlagsNone: i64 = 0;

/// Skip the [`TENSOR_ALIGNMENT`] check.
///
/// Upstream's own warning: "Setting this flag can cause crashes when calling
/// `Invoke()`. Use with caution." Kernels may issue aligned vector loads against
/// the buffer regardless of what the runtime was told.
#[allow(non_upper_case_globals)]
pub const kTfLiteCustomAllocationFlagsSkipAlignCheck: i64 = 1;

/// `kDefaultTensorAlignment` from `lite/util.h` — the alignment a custom
/// allocation must satisfy unless
/// [`kTfLiteCustomAllocationFlagsSkipAlignCheck`] is set.
pub const TENSOR_ALIGNMENT: usize = 64;

/// Function pointers for the experimental `TFLite` C API.
///
/// Loaded at runtime from the main `TFLite` shared library. Use
/// [`ExperimentalFunctions::try_load`]; it returns `None` on a runtime that
/// does not export the experimental surface.
#[derive(Debug, Clone, Copy)]
pub struct ExperimentalFunctions {
    /// `TfLiteInterpreterSetCustomAllocationForTensor`
    pub set_custom_allocation_for_tensor: unsafe extern "C" fn(
        *mut TfLiteInterpreter,
        std::ffi::c_int,
        *const TfLiteCustomAllocation,
        i64,
    ) -> TfLiteStatus,

    /// `TfLiteInterpreterGetInputTensorIndex` — maps an *input* index to the
    /// graph-wide tensor index that the setter above expects.
    pub get_input_tensor_index: unsafe extern "C" fn(*const TfLiteInterpreter, i32) -> i32,
}

impl ExperimentalFunctions {
    /// The symbols this table requires, in resolution order.
    #[must_use]
    pub const fn required_symbols() -> &'static [&'static str] {
        &[
            "TfLiteInterpreterSetCustomAllocationForTensor",
            "TfLiteInterpreterGetInputTensorIndex",
        ]
    }

    /// Attempt to resolve the experimental function pointers from `lib`.
    ///
    /// Returns `None` if any symbol is missing, which is the expected result
    /// on a `TFLite` build compiled without the experimental C API.
    ///
    /// # Safety
    ///
    /// The caller must ensure `lib` remains loaded for the lifetime of the
    /// returned struct, because the function pointers point into the library's
    /// code segment.
    #[must_use]
    pub unsafe fn try_load(lib: &libloading::Library) -> Option<Self> {
        // SAFETY: each `lib.get` resolves a symbol from the loaded library;
        // `.ok()?` short-circuits to `None` on the first missing one.
        unsafe {
            Some(Self {
                set_custom_allocation_for_tensor: *lib
                    .get(b"TfLiteInterpreterSetCustomAllocationForTensor\0")
                    .ok()?,
                get_input_tensor_index: *lib.get(b"TfLiteInterpreterGetInputTensorIndex\0").ok()?,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_allocation_matches_c_layout() {
        // `struct { void* data; size_t bytes; }` — two pointer-sized fields.
        assert_eq!(
            std::mem::size_of::<TfLiteCustomAllocation>(),
            2 * std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::align_of::<TfLiteCustomAllocation>(),
            std::mem::align_of::<*mut c_void>()
        );
    }

    #[test]
    fn flag_values_match_the_c_enum() {
        assert_eq!(kTfLiteCustomAllocationFlagsNone, 0);
        assert_eq!(kTfLiteCustomAllocationFlagsSkipAlignCheck, 1);
    }

    #[test]
    fn tensor_alignment_is_a_power_of_two() {
        assert_eq!(TENSOR_ALIGNMENT, 64);
        assert!(TENSOR_ALIGNMENT.is_power_of_two());
    }

    #[test]
    fn required_symbols_are_tflite_prefixed() {
        let symbols = ExperimentalFunctions::required_symbols();
        assert_eq!(symbols.len(), 2);
        assert!(symbols.iter().all(|s| s.starts_with("TfLite")));
    }
}
