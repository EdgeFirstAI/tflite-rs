// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! FFI bindings for the XNNPACK delegate API.
//!
//! XNNPACK is a built-in `TFLite` delegate — its symbols live in the main
//! `libtensorflowlite_c.so` (when compiled with `-DTFLITE_ENABLE_XNNPACK=ON`),
//! not in a separate delegate `.so`.
//!
//! These are loaded at runtime from the main `TFLite` library using
//! `libloading`. The C API is defined in `xnnpack_delegate.h`.

use std::ffi::c_int;

use crate::TfLiteDelegate;

/// Options for configuring the XNNPACK delegate.
///
/// Maps to `TfLiteXNNPackDelegateOptions` from `xnnpack_delegate.h`.
///
/// This struct intentionally does **not** implement `Default` because
/// zero-initialisation may diverge from the C library's defaults in
/// future `TFLite` versions. Always initialise via
/// [`XnnPackFunctions::options_default`], then override individual fields.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TfLiteXNNPackDelegateOptions {
    /// Number of threads for the XNNPACK threadpool.
    ///
    /// A value of 0 or negative lets XNNPACK choose based on the platform.
    pub num_threads: c_int,

    /// Bitmask of `TfLiteXNNPackDelegateFlags` (0 = default behaviour).
    pub flags: u32,
}

/// Function pointers for the XNNPACK delegate API.
///
/// Loaded at runtime from the main `TFLite` shared library. Use
/// [`XnnPackFunctions::try_load`] to attempt loading; returns `None`
/// when the library was not compiled with XNNPACK support.
#[derive(Debug)]
pub struct XnnPackFunctions {
    /// `TfLiteXNNPackDelegateOptionsDefault`
    pub options_default: unsafe extern "C" fn() -> TfLiteXNNPackDelegateOptions,

    /// `TfLiteXNNPackDelegateCreate`
    pub create: unsafe extern "C" fn(*const TfLiteXNNPackDelegateOptions) -> *mut TfLiteDelegate,

    /// `TfLiteXNNPackDelegateDelete`
    pub delete: unsafe extern "C" fn(*mut TfLiteDelegate),
}

impl XnnPackFunctions {
    /// Attempt to load all XNNPACK function pointers from a library.
    ///
    /// Returns `None` if any required symbol is missing. This is expected
    /// when the `TFLite` library was not compiled with XNNPACK support.
    ///
    /// # Safety
    ///
    /// The caller must ensure `lib` remains loaded for the lifetime of the
    /// returned struct, because the function pointers point into the
    /// library's code segment.
    #[must_use]
    pub unsafe fn try_load(lib: &libloading::Library) -> Option<Self> {
        // SAFETY: Each `lib.get` resolves a symbol from the loaded library.
        // The `.ok()?` short-circuits on any missing symbol, returning `None`.
        unsafe {
            Some(Self {
                options_default: *lib.get(b"TfLiteXNNPackDelegateOptionsDefault\0").ok()?,
                create: *lib.get(b"TfLiteXNNPackDelegateCreate\0").ok()?,
                delete: *lib.get(b"TfLiteXNNPackDelegateDelete\0").ok()?,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_clone_copy() {
        let opts = TfLiteXNNPackDelegateOptions {
            num_threads: 4,
            flags: 1,
        };
        let copied = opts;
        assert_eq!(copied.num_threads, 4);
        assert_eq!(copied.flags, 1);
    }

    #[test]
    fn options_debug() {
        let opts = TfLiteXNNPackDelegateOptions {
            num_threads: 2,
            flags: 0,
        };
        let debug = format!("{opts:?}");
        assert!(debug.contains("num_threads"));
        assert!(debug.contains('2'));
    }
}
