// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.

//! Shared test helpers.

use edgefirst_tflite::Library;

/// Report a skipped test, or fail it when `TFLITE_TEST_NO_SKIP` is set to
/// anything but `0`. Board CI sets it: the board runtime must have everything
/// these tests need, and a skip there would pass without testing anything.
pub fn skip(reason: impl std::fmt::Display) {
    let no_skip = std::env::var_os("TFLITE_TEST_NO_SKIP").is_some_and(|v| v != "0");
    assert!(
        !no_skip,
        "test would skip, but TFLITE_TEST_NO_SKIP is set: {reason}"
    );
    eprintln!("SKIPPED: {reason}");
}

/// The `TFLite` library from `TFLITE_TEST_LIB` or auto-discovery, or `None`
/// after reporting a skip.
pub fn require_library() -> Option<Library> {
    let lib = match std::env::var("TFLITE_TEST_LIB") {
        Ok(path) => Library::from_path(path),
        Err(_) => Library::new(),
    };
    match lib {
        Ok(lib) => Some(lib),
        Err(e) => {
            skip(format_args!(
                "TFLite shared library not available ({e}). \
                 Set TFLITE_TEST_LIB=/path/to/libtensorflowlite_c.so to enable."
            ));
            None
        }
    }
}
