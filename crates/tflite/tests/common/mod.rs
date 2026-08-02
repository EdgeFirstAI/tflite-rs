// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Shared test helpers for integration tests.
//!
//! Provides a minimal embedded model, library discovery helpers, and a
//! `require_tflite!()` macro that skips tests when `TFLite` is not available.

use edgefirst_tflite::{Interpreter, Library, Model};

/// Minimal `TFLite` model: float32[1,4] input -> Add constant -> float32[1,4] output.
pub const MINIMAL_MODEL: &[u8] = include_bytes!("../../../../testdata/minimal.tflite");

/// Returns `true` if a `TFLite` shared library is available for testing.
///
/// Checks the `TFLITE_TEST_LIB` environment variable first, then falls back
/// to auto-discovery via `Library::new()`.
pub fn tflite_available() -> bool {
    load_library().is_some()
}

/// Skip the calling test with a message if `TFLite` is not available.
macro_rules! require_tflite {
    () => {
        if !$crate::common::tflite_available() {
            eprintln!(
                "SKIPPED: TFLite shared library not available. \
                 Set TFLITE_TEST_LIB=/path/to/libtensorflowlite_c.so to enable."
            );
            return;
        }
    };
}
pub(crate) use require_tflite;

/// Skip the calling test unless `LiteRT` (`LiteRt*`) symbols are present.
///
/// Reports the unresolved symbol so a skip caused by a *partial* `LiteRT`
/// build is not mistaken for "this is a classic `TFLite` library".
// This module is shared by several integration test binaries, not all of which
// exercise LiteRT; the macro is legitimately unused in those.
#[allow(unused_macros)]
macro_rules! require_litert {
    () => {
        $crate::common::require_tflite!();
        let Some(__lib) = $crate::common::load_library() else {
            eprintln!(
                "SKIPPED: TFLite shared library became unavailable after the \
                 availability check. Set TFLITE_TEST_LIB to a stable path."
            );
            return;
        };
        if !__lib.has_litert() {
            eprintln!(
                "SKIPPED: LiteRT unavailable ({:?} unresolved). Point TFLITE_TEST_LIB \
                 at a LiteRT-capable shared library (e.g. Android libLiteRt.so) to enable.",
                __lib.litert_missing_symbol()
            );
            return;
        }
    };
}
#[allow(unused_imports)]
pub(crate) use require_litert;

/// Load the `TFLite` library from `TFLITE_TEST_LIB` env var or auto-discovery.
pub fn load_library() -> Option<Library> {
    if let Ok(path) = std::env::var("TFLITE_TEST_LIB") {
        Library::from_path(path).ok()
    } else {
        Library::new().ok()
    }
}

/// Load the minimal test model.
pub fn load_model(lib: &Library) -> Model<'_> {
    Model::from_bytes(lib, MINIMAL_MODEL).expect("failed to load minimal test model")
}

/// Build an interpreter for the minimal model with 1 thread.
#[allow(dead_code)]
pub fn build_interpreter<'lib>(lib: &'lib Library, model: &Model<'lib>) -> Interpreter<'lib> {
    Interpreter::builder(lib)
        .expect("failed to create interpreter builder")
        .num_threads(1)
        .build(model)
        .expect("failed to build interpreter")
}
