// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Build script for `edgefirst-tflite-sys`.
//!
//! When the `vendored` feature is enabled, downloads a pre-built
//! `libtensorflowlite_c` shared library from the `tflite-rs` GitHub Releases
//! and places it in `OUT_DIR` for runtime discovery.

#[cfg(feature = "vendored")]
mod vendored;

fn main() {
    #[cfg(feature = "vendored")]
    vendored::download_and_install();
}
