// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Library search and version probing for the `TFLite` C shared library.

use log::debug;
use std::path::Path;

use crate::tensorflowlite_c;

/// Default path for the `TFLite` C API shared library.
pub const DEFAULT_TFLITEC_PATH: &str = "libtensorflowlite_c.so";

/// Default path for the `TFLite` C++ shared library (wraps C API).
pub const DEFAULT_TFLITECPP_PATH: &str = "libtensorflow-lite.so";

/// Discover and load the `TFLite` shared library using version probing.
///
/// Tries versioned `libtensorflow-lite.so.2.{49..1}.{9..0}` first, then
/// falls back to `libtensorflowlite_c.so` and `libtensorflow-lite.so`.
///
/// # Performance
///
/// Probes up to ~500 versioned library paths before falling back to
/// unversioned names. On an i.MX 8M Plus EVK this completes in ~25 ms.
/// Use [`load`] directly if you know the exact library path.
///
/// # Errors
///
/// Returns a [`libloading::Error`] if no `TFLite` library can be found.
pub fn discover() -> Result<tensorflowlite_c, libloading::Error> {
    // Try versioned shared libraries first.
    // Takes around 25ms to try ~500 paths on the EVK.
    for version in (1..50).rev() {
        for patch in (0..10).rev() {
            let path = format!("{DEFAULT_TFLITECPP_PATH}.2.{version}.{patch}");
            if let Ok(lib) = load(&path) {
                debug!("Found TFLite library: {path}");
                return Ok(lib);
            }
        }
    }

    // Try unversioned paths.
    if let Ok(lib) = load(DEFAULT_TFLITEC_PATH) {
        debug!("Found TFLite library: {DEFAULT_TFLITEC_PATH}");
        return Ok(lib);
    }

    debug!("Trying fallback: {DEFAULT_TFLITECPP_PATH}");
    load(DEFAULT_TFLITECPP_PATH)
}

/// Load the `TFLite` shared library from a specific path.
///
/// # Errors
///
/// Returns a [`libloading::Error`] if the library cannot be loaded or
/// required symbols are missing.
pub fn load(path: impl AsRef<Path>) -> Result<tensorflowlite_c, libloading::Error> {
    // SAFETY: `tensorflowlite_c::new` loads a shared library and resolves all
    // TFLite C API symbols. The library remains loaded for the lifetime of the
    // returned struct. Callers must ensure the path points to a valid TFLite
    // shared library.
    unsafe { tensorflowlite_c::new(path.as_ref().as_os_str()) }
}
