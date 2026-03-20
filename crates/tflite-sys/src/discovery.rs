// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Library search and version probing for the `TFLite` C shared library.
//!
//! # Discovery Order
//!
//! 1. `TFLITE_LIBRARY_PATH` environment variable (explicit runtime override)
//! 2. Vendored library from `build.rs` (when `vendored` feature is enabled)
//! 3. Versioned system paths: `libtensorflow-lite.so.2.{49..1}.{9..0}`
//! 4. Unversioned fallbacks (platform-specific)
//!
//! # Debugging
//!
//! Set `RUST_LOG=edgefirst_tflite_sys=debug` to see which paths are probed
//! during discovery.

use log::{debug, warn};
use std::path::Path;

use crate::tensorflowlite_c;

/// Platform-specific default path for the `TFLite` C API shared library.
pub const DEFAULT_TFLITEC_PATH: &str = if cfg!(windows) {
    "tensorflowlite_c.dll"
} else if cfg!(target_os = "macos") {
    "libtensorflowlite_c.dylib"
} else {
    "libtensorflowlite_c.so"
};

/// Platform-specific default path for the `TFLite` C++ shared library
/// (wraps C API).
pub const DEFAULT_TFLITECPP_PATH: &str = if cfg!(windows) {
    "tensorflow-lite.dll"
} else if cfg!(target_os = "macos") {
    "libtensorflow-lite.dylib"
} else {
    "libtensorflow-lite.so"
};

/// Discover and load the `TFLite` shared library.
///
/// Searches for the library in priority order:
///
/// 1. **`TFLITE_LIBRARY_PATH`** — explicit runtime override via environment
///    variable. Set this to bypass all other discovery.
/// 2. **Vendored** — library downloaded by `build.rs` when the `vendored`
///    Cargo feature is enabled. The path is baked in at compile time via
///    `option_env!("EDGEFIRST_TFLITE_VENDORED_DIR")`.
/// 3. **Versioned system paths** — probes
///    `libtensorflow-lite.so.2.{49..1}.{9..0}` (Linux only).
/// 4. **Unversioned fallbacks** — `libtensorflowlite_c.{so,dylib,dll}` then
///    `libtensorflow-lite.{so,dylib,dll}`.
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
    // 1. Explicit override via environment variable.
    if let Ok(path) = std::env::var("TFLITE_LIBRARY_PATH") {
        debug!("TFLITE_LIBRARY_PATH={path}");
        return load(&path);
    }

    // 2. Vendored library from build.rs (compile-time path).
    if let Some(lib) = try_vendored() {
        return Ok(lib);
    }

    // 3. Try versioned shared libraries (Linux only — macOS/Windows don't use
    //    versioned .so symlinks).
    if cfg!(target_os = "linux") {
        for version in (1..50).rev() {
            for patch in (0..10).rev() {
                let path = format!("{DEFAULT_TFLITECPP_PATH}.2.{version}.{patch}");
                if let Ok(lib) = load(&path) {
                    debug!("Found TFLite library: {path}");
                    return Ok(lib);
                }
            }
        }
    }

    // 4. Try unversioned platform-specific paths.
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

/// Try loading the vendored library placed by `build.rs`.
///
/// Returns `None` when:
/// - The `vendored` feature was not enabled at compile time
/// - The vendored library file doesn't exist (stale `OUT_DIR` or deployed
///   binary without the library alongside it)
/// - The library fails to load (ABI mismatch, missing transitive deps, etc.)
fn try_vendored() -> Option<tensorflowlite_c> {
    let dir = option_env!("EDGEFIRST_TFLITE_VENDORED_DIR")?;

    let lib_name = if cfg!(windows) {
        "tensorflowlite_c.dll"
    } else if cfg!(target_os = "macos") {
        "libtensorflowlite_c.dylib"
    } else {
        "libtensorflowlite_c.so"
    };

    let path = std::path::Path::new(dir).join(lib_name);

    if !path.exists() {
        warn!(
            "Vendored TFLite library not found at {} \
             (built with `vendored` feature but library is missing at runtime — \
             copy the library alongside your binary for deployment)",
            path.display()
        );
        return None;
    }

    debug!("Trying vendored library: {}", path.display());
    match load(&path) {
        Ok(lib) => {
            debug!("Loaded vendored TFLite library: {}", path.display());
            Some(lib)
        }
        Err(e) => {
            warn!(
                "Vendored TFLite library at {} failed to load: {e}",
                path.display()
            );
            None
        }
    }
}
