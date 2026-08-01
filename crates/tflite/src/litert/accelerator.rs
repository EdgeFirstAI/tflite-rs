// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! `LiteRT` accelerator registry enumeration.

use std::ffi::CStr;
use std::fmt;

use edgefirst_tflite_sys::litert::{Accelerator, AcceleratorId, HwAcceleratorSet, ParamIndex};

use crate::error::{litert_status_to_result, Error, Result};
use crate::litert::options::HwAccelerators;
use crate::litert::Environment;

/// Upper bound on the accelerator count accepted from the C API.
///
/// The registry holds a handful of entries in practice (one per available
/// backend). A wildly larger value means the out-parameter was not written, and
/// pre-allocating from it would be an unbounded allocation driven by
/// uninitialised memory.
const MAX_PLAUSIBLE_ACCELERATORS: usize = 1024;

/// A hardware accelerator registered with a [`Environment`].
///
/// Enumerating these tells you what the runtime *could* dispatch to on this
/// device, which is the reliable way to check for NPU or GPU support — plugin
/// libraries are loaded at environment creation, so a backend that failed to
/// load simply will not appear.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceleratorInfo {
    /// Runtime-assigned identifier, stable within one [`Environment`].
    pub id: AcceleratorId,
    /// Human-readable name, e.g. `"CpuAccelerator"`.
    ///
    /// Empty if the runtime reported a null name. Non-UTF-8 bytes are replaced
    /// rather than rejected.
    pub name: String,
    /// The hardware classes this accelerator can serve.
    pub hardware: HwAccelerators,
}

impl fmt::Display for AcceleratorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}] (id {})", self.name, self.hardware, self.id)
    }
}

/// Enumerate the accelerators registered with `env`.
///
/// The list is populated when the [`Environment`] is created: the CPU
/// accelerator is always present, and GPU/NPU entries appear only if their
/// plugin libraries loaded successfully. An empty or CPU-only list on hardware
/// you expect to be accelerated usually means a missing or misplaced
/// accelerator `.so`.
///
/// # Errors
///
/// A `LiteRT` status error if the registry cannot be walked, or an
/// invalid-argument error if the runtime reports an implausible accelerator
/// count (currently more than 1024).
///
/// # Examples
///
/// ```no_run
/// use edgefirst_tflite::{litert, Library};
///
/// let lib = Library::new()?;
/// let env = litert::Environment::new(&lib)?;
///
/// let accelerators = litert::accelerators(&env)?;
/// let has_npu = accelerators
///     .iter()
///     .any(|a| a.hardware.contains(litert::HwAccelerators::NPU));
///
/// for accel in &accelerators {
///     println!("{accel}");
/// }
/// println!("NPU available: {has_npu}");
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
pub fn accelerators(env: &Environment<'_>) -> Result<Vec<AcceleratorInfo>> {
    let fns = env.functions();
    let mut num: ParamIndex = 0;
    // SAFETY: `env` is live; `num` is a valid out-parameter.
    let status = unsafe { (fns.get_num_accelerators)(env.as_raw(), &raw mut num) };
    litert_status_to_result(status).map_err(|e| e.with_context("LiteRtGetNumAccelerators"))?;

    if num > MAX_PLAUSIBLE_ACCELERATORS {
        return Err(Error::invalid_argument(format!(
            "LiteRtGetNumAccelerators reported {num} accelerators, above the \
             plausible maximum of {MAX_PLAUSIBLE_ACCELERATORS}"
        )));
    }

    let mut out = Vec::with_capacity(num);
    for index in 0..num {
        let mut accel: Accelerator = std::ptr::null_mut();
        // SAFETY: `index < num`, the count the runtime just reported.
        let status = unsafe { (fns.get_accelerator)(env.as_raw(), index, &raw mut accel) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtGetAccelerator"))?;
        if accel.is_null() {
            return Err(Error::null_pointer(format!(
                "LiteRtGetAccelerator returned null for index {index}"
            )));
        }

        let mut id: AcceleratorId = 0;
        // SAFETY: `accel` is a live, non-null accelerator handle.
        let status = unsafe { (fns.get_accelerator_id)(accel, &raw mut id) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtGetAcceleratorId"))?;

        let mut name_ptr: *const std::os::raw::c_char = std::ptr::null();
        // SAFETY: as above; `name_ptr` is a valid out-parameter.
        let status = unsafe { (fns.get_accelerator_name)(accel, &raw mut name_ptr) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtGetAcceleratorName"))?;
        let name = if name_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: the runtime owns this NUL-terminated string and keeps it
            // alive for the accelerator's lifetime, which outlives `env`. It is
            // copied into an owned `String` before this borrow ends.
            unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned()
        };

        let mut hw: HwAcceleratorSet = 0;
        // SAFETY: `accel` is live; `hw` is a valid out-parameter.
        let status = unsafe { (fns.get_accelerator_hardware_support)(accel, &raw mut hw) };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetAcceleratorHardwareSupport"))?;

        out.push(AcceleratorInfo {
            id,
            name,
            hardware: HwAccelerators::from_raw(hw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerator_info_display() {
        let info = AcceleratorInfo {
            id: 0,
            name: "CpuAccelerator".to_string(),
            hardware: HwAccelerators::CPU,
        };
        assert_eq!(info.to_string(), "CpuAccelerator [CPU] (id 0)");
    }
}
