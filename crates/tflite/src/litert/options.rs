// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! `LiteRT` compilation options.

use std::fmt;

use edgefirst_tflite_sys::litert::{
    kLiteRtHwAcceleratorCpu, kLiteRtHwAcceleratorGpu, kLiteRtHwAcceleratorNpu, HwAcceleratorSet,
    LiteRtFunctions, Options as RawOptions,
};

use crate::error::{litert_status_to_result, Error, Result};
use crate::litert::require_litert;
use crate::Library;

/// A set of hardware accelerators, as a bitmask.
///
/// This is a *request*, not an assertion: asking for
/// [`NPU`](HwAccelerators::NPU) permits the runtime to dispatch to an NPU, but
/// it will silently fall back for any operator the NPU cannot execute. Use
/// [`CompiledModel::is_fully_accelerated`](crate::litert::CompiledModel::is_fully_accelerated)
/// after compiling to find out what actually happened.
///
/// Combine accelerators with `|`; the runtime picks among the permitted set.
///
/// # Examples
///
/// ```
/// use edgefirst_tflite::litert::HwAccelerators;
///
/// let npu_or_cpu = HwAccelerators::NPU | HwAccelerators::CPU;
/// assert!(npu_or_cpu.contains(HwAccelerators::NPU));
/// assert!(!npu_or_cpu.contains(HwAccelerators::GPU));
/// assert_eq!(npu_or_cpu.to_string(), "CPU|NPU");
///
/// assert_eq!(HwAccelerators::NONE.to_string(), "none");
/// ```
///
/// # Note on `WebNn`
///
/// `LiteRT` defines a fourth accelerator bit, `kLiteRtHwAcceleratorWebNn`, but
/// guards it with `#if defined(__EMSCRIPTEN__)`. It is therefore absent from
/// the generated bindings on every target this crate supports and has no
/// constant here. [`Display`](fmt::Display) still renders any unrecognised bit
/// as `unknown(0x…)` rather than dropping it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct HwAccelerators(HwAcceleratorSet);

impl HwAccelerators {
    /// The empty set — no accelerator requested.
    pub const NONE: Self = Self(0);
    /// Run on the host CPU (with XNNPACK where available).
    pub const CPU: Self = Self(kLiteRtHwAcceleratorCpu);
    /// Permit dispatch to the GPU.
    pub const GPU: Self = Self(kLiteRtHwAcceleratorGpu);
    /// Permit dispatch to an NPU / dedicated ML accelerator.
    pub const NPU: Self = Self(kLiteRtHwAcceleratorNpu);

    /// Union of two accelerator sets.
    ///
    /// Available in `const` context; `|` ([`std::ops::BitOr`]) is the usual
    /// spelling elsewhere.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns `true` if every accelerator in `other` is present in `self`.
    ///
    /// `x.contains(HwAccelerators::NONE)` is always `true`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns `true` if no accelerator is selected.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The raw `LiteRtHwAcceleratorSet` bitmask.
    #[must_use]
    pub const fn bits(self) -> HwAcceleratorSet {
        self.0
    }

    /// Wrap a raw bitmask received from the C API.
    ///
    /// Unrecognised bits are preserved rather than masked off, so a value
    /// round-trips through [`HwAccelerators::bits`] unchanged even on a runtime
    /// that defines accelerators these bindings do not know about.
    pub(crate) const fn from_raw(raw: HwAcceleratorSet) -> Self {
        Self(raw)
    }
}

impl fmt::Display for HwAccelerators {
    /// Renders as `CPU`, `CPU|NPU`, `none`, or — for bits these bindings do not
    /// recognise — `unknown(0x8)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        let mut remaining = self.0;
        let mut first = true;
        for (bit, name) in [
            (kLiteRtHwAcceleratorCpu, "CPU"),
            (kLiteRtHwAcceleratorGpu, "GPU"),
            (kLiteRtHwAcceleratorNpu, "NPU"),
        ] {
            if remaining & bit != 0 {
                if !first {
                    f.write_str("|")?;
                }
                f.write_str(name)?;
                first = false;
                remaining &= !bit;
            }
        }
        if remaining != 0 {
            if !first {
                f.write_str("|")?;
            }
            write!(f, "unknown({remaining:#x})")?;
        }
        Ok(())
    }
}

impl std::ops::BitOr for HwAccelerators {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for HwAccelerators {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

/// Compilation options for [`CompiledModel`](crate::litert::CompiledModel).
///
/// Built with a consuming builder so that a configured `Options` can be reused
/// across several [`CompiledModel::create`](crate::litert::CompiledModel::create)
/// calls:
///
/// ```no_run
/// use edgefirst_tflite::{litert, Library};
///
/// let lib = Library::new()?;
/// let opts = litert::Options::new(&lib)?
///     .hardware_accelerators(litert::HwAccelerators::NPU | litert::HwAccelerators::CPU)?;
///
/// assert!(opts.hardware_accelerator_set()?.contains(litert::HwAccelerators::NPU));
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
///
/// # Lifetime
///
/// `Options<'lib>` borrows the [`Library`] rather than an
/// [`Environment`](crate::litert::Environment): options are inert configuration
/// and are not bound to a particular runtime instance.
///
/// # Thread safety
///
/// `Options` is [`Send`] and [`Sync`]. Mutation happens only through
/// [`Options::hardware_accelerators`], which consumes `self`, so no C call can
/// mutate the options through a shared reference.
#[derive(Debug)]
pub struct Options<'lib> {
    ptr: RawOptions,
    fns: &'lib LiteRtFunctions,
}

impl<'lib> Options<'lib> {
    /// Create compilation options with `LiteRT`'s defaults.
    ///
    /// The default accelerator set is **empty**
    /// ([`HwAccelerators::NONE`]). Compiling with an empty set is rejected, so
    /// call [`Options::hardware_accelerators`] before passing these to
    /// [`CompiledModel::create`](crate::litert::CompiledModel::create) —
    /// [`HwAccelerators::CPU`] is the always-available baseline.
    ///
    /// # Errors
    ///
    /// - [`Error::is_litert_unavailable`] when `lib` does not export the
    ///   `LiteRt*` symbols.
    /// - A `LiteRT` status error if allocation fails.
    pub fn new(lib: &'lib Library) -> Result<Self> {
        let fns = require_litert(lib)?;
        let mut opts: RawOptions = std::ptr::null_mut();
        // SAFETY: `create_options` writes a newly owned handle through `opts`.
        let status = unsafe { (fns.create_options)(&raw mut opts) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtCreateOptions"))?;
        if opts.is_null() {
            return Err(Error::null_pointer("LiteRtCreateOptions returned null"));
        }
        Ok(Self { ptr: opts, fns })
    }

    /// Select which hardware accelerators the runtime may dispatch to.
    ///
    /// Consumes and returns `self` so calls chain from [`Options::new`]. On
    /// error the options are dropped and destroyed.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if the bitmask is rejected.
    pub fn hardware_accelerators(self, accelerators: HwAccelerators) -> Result<Self> {
        // SAFETY: `self.ptr` is a valid handle from `create_options`.
        let status =
            unsafe { (self.fns.set_options_hardware_accelerators)(self.ptr, accelerators.bits()) };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtSetOptionsHardwareAccelerators"))?;
        Ok(self)
    }

    /// Read back the currently selected accelerators.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if the options cannot be queried.
    pub fn hardware_accelerator_set(&self) -> Result<HwAccelerators> {
        let mut raw: HwAcceleratorSet = 0;
        // SAFETY: `self.ptr` is a valid handle; `raw` is a valid out-parameter.
        let status =
            unsafe { (self.fns.get_options_hardware_accelerators)(self.ptr, &raw mut raw) };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetOptionsHardwareAccelerators"))?;
        Ok(HwAccelerators::from_raw(raw))
    }

    pub(crate) fn as_raw(&self) -> RawOptions {
        self.ptr
    }
}

// SAFETY: `Options` owns a `LiteRtOptions`, an opaque heap handle with no
// thread affinity, so moving it between threads is sound.
unsafe impl Send for Options<'_> {}

// SAFETY: the only C call reachable through `&Options` is
// `LiteRtGetOptionsHardwareAccelerators`, a read. The setter consumes `self`,
// so no mutation is possible through a shared reference.
unsafe impl Sync for Options<'_> {}

impl Drop for Options<'_> {
    fn drop(&mut self) {
        // SAFETY: `ptr` was created by `LiteRtCreateOptions`, is non-null
        // (checked in `new`), and is destroyed exactly once. The C API
        // documents compilation options as caller-owned and not retained by a
        // compiled model, so destroying them here is safe even while a
        // `CompiledModel` built with them is still alive.
        unsafe {
            (self.fns.destroy_options)(self.ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_and_contains() {
        let both = HwAccelerators::NPU | HwAccelerators::CPU;
        assert!(both.contains(HwAccelerators::NPU));
        assert!(both.contains(HwAccelerators::CPU));
        assert!(!both.contains(HwAccelerators::GPU));
        assert!(both.contains(HwAccelerators::NONE));
        assert!(!both.is_empty());
        assert!(HwAccelerators::NONE.is_empty());
        assert_eq!(HwAccelerators::default(), HwAccelerators::NONE);
    }

    #[test]
    fn bitor_assign_accumulates() {
        let mut set = HwAccelerators::CPU;
        set |= HwAccelerators::GPU;
        assert_eq!(set, HwAccelerators::CPU | HwAccelerators::GPU);
    }

    #[test]
    fn display_is_readable() {
        assert_eq!(HwAccelerators::NONE.to_string(), "none");
        assert_eq!(HwAccelerators::CPU.to_string(), "CPU");
        assert_eq!(
            (HwAccelerators::CPU | HwAccelerators::NPU).to_string(),
            "CPU|NPU"
        );
        assert_eq!(
            (HwAccelerators::CPU | HwAccelerators::GPU | HwAccelerators::NPU).to_string(),
            "CPU|GPU|NPU"
        );
    }

    #[test]
    fn display_preserves_unknown_bits() {
        // Bit 3 is `kLiteRtHwAcceleratorWebNn` on Emscripten builds and is not
        // exposed here; it must still be visible rather than silently dropped.
        let webnn_only = HwAccelerators::from_raw(1 << 3);
        assert_eq!(webnn_only.to_string(), "unknown(0x8)");
        let mixed = HwAccelerators::from_raw(kLiteRtHwAcceleratorCpu | (1 << 3));
        assert_eq!(mixed.to_string(), "CPU|unknown(0x8)");
    }

    #[test]
    fn raw_bits_round_trip_unknown_values() {
        let raw = kLiteRtHwAcceleratorNpu | (1 << 5);
        assert_eq!(HwAccelerators::from_raw(raw).bits(), raw);
    }
}
