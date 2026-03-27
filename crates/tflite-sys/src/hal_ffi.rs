// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! FFI bindings for the HAL Delegate DMA-BUF API.
//!
//! These types mirror the stable C ABI defined in `edgefirst-hal-capi`
//! (`hal.h`). The HAL owns the type definitions; function implementations
//! live in delegate shared libraries (e.g., Neutron NPU delegate).
//!
//! Function pointers are loaded at runtime from the delegate `.so` using
//! `libloading`, following the same pattern as [`super::vx_ffi`].

use std::ffi::{c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// HalDtype
// ---------------------------------------------------------------------------

/// Element data type for HAL tensors.
///
/// Mirrors `hal_dtype` from `edgefirst-hal-capi` (`hal.h`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HalDtype {
    /// Unsigned 8-bit integer.
    U8 = 0,
    /// Signed 8-bit integer.
    I8 = 1,
    /// Unsigned 16-bit integer.
    U16 = 2,
    /// Signed 16-bit integer.
    I16 = 3,
    /// Unsigned 32-bit integer.
    U32 = 4,
    /// Signed 32-bit integer.
    I32 = 5,
    /// Unsigned 64-bit integer.
    U64 = 6,
    /// Signed 64-bit integer.
    I64 = 7,
    /// 16-bit floating point (half).
    F16 = 8,
    /// 32-bit floating point (float).
    F32 = 9,
    /// 64-bit floating point (double).
    F64 = 10,
}

// ---------------------------------------------------------------------------
// HalDmabufTensorInfo
// ---------------------------------------------------------------------------

/// Maximum number of dimensions in a delegate tensor shape.
pub const HAL_DMABUF_MAX_NDIM: usize = 8;

/// DMA-BUF tensor information returned by a delegate.
///
/// Describes a single tensor's DMA-BUF allocation, including the file
/// descriptor, buffer geometry, and element type. The `fd` is borrowed
/// from the delegate and must **not** be closed by the caller.
///
/// Fields are ordered to eliminate padding on LP64: all `usize` fields
/// first (8-byte aligned), then smaller `c_int` and enum fields (4 bytes
/// each). Total size: 96 bytes on LP64.
///
/// Mirrors `hal_dmabuf_tensor_info` from `edgefirst-hal-capi` (`hal.h`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalDmabufTensorInfo {
    /// Buffer size in bytes.
    pub size: usize,
    /// Byte offset within the DMA-BUF.
    pub offset: usize,
    /// Tensor dimensions (up to [`HAL_DMABUF_MAX_NDIM`]).
    pub shape: [usize; 8],
    /// Number of valid entries in `shape`.
    pub ndim: usize,
    /// DMA-BUF file descriptor (borrowed — do **not** close).
    pub fd: c_int,
    /// Element data type.
    pub dtype: HalDtype,
}

impl Default for HalDmabufTensorInfo {
    fn default() -> Self {
        // SAFETY: HalDmabufTensorInfo is a #[repr(C)] plain-old-data struct.
        // The HAL ABI contract requires zero-initialization, so an all-zero
        // bit-pattern is a valid "empty" state.
        unsafe { std::mem::zeroed() }
    }
}

// Ensure the shape array length matches HAL_DMABUF_MAX_NDIM.
const _: () = assert!(
    std::mem::size_of::<[usize; 8]>() == std::mem::size_of::<[usize; HAL_DMABUF_MAX_NDIM]>()
);

// Compile-time layout assertion: 11 × usize + 1 × c_int + 1 × HalDtype
// = 11×8 + 4 + 4 = 96 bytes on LP64 with no internal padding.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<HalDmabufTensorInfo>() == 96);

// ---------------------------------------------------------------------------
// HalDmaBufFunctions
// ---------------------------------------------------------------------------

/// Function pointers for the HAL Delegate DMA-BUF API.
///
/// Loaded at runtime from the delegate shared library. Use
/// [`HalDmaBufFunctions::try_load`] to attempt loading.
///
/// The `hal_delegate_t` parameter (`*mut c_void`) is the `TfLiteDelegate*`
/// cast to an opaque handle.
#[derive(Debug)]
pub struct HalDmaBufFunctions {
    /// `hal_dmabuf_get_instance` — returns the delegate handle.
    pub get_instance: unsafe extern "C" fn() -> *mut c_void,
    /// `hal_dmabuf_is_supported` — returns 1 if supported, 0 otherwise.
    pub is_supported: unsafe extern "C" fn(*mut c_void) -> c_int,
    /// `hal_dmabuf_get_tensor_info` — fills `info` for the given tensor index.
    pub get_tensor_info:
        unsafe extern "C" fn(*mut c_void, c_int, *mut HalDmabufTensorInfo, usize) -> c_int,
    /// `hal_dmabuf_sync_for_device` — flush CPU caches for device access.
    pub sync_for_device: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
    /// `hal_dmabuf_sync_for_cpu` — invalidate caches for CPU access.
    pub sync_for_cpu: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
}

impl HalDmaBufFunctions {
    /// Attempt to load all HAL DMA-BUF function pointers from a delegate library.
    ///
    /// Returns `None` if any required symbol is missing.
    ///
    /// # Safety
    ///
    /// The library must remain loaded for the lifetime of this struct.
    #[must_use]
    pub unsafe fn try_load(lib: &libloading::Library) -> Option<Self> {
        unsafe {
            Some(Self {
                get_instance: *lib.get(b"hal_dmabuf_get_instance\0").ok()?,
                is_supported: *lib.get(b"hal_dmabuf_is_supported\0").ok()?,
                get_tensor_info: *lib.get(b"hal_dmabuf_get_tensor_info\0").ok()?,
                sync_for_device: *lib.get(b"hal_dmabuf_sync_for_device\0").ok()?,
                sync_for_cpu: *lib.get(b"hal_dmabuf_sync_for_cpu\0").ok()?,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// HalCameraAdaptorFormatInfo
// ---------------------------------------------------------------------------

/// Camera adaptor format information returned by a delegate.
///
/// Describes a camera format's channel layout and V4L2 `FourCC` code.
///
/// Mirrors `hal_camera_adaptor_format_info` from `edgefirst-hal-capi`
/// (`hal.h`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalCameraAdaptorFormatInfo {
    /// Number of input channels for this format.
    pub input_channels: c_int,
    /// Number of output channels for this format.
    pub output_channels: c_int,
    /// V4L2 `FourCC` code (NUL-padded, up to 8 bytes).
    pub fourcc: [u8; 8],
}

impl Default for HalCameraAdaptorFormatInfo {
    fn default() -> Self {
        // SAFETY: HalCameraAdaptorFormatInfo is a #[repr(C)] plain-old-data
        // struct. The HAL ABI contract requires zero-initialization, so an
        // all-zero bit-pattern is a valid "empty" state.
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// HalCameraAdaptorFunctions
// ---------------------------------------------------------------------------

/// Function pointers for the HAL Delegate Camera Adaptor API.
///
/// Loaded at runtime from the delegate shared library. Use
/// [`HalCameraAdaptorFunctions::try_load`] to attempt loading.
///
/// The `hal_delegate_t` parameter (`*mut c_void`) is the `TfLiteDelegate*`
/// cast to an opaque handle.
#[derive(Debug)]
pub struct HalCameraAdaptorFunctions {
    /// `hal_camera_adaptor_is_supported` — returns 1 if format is supported.
    pub is_supported: unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int,
    /// `hal_camera_adaptor_get_format_info` — fills `info` for the given format.
    pub get_format_info: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        *mut HalCameraAdaptorFormatInfo,
        usize,
    ) -> c_int,
}

impl HalCameraAdaptorFunctions {
    /// Attempt to load all HAL Camera Adaptor function pointers from a delegate library.
    ///
    /// Returns `None` if any required symbol is missing.
    ///
    /// # Safety
    ///
    /// The library must remain loaded for the lifetime of this struct.
    #[must_use]
    pub unsafe fn try_load(lib: &libloading::Library) -> Option<Self> {
        unsafe {
            Some(Self {
                is_supported: *lib.get(b"hal_camera_adaptor_is_supported\0").ok()?,
                get_format_info: *lib.get(b"hal_camera_adaptor_get_format_info\0").ok()?,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hal_dtype_discriminants() {
        assert_eq!(HalDtype::U8 as u32, 0);
        assert_eq!(HalDtype::I8 as u32, 1);
        assert_eq!(HalDtype::U16 as u32, 2);
        assert_eq!(HalDtype::I16 as u32, 3);
        assert_eq!(HalDtype::U32 as u32, 4);
        assert_eq!(HalDtype::I32 as u32, 5);
        assert_eq!(HalDtype::U64 as u32, 6);
        assert_eq!(HalDtype::I64 as u32, 7);
        assert_eq!(HalDtype::F16 as u32, 8);
        assert_eq!(HalDtype::F32 as u32, 9);
        assert_eq!(HalDtype::F64 as u32, 10);
    }

    #[test]
    fn tensor_info_default_is_zeroed() {
        let info = HalDmabufTensorInfo::default();
        assert_eq!(info.size, 0);
        assert_eq!(info.offset, 0);
        assert_eq!(info.ndim, 0);
        assert_eq!(info.fd, 0);
        assert!(info.shape.iter().all(|&s| s == 0));
    }

    #[test]
    fn hal_dmabuf_max_ndim() {
        assert_eq!(HAL_DMABUF_MAX_NDIM, 8);
    }

    #[test]
    fn camera_adaptor_format_info_default_is_zeroed() {
        let info = HalCameraAdaptorFormatInfo::default();
        assert_eq!(info.input_channels, 0);
        assert_eq!(info.output_channels, 0);
        assert!(info.fourcc.iter().all(|&b| b == 0));
    }

    #[test]
    fn camera_adaptor_format_info_clone_copy() {
        let mut info = HalCameraAdaptorFormatInfo {
            input_channels: 4,
            output_channels: 3,
            ..Default::default()
        };
        info.fourcc[..4].copy_from_slice(b"RGBA");

        let copied = info;
        assert_eq!(copied.input_channels, 4);
        assert_eq!(copied.output_channels, 3);
        assert_eq!(&copied.fourcc[..4], b"RGBA");
    }

    #[test]
    fn camera_adaptor_format_info_debug() {
        let info = HalCameraAdaptorFormatInfo::default();
        let debug = format!("{info:?}");
        assert!(debug.contains("HalCameraAdaptorFormatInfo"));
        assert!(debug.contains("input_channels"));
    }
}
