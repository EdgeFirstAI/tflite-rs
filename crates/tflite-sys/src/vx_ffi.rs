// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! FFI bindings for the `VxDelegate` DMA-BUF and `CameraAdaptor` APIs.
//!
//! These are loaded at runtime from the `VxDelegate` shared library using
//! `libloading`. The C API is defined in `vx_delegate_dmabuf.h` (MIT license).

use std::ffi::{c_char, c_int, c_void};
use std::os::raw::c_uint;

use crate::{TfLiteBufferHandle, TfLiteDelegate, TfLiteStatus};

/// DMA-BUF synchronization modes for cache coherency.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VxDmaBufSyncMode {
    /// No synchronization needed.
    None = 0,
    /// CPU will read from buffer.
    Read = 1,
    /// CPU will write to buffer.
    Write = 2,
    /// CPU will read and write.
    ReadWrite = 3,
}

/// DMA-BUF ownership model.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VxDmaBufOwnership {
    /// Client owns the buffer (import mode).
    Client = 0,
    /// Delegate owns the buffer (export mode).
    Delegate = 1,
}

/// DMA-BUF descriptor returned when requesting a delegate-allocated buffer.
#[repr(C)]
#[derive(Debug)]
pub struct VxDmaBufDesc {
    /// DMA-BUF file descriptor.
    pub fd: c_int,
    /// Buffer size in bytes.
    pub size: usize,
    /// Optional: mmap'd pointer (NULL if not mapped).
    pub map_ptr: *mut c_void,
}

impl Default for VxDmaBufDesc {
    fn default() -> Self {
        Self {
            fd: -1,
            size: 0,
            map_ptr: std::ptr::null_mut(),
        }
    }
}

/// Function pointers for `VxDelegate` DMA-BUF operations.
///
/// Loaded at runtime from the delegate shared library. Use [`VxDmaBufFunctions::try_load`]
/// to attempt loading from a `libloading::Library`.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
pub struct VxDmaBufFunctions {
    /// Returns the inner `TfLiteDelegate` pointer for the `VxDelegate` singleton.
    ///
    /// This is an advanced escape hatch only needed when the delegate was
    /// created via `TfLiteExternalDelegateCreate` (e.g., nnstreamer path)
    /// rather than `tflite_plugin_create_delegate`.
    ///
    /// Maps to `VxDelegateGetInstance`.
    pub get_instance: unsafe extern "C" fn() -> *mut TfLiteDelegate,
    /// `VxDelegateRegisterDmaBuf`
    pub register: unsafe extern "C" fn(
        *mut TfLiteDelegate,
        c_int,
        usize,
        VxDmaBufSyncMode,
    ) -> TfLiteBufferHandle,
    /// `VxDelegateUnregisterDmaBuf`
    pub unregister: unsafe extern "C" fn(*mut TfLiteDelegate, TfLiteBufferHandle) -> TfLiteStatus,
    /// `VxDelegateRequestDmaBuf`
    pub request: unsafe extern "C" fn(
        *mut TfLiteDelegate,
        c_int,
        VxDmaBufOwnership,
        *mut VxDmaBufDesc,
    ) -> TfLiteBufferHandle,
    /// `VxDelegateReleaseDmaBuf`
    pub release: unsafe extern "C" fn(*mut TfLiteDelegate, TfLiteBufferHandle) -> TfLiteStatus,
    /// `VxDelegateBeginCpuAccess`
    pub begin_cpu_access: unsafe extern "C" fn(
        *mut TfLiteDelegate,
        TfLiteBufferHandle,
        VxDmaBufSyncMode,
    ) -> TfLiteStatus,
    /// `VxDelegateEndCpuAccess`
    pub end_cpu_access: unsafe extern "C" fn(
        *mut TfLiteDelegate,
        TfLiteBufferHandle,
        VxDmaBufSyncMode,
    ) -> TfLiteStatus,
    /// `VxDelegateBindDmaBufToTensor`
    pub bind_to_tensor:
        unsafe extern "C" fn(*mut TfLiteDelegate, TfLiteBufferHandle, c_int) -> TfLiteStatus,
    /// `VxDelegateIsDmaBufSupported`
    pub is_supported: unsafe extern "C" fn(*mut TfLiteDelegate) -> bool,
    /// `VxDelegateGetDmaBufFd`
    pub get_fd: unsafe extern "C" fn(*mut TfLiteDelegate, TfLiteBufferHandle) -> c_int,
    /// `VxDelegateSyncForDevice`
    pub sync_for_device:
        unsafe extern "C" fn(*mut TfLiteDelegate, TfLiteBufferHandle) -> TfLiteStatus,
    /// `VxDelegateSyncForCpu`
    pub sync_for_cpu: unsafe extern "C" fn(*mut TfLiteDelegate, TfLiteBufferHandle) -> TfLiteStatus,
    /// `VxDelegateSetActiveDmaBuf`
    pub set_active:
        unsafe extern "C" fn(*mut TfLiteDelegate, c_int, TfLiteBufferHandle) -> TfLiteStatus,
    /// `VxDelegateInvalidateGraph`
    pub invalidate_graph: unsafe extern "C" fn(*mut TfLiteDelegate) -> TfLiteStatus,
    /// `VxDelegateIsGraphCompiled`
    pub is_graph_compiled: unsafe extern "C" fn(*mut TfLiteDelegate) -> bool,
    /// `VxDelegateGetActiveBuffer`
    pub get_active_buffer: unsafe extern "C" fn(*mut TfLiteDelegate, c_int) -> TfLiteBufferHandle,
}

impl VxDmaBufFunctions {
    /// Attempt to load all DMA-BUF function pointers from a delegate library.
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
                get_instance: *lib.get(b"VxDelegateGetInstance\0").ok()?,
                register: *lib.get(b"VxDelegateRegisterDmaBuf\0").ok()?,
                unregister: *lib.get(b"VxDelegateUnregisterDmaBuf\0").ok()?,
                request: *lib.get(b"VxDelegateRequestDmaBuf\0").ok()?,
                release: *lib.get(b"VxDelegateReleaseDmaBuf\0").ok()?,
                begin_cpu_access: *lib.get(b"VxDelegateBeginCpuAccess\0").ok()?,
                end_cpu_access: *lib.get(b"VxDelegateEndCpuAccess\0").ok()?,
                bind_to_tensor: *lib.get(b"VxDelegateBindDmaBufToTensor\0").ok()?,
                is_supported: *lib.get(b"VxDelegateIsDmaBufSupported\0").ok()?,
                get_fd: *lib.get(b"VxDelegateGetDmaBufFd\0").ok()?,
                sync_for_device: *lib.get(b"VxDelegateSyncForDevice\0").ok()?,
                sync_for_cpu: *lib.get(b"VxDelegateSyncForCpu\0").ok()?,
                set_active: *lib.get(b"VxDelegateSetActiveDmaBuf\0").ok()?,
                invalidate_graph: *lib.get(b"VxDelegateInvalidateGraph\0").ok()?,
                is_graph_compiled: *lib.get(b"VxDelegateIsGraphCompiled\0").ok()?,
                get_active_buffer: *lib.get(b"VxDelegateGetActiveBuffer\0").ok()?,
            })
        }
    }
}

/// Function pointers for `VxDelegate` `CameraAdaptor` operations.
///
/// Loaded at runtime from the delegate shared library. Use
/// [`VxCameraAdaptorFunctions::try_load`] to attempt loading.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
pub struct VxCameraAdaptorFunctions {
    /// `VxCameraAdaptorSetFormat`
    pub set_format: unsafe extern "C" fn(*mut TfLiteDelegate, c_int, *const c_char) -> TfLiteStatus,
    /// `VxCameraAdaptorSetFormatEx`
    pub set_format_ex: unsafe extern "C" fn(
        *mut TfLiteDelegate,
        c_int,
        *const c_char,
        c_uint,
        c_uint,
        bool,
        c_uint,
    ) -> TfLiteStatus,
    /// `VxCameraAdaptorSetFormats`
    pub set_formats: unsafe extern "C" fn(
        *mut TfLiteDelegate,
        c_int,
        *const c_char,
        *const c_char,
    ) -> TfLiteStatus,
    /// `VxCameraAdaptorSetFourCC`
    pub set_fourcc: unsafe extern "C" fn(*mut TfLiteDelegate, c_int, c_uint) -> TfLiteStatus,
    /// `VxCameraAdaptorGetFormat`
    pub get_format: unsafe extern "C" fn(*mut TfLiteDelegate, c_int) -> *const c_char,
    /// `VxCameraAdaptorIsSupported`
    pub is_supported: unsafe extern "C" fn(*const c_char) -> bool,
    /// `VxCameraAdaptorGetInputChannels`
    pub get_input_channels: unsafe extern "C" fn(*const c_char) -> c_int,
    /// `VxCameraAdaptorGetOutputChannels`
    pub get_output_channels: unsafe extern "C" fn(*const c_char) -> c_int,
    /// `VxCameraAdaptorGetFourCC`
    pub get_fourcc: unsafe extern "C" fn(*const c_char) -> *const c_char,
    /// `VxCameraAdaptorFromFourCC`
    pub from_fourcc: unsafe extern "C" fn(*const c_char) -> *const c_char,
}

impl VxCameraAdaptorFunctions {
    /// Attempt to load all `CameraAdaptor` function pointers from a delegate library.
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
                set_format: *lib.get(b"VxCameraAdaptorSetFormat\0").ok()?,
                set_format_ex: *lib.get(b"VxCameraAdaptorSetFormatEx\0").ok()?,
                set_formats: *lib.get(b"VxCameraAdaptorSetFormats\0").ok()?,
                set_fourcc: *lib.get(b"VxCameraAdaptorSetFourCC\0").ok()?,
                get_format: *lib.get(b"VxCameraAdaptorGetFormat\0").ok()?,
                is_supported: *lib.get(b"VxCameraAdaptorIsSupported\0").ok()?,
                get_input_channels: *lib.get(b"VxCameraAdaptorGetInputChannels\0").ok()?,
                get_output_channels: *lib.get(b"VxCameraAdaptorGetOutputChannels\0").ok()?,
                get_fourcc: *lib.get(b"VxCameraAdaptorGetFourCC\0").ok()?,
                from_fourcc: *lib.get(b"VxCameraAdaptorFromFourCC\0").ok()?,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_buffer_handle_is_negative_one() {
        assert_eq!(crate::kTfLiteNullBufferHandle, -1);
    }

    #[test]
    fn dma_buf_sync_mode_discriminants() {
        assert_eq!(VxDmaBufSyncMode::None as u32, 0);
        assert_eq!(VxDmaBufSyncMode::Read as u32, 1);
        assert_eq!(VxDmaBufSyncMode::Write as u32, 2);
        assert_eq!(VxDmaBufSyncMode::ReadWrite as u32, 3);
    }

    #[test]
    fn dma_buf_ownership_discriminants() {
        assert_eq!(VxDmaBufOwnership::Client as u32, 0);
        assert_eq!(VxDmaBufOwnership::Delegate as u32, 1);
    }

    #[test]
    fn dma_buf_desc_default() {
        let desc = VxDmaBufDesc::default();
        assert_eq!(desc.fd, -1);
        assert_eq!(desc.size, 0);
        assert!(desc.map_ptr.is_null());
    }
}
