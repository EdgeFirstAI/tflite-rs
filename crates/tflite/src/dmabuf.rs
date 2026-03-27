// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Safe DMA-BUF zero-copy API for `TFLite` delegates.
//!
//! This module provides [`DmaBuf`], a safe wrapper around the HAL Delegate
//! DMA-BUF API (`hal_dmabuf_*`). It enables zero-copy inference by querying
//! DMA-BUF tensor metadata and synchronizing cache coherency between CPU
//! and device (NPU).
//!
//! The HAL API is the primary interface. Legacy `VxDelegate` DMA-BUF
//! symbols are used as a fallback for delegates that have not yet adopted
//! the HAL API, but all `VxDelegate`-specific methods are deprecated and
//! will be removed in a future release.
//!
//! # Primary API (HAL Delegate)
//!
//! ```no_run
//! use edgefirst_tflite::Delegate;
//!
//! let delegate = Delegate::load("libvx_delegate.so")?;
//! let dmabuf = delegate.dmabuf().expect("DMA-BUF not supported");
//!
//! if dmabuf.is_supported() {
//!     // Query tensor DMA-BUF metadata
//!     let info = dmabuf.tensor_info(0)?;
//!     println!("fd={}, shape={:?}, dtype={:?}", info.fd, info.shape, info.dtype);
//!
//!     // Sync for NPU inference, then sync back for CPU
//!     dmabuf.sync_for_device(0)?;
//!     // ... invoke interpreter ...
//!     dmabuf.sync_for_cpu(0)?;
//! }
//! # Ok::<(), edgefirst_tflite::Error>(())
//! ```

use std::ffi::c_void;
use std::ptr::NonNull;

use edgefirst_tflite_sys::hal_ffi::{HalDmaBufFunctions, HalDmabufTensorInfo, HalDtype};
use edgefirst_tflite_sys::vx_ffi::{
    VxDmaBufDesc, VxDmaBufFunctions, VxDmaBufOwnership, VxDmaBufSyncMode,
};
use edgefirst_tflite_sys::{kTfLiteNullBufferHandle, TfLiteDelegate};

use crate::error::{self, Error, Result};

// ---------------------------------------------------------------------------
// Public types — HAL API
// ---------------------------------------------------------------------------

/// Element data type for DMA-BUF tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// 16-bit floating point.
    F16,
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
}

impl DType {
    fn from_hal(h: HalDtype) -> Self {
        match h {
            HalDtype::U8 => Self::U8,
            HalDtype::I8 => Self::I8,
            HalDtype::U16 => Self::U16,
            HalDtype::I16 => Self::I16,
            HalDtype::U32 => Self::U32,
            HalDtype::I32 => Self::I32,
            HalDtype::U64 => Self::U64,
            HalDtype::I64 => Self::I64,
            HalDtype::F16 => Self::F16,
            HalDtype::F32 => Self::F32,
            HalDtype::F64 => Self::F64,
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}

/// DMA-BUF tensor information returned by [`DmaBuf::tensor_info`].
///
/// Contains the DMA-BUF file descriptor, buffer geometry, and element type
/// for a single tensor. The `fd` is borrowed from the delegate and must
/// **not** be closed by the caller.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Buffer size in bytes.
    pub size: usize,
    /// Byte offset within the DMA-BUF.
    pub offset: usize,
    /// Tensor shape (dimensions).
    pub shape: Vec<usize>,
    /// DMA-BUF file descriptor (borrowed — do **not** close).
    pub fd: i32,
    /// Element data type.
    pub dtype: DType,
}

// ---------------------------------------------------------------------------
// Public types — Legacy `VxDelegate` API (deprecated)
// ---------------------------------------------------------------------------

/// Synchronization modes for DMA-BUF cache coherency.
///
/// Used by the deprecated `VxDelegate`-specific buffer registration API.
#[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyncMode {
    /// No synchronization needed.
    None,
    /// CPU will read from buffer.
    Read,
    /// CPU will write to buffer.
    Write,
    /// CPU will read and write.
    ReadWrite,
}

#[allow(deprecated)]
impl SyncMode {
    fn to_raw(self) -> VxDmaBufSyncMode {
        match self {
            Self::None => VxDmaBufSyncMode::None,
            Self::Read => VxDmaBufSyncMode::Read,
            Self::Write => VxDmaBufSyncMode::Write,
            Self::ReadWrite => VxDmaBufSyncMode::ReadWrite,
        }
    }
}

/// Ownership model for DMA-BUF buffers.
///
/// Used by the deprecated `VxDelegate`-specific buffer allocation API.
#[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ownership {
    /// Client owns the buffer (import mode).
    Client,
    /// Delegate owns the buffer (export mode).
    Delegate,
}

#[allow(deprecated)]
impl Ownership {
    fn to_raw(self) -> VxDmaBufOwnership {
        match self {
            Self::Client => VxDmaBufOwnership::Client,
            Self::Delegate => VxDmaBufOwnership::Delegate,
        }
    }
}

/// Descriptor for a delegate-allocated DMA-BUF.
///
/// Returned by the deprecated [`DmaBuf::request`] method.
#[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
#[derive(Debug)]
pub struct BufferDesc {
    /// DMA-BUF file descriptor.
    pub fd: i32,
    /// Buffer size in bytes.
    pub size: usize,
    /// Optional mmap'd pointer (null if not mapped).
    pub map_ptr: Option<*mut std::ffi::c_void>,
}

/// Opaque handle to a registered DMA-BUF.
///
/// Used by the deprecated `VxDelegate`-specific buffer management API.
#[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(i32);

#[allow(deprecated)]
impl BufferHandle {
    /// Create a `BufferHandle` from a raw integer value.
    ///
    /// This is intended for FFI consumers (e.g., Python bindings) that need
    /// to reconstruct a handle from a previously obtained raw value.
    #[must_use]
    pub fn from_raw(value: i32) -> Self {
        Self(value)
    }

    /// Returns the raw buffer handle value.
    #[must_use]
    pub fn raw(self) -> i32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// DmaBuf
// ---------------------------------------------------------------------------

/// Safe interface for delegate DMA-BUF zero-copy operations.
///
/// Obtained from [`Delegate::dmabuf()`](crate::Delegate::dmabuf) when the
/// loaded delegate supports DMA-BUF. Uses the HAL Delegate DMA-BUF API
/// (`hal_dmabuf_*`) as the primary backend, with legacy `VxDelegate` symbols
/// as a fallback.
#[derive(Debug)]
pub struct DmaBuf<'a> {
    delegate: NonNull<TfLiteDelegate>,
    /// Inner delegate handle from `hal_dmabuf_get_instance()`.
    ///
    /// This opaque `hal_delegate_t` pointer is what HAL API functions expect.
    /// It is `None` when HAL symbols are not present or `get_instance()`
    /// returned null.
    hal_handle: Option<*mut c_void>,
    hal_fns: Option<&'a HalDmaBufFunctions>,
    vx_fns: Option<&'a VxDmaBufFunctions>,
}

impl<'a> DmaBuf<'a> {
    /// Create a new `DmaBuf` wrapper with HAL and/or `VxDelegate` backends.
    pub(crate) fn new(
        delegate: NonNull<TfLiteDelegate>,
        hal_handle: Option<*mut c_void>,
        hal_fns: Option<&'a HalDmaBufFunctions>,
        vx_fns: Option<&'a VxDmaBufFunctions>,
    ) -> Self {
        Self {
            delegate,
            hal_handle,
            hal_fns,
            vx_fns,
        }
    }

    /// Returns the inner `hal_delegate_t` handle for HAL API calls.
    ///
    /// Returns the handle from `hal_dmabuf_get_instance()` when available,
    /// falling back to casting the outer `TfLiteDelegate*` pointer. The
    /// fallback is kept for delegates that expose HAL symbols but return a
    /// null instance handle (should not happen in practice).
    fn hal_delegate_ptr(&self) -> *mut c_void {
        self.hal_handle
            .unwrap_or_else(|| self.delegate.as_ptr().cast::<c_void>())
    }

    // =======================================================================
    // Primary API (HAL Delegate DMA-BUF)
    // =======================================================================

    /// Check if DMA-BUF zero-copy is supported by this delegate.
    ///
    /// Uses the HAL API when available, falling back to `VxDelegate`.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        if let Some(hal) = self.hal_fns {
            // SAFETY: delegate pointer is valid and cast to the opaque
            // hal_delegate_t (void*) that the HAL API expects.
            unsafe { (hal.is_supported)(self.hal_delegate_ptr()) == 1 }
        } else if let Some(vx) = self.vx_fns {
            // SAFETY: delegate pointer is valid; function pointer was
            // loaded from the same library that created the delegate.
            unsafe { (vx.is_supported)(self.delegate.as_ptr()) }
        } else {
            false
        }
    }

    /// Get DMA-BUF tensor information for a given tensor index.
    ///
    /// Returns the DMA-BUF file descriptor, buffer size, offset, shape,
    /// and element type. The file descriptor is borrowed from the delegate
    /// and must **not** be closed by the caller.
    ///
    /// This method requires the HAL Delegate DMA-BUF API. It is not
    /// available when only the legacy `VxDelegate` backend is present.
    ///
    /// # Errors
    ///
    /// Returns an error if the HAL backend is not available, or if the
    /// underlying `hal_dmabuf_get_tensor_info` call fails (e.g., invalid
    /// tensor index, DMA-BUF not supported).
    pub fn tensor_info(&self, tensor_index: i32) -> Result<TensorInfo> {
        let hal = self.hal_fns.ok_or_else(|| {
            Error::invalid_argument("tensor_info requires the HAL Delegate DMA-BUF API")
        })?;

        let mut info = HalDmabufTensorInfo::default();
        // SAFETY: delegate pointer is valid. `info` is a valid mutable
        // reference that the C function will populate. We pass the struct
        // size for ABI versioning compatibility.
        let ret = unsafe {
            (hal.get_tensor_info)(
                self.hal_delegate_ptr(),
                tensor_index,
                &mut info,
                std::mem::size_of::<HalDmabufTensorInfo>(),
            )
        };
        error::hal_to_result(ret, "hal_dmabuf_get_tensor_info")?;

        let ndim = info.ndim.min(info.shape.len());
        Ok(TensorInfo {
            size: info.size,
            offset: info.offset,
            shape: info.shape[..ndim].to_vec(),
            fd: info.fd,
            dtype: DType::from_hal(info.dtype),
        })
    }

    /// Sync tensor buffer for device (NPU) access.
    ///
    /// Flushes CPU caches so the device can read the buffer contents.
    /// Call this after writing to an input tensor and before invoking
    /// inference.
    ///
    /// Uses the HAL API when available. Falls back to `VxDelegate` by
    /// looking up the active buffer handle for the tensor index.
    ///
    /// # Errors
    ///
    /// Returns an error if the sync operation fails or no DMA-BUF
    /// backend is available.
    pub fn sync_for_device(&self, tensor_index: i32) -> Result<()> {
        if let Some(hal) = self.hal_fns {
            // SAFETY: delegate pointer is valid, cast to hal_delegate_t.
            let ret = unsafe { (hal.sync_for_device)(self.hal_delegate_ptr(), tensor_index) };
            error::hal_to_result(ret, "hal_dmabuf_sync_for_device")
        } else if let Some(vx) = self.vx_fns {
            // Fallback: look up the active buffer handle for this tensor.
            // SAFETY: delegate pointer is valid.
            let handle = unsafe { (vx.get_active_buffer)(self.delegate.as_ptr(), tensor_index) };
            if handle == kTfLiteNullBufferHandle {
                return Err(Error::invalid_argument(format!(
                    "no active DMA-BUF for tensor index {tensor_index}"
                )));
            }
            // SAFETY: delegate pointer is valid; handle was obtained above.
            let status = unsafe { (vx.sync_for_device)(self.delegate.as_ptr(), handle) };
            error::status_to_result(status)
        } else {
            Err(Error::invalid_argument("no DMA-BUF backend available"))
        }
    }

    /// Sync tensor buffer for CPU access.
    ///
    /// Invalidates CPU caches so the CPU sees device-written data.
    /// Call this after inference completes and before reading output
    /// tensor data on the CPU.
    ///
    /// Uses the HAL API when available. Falls back to `VxDelegate` by
    /// looking up the active buffer handle for the tensor index.
    ///
    /// # Errors
    ///
    /// Returns an error if the sync operation fails or no DMA-BUF
    /// backend is available.
    pub fn sync_for_cpu(&self, tensor_index: i32) -> Result<()> {
        if let Some(hal) = self.hal_fns {
            // SAFETY: delegate pointer is valid, cast to hal_delegate_t.
            let ret = unsafe { (hal.sync_for_cpu)(self.hal_delegate_ptr(), tensor_index) };
            error::hal_to_result(ret, "hal_dmabuf_sync_for_cpu")
        } else if let Some(vx) = self.vx_fns {
            // Fallback: look up the active buffer handle for this tensor.
            // SAFETY: delegate pointer is valid.
            let handle = unsafe { (vx.get_active_buffer)(self.delegate.as_ptr(), tensor_index) };
            if handle == kTfLiteNullBufferHandle {
                return Err(Error::invalid_argument(format!(
                    "no active DMA-BUF for tensor index {tensor_index}"
                )));
            }
            // SAFETY: delegate pointer is valid; handle was obtained above.
            let status = unsafe { (vx.sync_for_cpu)(self.delegate.as_ptr(), handle) };
            error::status_to_result(status)
        } else {
            Err(Error::invalid_argument("no DMA-BUF backend available"))
        }
    }

    // =======================================================================
    // Legacy `VxDelegate` API (deprecated)
    // =======================================================================

    /// Helper: get the `VxDelegate` function pointers or return an error.
    fn vx(&self) -> Result<&VxDmaBufFunctions> {
        self.vx_fns.ok_or_else(|| {
            Error::invalid_argument(
                "this method requires the `VxDelegate` DMA-BUF API, which is not available",
            )
        })
    }

    // --- Buffer Registration (Import Mode) ---

    /// Register an externally-allocated DMA-BUF.
    ///
    /// # Errors
    ///
    /// Returns an error if registration fails (null buffer handle returned).
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[allow(deprecated)]
    pub fn register(&self, fd: i32, size: usize, sync_mode: SyncMode) -> Result<BufferHandle> {
        let vx = self.vx()?;
        // SAFETY: `self.delegate` is a valid delegate pointer; function pointer
        // was loaded from the same library that created the delegate.
        let handle = unsafe { (vx.register)(self.delegate.as_ptr(), fd, size, sync_mode.to_raw()) };
        if handle == kTfLiteNullBufferHandle {
            return Err(Error::null_pointer(
                "`VxDelegate`RegisterDmaBuf returned null handle",
            ));
        }
        Ok(BufferHandle(handle))
    }

    /// Unregister a previously registered DMA-BUF.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[allow(deprecated)]
    pub fn unregister(&self, handle: BufferHandle) -> Result<()> {
        let vx = self.vx()?;
        // SAFETY: delegate and function pointers are valid; handle was obtained
        // from a prior `register` call on this delegate.
        let status = unsafe { (vx.unregister)(self.delegate.as_ptr(), handle.0) };
        error::status_to_result(status)
    }

    // --- Buffer Allocation (Export Mode) ---

    /// Request the delegate to allocate a DMA-BUF for a tensor.
    ///
    /// The `size` parameter must match the tensor's byte size (obtained from
    /// [`Tensor::byte_size`](crate::Tensor::byte_size) after the interpreter
    /// is built). When `CameraAdaptor` is active, the size should reflect the
    /// camera format (e.g., 4 channels for RGBA instead of 3 for RGB).
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[allow(deprecated)]
    pub fn request(
        &self,
        tensor_index: i32,
        ownership: Ownership,
        size: usize,
    ) -> Result<(BufferHandle, BufferDesc)> {
        let vx = self.vx()?;
        let mut desc = VxDmaBufDesc {
            size,
            ..VxDmaBufDesc::default()
        };
        // SAFETY: delegate pointer is valid; `desc` is a valid mutable reference
        // that the C function will populate with the allocated buffer info.
        let handle = unsafe {
            (vx.request)(
                self.delegate.as_ptr(),
                tensor_index,
                ownership.to_raw(),
                &mut desc,
            )
        };
        if handle == kTfLiteNullBufferHandle {
            return Err(Error::null_pointer(
                "`VxDelegate`RequestDmaBuf returned null handle",
            ));
        }
        let map_ptr = if desc.map_ptr.is_null() {
            Option::None
        } else {
            Some(desc.map_ptr)
        };
        Ok((
            BufferHandle(handle),
            BufferDesc {
                fd: desc.fd,
                size: desc.size,
                map_ptr,
            },
        ))
    }

    /// Release a delegate-allocated DMA-BUF.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[allow(deprecated)]
    pub fn release(&self, handle: BufferHandle) -> Result<()> {
        let vx = self.vx()?;
        // SAFETY: delegate pointer is valid; handle was obtained from `request`.
        let status = unsafe { (vx.release)(self.delegate.as_ptr(), handle.0) };
        error::status_to_result(status)
    }

    // --- Tensor Binding ---

    /// Bind a DMA-BUF to a tensor for zero-copy inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[allow(deprecated)]
    pub fn bind_to_tensor(&self, handle: BufferHandle, tensor_index: i32) -> Result<()> {
        let vx = self.vx()?;
        // SAFETY: delegate pointer is valid; handle and tensor_index are caller-provided.
        let status = unsafe { (vx.bind_to_tensor)(self.delegate.as_ptr(), handle.0, tensor_index) };
        error::status_to_result(status)
    }

    /// Get the file descriptor for a buffer handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns -1.
    #[deprecated(
        note = "`VxDelegate`-specific, use DmaBuf::tensor_info() instead; will be removed in a future release"
    )]
    #[allow(deprecated)]
    pub fn buffer_fd(&self, handle: BufferHandle) -> Result<i32> {
        let vx = self.vx()?;
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let fd = unsafe { (vx.get_fd)(self.delegate.as_ptr(), handle.0) };
        if fd < 0 {
            return Err(Error::invalid_argument(format!(
                "`VxDelegate`GetDmaBufFd returned {fd} for handle {}",
                handle.0
            )));
        }
        Ok(fd)
    }

    // --- Cache Synchronization (Legacy) ---

    /// Begin CPU access to a DMA-BUF (ensure cache coherency).
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(
        note = "`VxDelegate`-specific, use DmaBuf::sync_for_cpu() instead; will be removed in a future release"
    )]
    #[allow(deprecated)]
    pub fn begin_cpu_access(&self, handle: BufferHandle, mode: SyncMode) -> Result<()> {
        let vx = self.vx()?;
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let status =
            unsafe { (vx.begin_cpu_access)(self.delegate.as_ptr(), handle.0, mode.to_raw()) };
        error::status_to_result(status)
    }

    /// End CPU access to a DMA-BUF (flush caches).
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(
        note = "`VxDelegate`-specific, use DmaBuf::sync_for_device() instead; will be removed in a future release"
    )]
    #[allow(deprecated)]
    pub fn end_cpu_access(&self, handle: BufferHandle, mode: SyncMode) -> Result<()> {
        let vx = self.vx()?;
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let status =
            unsafe { (vx.end_cpu_access)(self.delegate.as_ptr(), handle.0, mode.to_raw()) };
        error::status_to_result(status)
    }

    /// Sync buffer for device (NPU) access by buffer handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(
        note = "`VxDelegate`-specific, use DmaBuf::sync_for_device(tensor_index) instead; will be removed in a future release"
    )]
    #[allow(deprecated)]
    pub fn sync_for_device_by_handle(&self, handle: BufferHandle) -> Result<()> {
        let vx = self.vx()?;
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let status = unsafe { (vx.sync_for_device)(self.delegate.as_ptr(), handle.0) };
        error::status_to_result(status)
    }

    /// Sync buffer for CPU access by buffer handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(
        note = "`VxDelegate`-specific, use DmaBuf::sync_for_cpu(tensor_index) instead; will be removed in a future release"
    )]
    #[allow(deprecated)]
    pub fn sync_for_cpu_by_handle(&self, handle: BufferHandle) -> Result<()> {
        let vx = self.vx()?;
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let status = unsafe { (vx.sync_for_cpu)(self.delegate.as_ptr(), handle.0) };
        error::status_to_result(status)
    }

    // --- Buffer Cycling (Legacy) ---

    /// Set the active DMA-BUF for a tensor (buffer pool cycling).
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[allow(deprecated)]
    pub fn set_active(&self, tensor_index: i32, handle: BufferHandle) -> Result<()> {
        let vx = self.vx()?;
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let status = unsafe { (vx.set_active)(self.delegate.as_ptr(), tensor_index, handle.0) };
        error::status_to_result(status)
    }

    /// Get the currently active buffer for a tensor.
    ///
    /// Returns `None` if no buffer is active for this tensor.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[allow(deprecated)]
    #[must_use]
    pub fn active_buffer(&self, tensor_index: i32) -> Option<BufferHandle> {
        let vx = self.vx_fns?;
        // SAFETY: delegate pointer is valid.
        let handle = unsafe { (vx.get_active_buffer)(self.delegate.as_ptr(), tensor_index) };
        if handle == kTfLiteNullBufferHandle {
            None
        } else {
            Some(BufferHandle(handle))
        }
    }

    /// Invalidate the compiled graph (forces recompilation on next invoke).
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    pub fn invalidate_graph(&self) -> Result<()> {
        let vx = self.vx()?;
        // SAFETY: delegate pointer is valid.
        let status = unsafe { (vx.invalidate_graph)(self.delegate.as_ptr()) };
        error::status_to_result(status)
    }

    /// Check if the graph has been compiled.
    #[deprecated(note = "`VxDelegate`-specific, will be removed in a future release")]
    #[must_use]
    pub fn is_graph_compiled(&self) -> bool {
        self.vx_fns.is_some_and(|vx| {
            // SAFETY: delegate pointer is valid.
            unsafe { (vx.is_graph_compiled)(self.delegate.as_ptr()) }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_from_hal_roundtrip() {
        let cases = [
            (HalDtype::U8, DType::U8),
            (HalDtype::I8, DType::I8),
            (HalDtype::U16, DType::U16),
            (HalDtype::I16, DType::I16),
            (HalDtype::U32, DType::U32),
            (HalDtype::I32, DType::I32),
            (HalDtype::U64, DType::U64),
            (HalDtype::I64, DType::I64),
            (HalDtype::F16, DType::F16),
            (HalDtype::F32, DType::F32),
            (HalDtype::F64, DType::F64),
        ];
        for (hal, expected) in cases {
            assert_eq!(DType::from_hal(hal), expected);
        }
    }

    #[test]
    fn dtype_display() {
        assert_eq!(DType::U8.to_string(), "u8");
        assert_eq!(DType::I8.to_string(), "i8");
        assert_eq!(DType::F32.to_string(), "f32");
        assert_eq!(DType::F64.to_string(), "f64");
    }

    #[test]
    fn dtype_clone_copy_eq_hash() {
        use std::collections::HashSet;

        let a = DType::F32;
        let b = a;
        assert_eq!(a, b);

        let mut set = HashSet::new();
        set.insert(DType::U8);
        set.insert(DType::I8);
        set.insert(DType::U8); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn tensor_info_debug() {
        let info = TensorInfo {
            size: 4096,
            offset: 0,
            shape: vec![1, 3, 224, 224],
            fd: 5,
            dtype: DType::U8,
        };
        let debug = format!("{info:?}");
        assert!(debug.contains("4096"));
        assert!(debug.contains("224"));
        assert!(debug.contains("U8"));
    }

    #[allow(deprecated)]
    #[test]
    fn sync_mode_to_raw() {
        assert_eq!(SyncMode::None.to_raw(), VxDmaBufSyncMode::None);
        assert_eq!(SyncMode::Read.to_raw(), VxDmaBufSyncMode::Read);
        assert_eq!(SyncMode::Write.to_raw(), VxDmaBufSyncMode::Write);
        assert_eq!(SyncMode::ReadWrite.to_raw(), VxDmaBufSyncMode::ReadWrite);
    }

    #[allow(deprecated)]
    #[test]
    fn ownership_to_raw() {
        assert_eq!(Ownership::Client.to_raw(), VxDmaBufOwnership::Client);
        assert_eq!(Ownership::Delegate.to_raw(), VxDmaBufOwnership::Delegate);
    }

    #[allow(deprecated)]
    #[test]
    fn buffer_handle_raw() {
        let handle = BufferHandle(42);
        assert_eq!(handle.raw(), 42);
    }

    #[allow(deprecated)]
    #[test]
    fn buffer_handle_from_raw() {
        let handle = BufferHandle::from_raw(7);
        assert_eq!(handle.raw(), 7);
    }

    #[allow(deprecated)]
    #[test]
    fn buffer_handle_equality() {
        let a = BufferHandle(7);
        let b = BufferHandle(7);
        let c = BufferHandle(99);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[allow(deprecated)]
    #[test]
    fn buffer_desc_debug() {
        let desc = BufferDesc {
            fd: 3,
            size: 4096,
            map_ptr: Option::None,
        };
        let debug = format!("{desc:?}");
        assert!(debug.contains("fd"));
        assert!(debug.contains('3'));
    }
}
