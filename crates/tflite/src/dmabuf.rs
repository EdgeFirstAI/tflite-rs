// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Safe DMA-BUF zero-copy API for `VxDelegate`.
//!
//! This module provides [`DmaBuf`], a safe wrapper around the `VxDelegate`
//! DMA-BUF C API. It enables zero-copy inference by binding DMA-BUF file
//! descriptors directly to `TFLite` tensors, avoiding CPU-side memory copies.
//!
//! # Modes
//!
//! - **Import mode**: The application owns the buffers (e.g., from V4L2 or
//!   DRM). Register them with [`DmaBuf::register`].
//! - **Export mode**: The delegate allocates buffers. Request them with
//!   [`DmaBuf::request`].
//!
//! # Example
//!
//! ```no_run
//! use edgefirst_tflite::Delegate;
//! use edgefirst_tflite::dmabuf::SyncMode;
//!
//! # let camera_fd = 0i32;
//! # let buffer_size = 0usize;
//! let delegate = Delegate::load("libvx_delegate.so")?;
//! let dmabuf = delegate.dmabuf().expect("DMA-BUF not supported");
//!
//! // Register an externally-allocated DMA-BUF
//! let handle = dmabuf.register(camera_fd, buffer_size, SyncMode::None)?;
//! dmabuf.bind_to_tensor(handle, 0)?;
//! # Ok::<(), edgefirst_tflite::Error>(())
//! ```

use std::ptr::NonNull;

use edgefirst_tflite_sys::vx_ffi::{
    VxDmaBufDesc, VxDmaBufFunctions, VxDmaBufOwnership, VxDmaBufSyncMode,
};
use edgefirst_tflite_sys::{kTfLiteNullBufferHandle, TfLiteDelegate};

use crate::error::{self, Error, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Synchronization modes for DMA-BUF cache coherency.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ownership {
    /// Client owns the buffer (import mode).
    Client,
    /// Delegate owns the buffer (export mode).
    Delegate,
}

impl Ownership {
    fn to_raw(self) -> VxDmaBufOwnership {
        match self {
            Self::Client => VxDmaBufOwnership::Client,
            Self::Delegate => VxDmaBufOwnership::Delegate,
        }
    }
}

/// Descriptor for a delegate-allocated DMA-BUF.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(i32);

impl BufferHandle {
    /// Returns the raw buffer handle value.
    #[must_use]
    pub fn raw(self) -> i32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// DmaBuf
// ---------------------------------------------------------------------------

/// Safe interface for `VxDelegate` DMA-BUF zero-copy operations.
///
/// Obtained from [`Delegate::dmabuf()`](crate::Delegate::dmabuf) when the
/// loaded delegate supports DMA-BUF.
#[derive(Debug)]
pub struct DmaBuf<'a> {
    delegate: NonNull<TfLiteDelegate>,
    fns: &'a VxDmaBufFunctions,
}

impl<'a> DmaBuf<'a> {
    /// Create a new `DmaBuf` wrapper.
    pub(crate) fn new(delegate: NonNull<TfLiteDelegate>, fns: &'a VxDmaBufFunctions) -> Self {
        Self { delegate, fns }
    }

    /// Check if DMA-BUF zero-copy is supported.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        // SAFETY: `self.delegate` is a valid non-null delegate pointer and
        // `self.fns` contains valid function pointers loaded from the same library.
        unsafe { (self.fns.is_supported)(self.delegate.as_ptr()) }
    }

    // --- Buffer Registration (Import Mode) ---

    /// Register an externally-allocated DMA-BUF.
    ///
    /// # Errors
    ///
    /// Returns an error if registration fails (null buffer handle returned).
    pub fn register(&self, fd: i32, size: usize, sync_mode: SyncMode) -> Result<BufferHandle> {
        // SAFETY: `self.delegate` is a valid delegate pointer; function pointer
        // was loaded from the same library that created the delegate.
        let handle =
            unsafe { (self.fns.register)(self.delegate.as_ptr(), fd, size, sync_mode.to_raw()) };
        if handle == kTfLiteNullBufferHandle {
            return Err(Error::null_pointer(
                "VxDelegateRegisterDmaBuf returned null handle",
            ));
        }
        Ok(BufferHandle(handle))
    }

    /// Unregister a previously registered DMA-BUF.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    pub fn unregister(&self, handle: BufferHandle) -> Result<()> {
        // SAFETY: delegate and function pointers are valid; handle was obtained
        // from a prior `register` call on this delegate.
        let status = unsafe { (self.fns.unregister)(self.delegate.as_ptr(), handle.0) };
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
    pub fn request(
        &self,
        tensor_index: i32,
        ownership: Ownership,
        size: usize,
    ) -> Result<(BufferHandle, BufferDesc)> {
        let mut desc = VxDmaBufDesc {
            size,
            ..VxDmaBufDesc::default()
        };
        // SAFETY: delegate pointer is valid; `desc` is a valid mutable reference
        // that the C function will populate with the allocated buffer info.
        let handle = unsafe {
            (self.fns.request)(
                self.delegate.as_ptr(),
                tensor_index,
                ownership.to_raw(),
                &mut desc,
            )
        };
        if handle == kTfLiteNullBufferHandle {
            return Err(Error::null_pointer(
                "VxDelegateRequestDmaBuf returned null handle",
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
    pub fn release(&self, handle: BufferHandle) -> Result<()> {
        // SAFETY: delegate pointer is valid; handle was obtained from `request`.
        let status = unsafe { (self.fns.release)(self.delegate.as_ptr(), handle.0) };
        error::status_to_result(status)
    }

    // --- Tensor Binding ---

    /// Bind a DMA-BUF to a tensor for zero-copy inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    pub fn bind_to_tensor(&self, handle: BufferHandle, tensor_index: i32) -> Result<()> {
        // SAFETY: delegate pointer is valid; handle and tensor_index are caller-provided.
        let status =
            unsafe { (self.fns.bind_to_tensor)(self.delegate.as_ptr(), handle.0, tensor_index) };
        error::status_to_result(status)
    }

    /// Get the file descriptor for a buffer handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns -1.
    pub fn fd(&self, handle: BufferHandle) -> Result<i32> {
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let fd = unsafe { (self.fns.get_fd)(self.delegate.as_ptr(), handle.0) };
        if fd < 0 {
            return Err(Error::invalid_argument(format!(
                "VxDelegateGetDmaBufFd returned {fd} for handle {}",
                handle.0
            )));
        }
        Ok(fd)
    }

    // --- Cache Synchronization ---

    /// Begin CPU access to a DMA-BUF (ensure cache coherency).
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    pub fn begin_cpu_access(&self, handle: BufferHandle, mode: SyncMode) -> Result<()> {
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let status =
            unsafe { (self.fns.begin_cpu_access)(self.delegate.as_ptr(), handle.0, mode.to_raw()) };
        error::status_to_result(status)
    }

    /// End CPU access to a DMA-BUF (flush caches).
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    pub fn end_cpu_access(&self, handle: BufferHandle, mode: SyncMode) -> Result<()> {
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let status =
            unsafe { (self.fns.end_cpu_access)(self.delegate.as_ptr(), handle.0, mode.to_raw()) };
        error::status_to_result(status)
    }

    /// Sync buffer for device (NPU) access.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    pub fn sync_for_device(&self, handle: BufferHandle) -> Result<()> {
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let status = unsafe { (self.fns.sync_for_device)(self.delegate.as_ptr(), handle.0) };
        error::status_to_result(status)
    }

    /// Sync buffer for CPU access.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    pub fn sync_for_cpu(&self, handle: BufferHandle) -> Result<()> {
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let status = unsafe { (self.fns.sync_for_cpu)(self.delegate.as_ptr(), handle.0) };
        error::status_to_result(status)
    }

    // --- Buffer Cycling ---

    /// Set the active DMA-BUF for a tensor (buffer pool cycling).
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returns a non-OK status.
    pub fn set_active(&self, tensor_index: i32, handle: BufferHandle) -> Result<()> {
        // SAFETY: delegate pointer is valid; handle was obtained from register/request.
        let status =
            unsafe { (self.fns.set_active)(self.delegate.as_ptr(), tensor_index, handle.0) };
        error::status_to_result(status)
    }

    /// Get the currently active buffer for a tensor.
    ///
    /// Returns `None` if no buffer is active for this tensor.
    #[must_use]
    pub fn active_buffer(&self, tensor_index: i32) -> Option<BufferHandle> {
        // SAFETY: delegate pointer is valid.
        let handle = unsafe { (self.fns.get_active_buffer)(self.delegate.as_ptr(), tensor_index) };
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
    pub fn invalidate_graph(&self) -> Result<()> {
        // SAFETY: delegate pointer is valid.
        let status = unsafe { (self.fns.invalidate_graph)(self.delegate.as_ptr()) };
        error::status_to_result(status)
    }

    /// Check if the graph has been compiled.
    #[must_use]
    pub fn is_graph_compiled(&self) -> bool {
        // SAFETY: delegate pointer is valid.
        unsafe { (self.fns.is_graph_compiled)(self.delegate.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_mode_clone_copy() {
        let a = SyncMode::Read;
        let b = a;
        let c = a;
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn sync_mode_eq_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(SyncMode::None);
        set.insert(SyncMode::Read);
        set.insert(SyncMode::Write);
        set.insert(SyncMode::ReadWrite);
        assert_eq!(set.len(), 4);

        // Duplicate insert should not increase size.
        set.insert(SyncMode::None);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn sync_mode_inequality() {
        assert_ne!(SyncMode::None, SyncMode::Read);
        assert_ne!(SyncMode::Read, SyncMode::Write);
        assert_ne!(SyncMode::Write, SyncMode::ReadWrite);
        assert_ne!(SyncMode::None, SyncMode::ReadWrite);
    }

    #[test]
    fn sync_mode_to_raw() {
        assert_eq!(SyncMode::None.to_raw(), VxDmaBufSyncMode::None);
        assert_eq!(SyncMode::Read.to_raw(), VxDmaBufSyncMode::Read);
        assert_eq!(SyncMode::Write.to_raw(), VxDmaBufSyncMode::Write);
        assert_eq!(SyncMode::ReadWrite.to_raw(), VxDmaBufSyncMode::ReadWrite);
    }

    #[test]
    fn sync_mode_debug() {
        assert_eq!(format!("{:?}", SyncMode::None), "None");
        assert_eq!(format!("{:?}", SyncMode::Read), "Read");
        assert_eq!(format!("{:?}", SyncMode::Write), "Write");
        assert_eq!(format!("{:?}", SyncMode::ReadWrite), "ReadWrite");
    }

    #[test]
    fn ownership_clone_copy() {
        let a = Ownership::Client;
        let b = a;
        let c = a;
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn ownership_eq_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Ownership::Client);
        set.insert(Ownership::Delegate);
        assert_eq!(set.len(), 2);

        set.insert(Ownership::Client);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn ownership_inequality() {
        assert_ne!(Ownership::Client, Ownership::Delegate);
    }

    #[test]
    fn ownership_to_raw() {
        assert_eq!(Ownership::Client.to_raw(), VxDmaBufOwnership::Client);
        assert_eq!(Ownership::Delegate.to_raw(), VxDmaBufOwnership::Delegate);
    }

    #[test]
    fn ownership_debug() {
        assert_eq!(format!("{:?}", Ownership::Client), "Client");
        assert_eq!(format!("{:?}", Ownership::Delegate), "Delegate");
    }

    #[test]
    fn buffer_handle_raw() {
        let handle = BufferHandle(42);
        assert_eq!(handle.raw(), 42);
    }

    #[test]
    fn buffer_handle_raw_negative() {
        let handle = BufferHandle(-1);
        assert_eq!(handle.raw(), -1);
    }

    #[test]
    fn buffer_handle_equality() {
        let a = BufferHandle(7);
        let b = BufferHandle(7);
        let c = BufferHandle(99);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn buffer_handle_clone_copy() {
        let a = BufferHandle(10);
        let b = a;
        let c = a;
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn buffer_handle_debug() {
        let handle = BufferHandle(5);
        let debug = format!("{handle:?}");
        assert!(debug.contains('5'));
    }

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
        assert!(debug.contains("size"));
        assert!(debug.contains("4096"));
    }
}
