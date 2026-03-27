// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Python `DmaBuf` class wrapping the HAL Delegate DMA-BUF API.

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[allow(deprecated)]
use edgefirst_tflite::dmabuf::{BufferHandle, Ownership, SyncMode};

use crate::error::{self, InvalidArgumentError, TfLiteError};
use crate::interpreter::PyInterpreter;

/// Parse a sync mode string into the Rust enum.
#[allow(deprecated)]
fn parse_sync_mode(s: &str) -> PyResult<SyncMode> {
    match s {
        "none" => Ok(SyncMode::None),
        "read" => Ok(SyncMode::Read),
        "write" => Ok(SyncMode::Write),
        "readwrite" | "read_write" => Ok(SyncMode::ReadWrite),
        _ => Err(InvalidArgumentError::new_err(format!(
            "invalid sync_mode: {s:?} (expected: none, read, write, readwrite)"
        ))),
    }
}

/// Parse an ownership string into the Rust enum.
#[allow(deprecated)]
fn parse_ownership(s: &str) -> PyResult<Ownership> {
    match s {
        "client" => Ok(Ownership::Client),
        "delegate" => Ok(Ownership::Delegate),
        _ => Err(InvalidArgumentError::new_err(format!(
            "invalid ownership: {s:?} (expected: client, delegate)"
        ))),
    }
}

/// DMA-BUF zero-copy interface for `TFLite` delegates.
///
/// Provides access to the HAL Delegate DMA-BUF API for querying tensor
/// DMA-BUF metadata and synchronizing cache coherency. Legacy `VxDelegate`
/// methods are available as deprecated fallbacks.
///
/// Obtained via `interp.delegate(0)` and then accessing DMA-BUF methods.
/// This class borrows the delegate from the interpreter — the interpreter
/// must remain alive.
#[pyclass(name = "DmaBuf", unsendable)]
pub struct PyDmaBuf {
    pub(crate) interp: Py<PyInterpreter>,
    pub(crate) delegate_index: usize,
}

impl std::fmt::Debug for PyDmaBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DmaBuf")
            .field("delegate_index", &self.delegate_index)
            .finish_non_exhaustive()
    }
}

impl PyDmaBuf {
    /// Helper: borrow the interpreter, get the delegate, get `DmaBuf`, call `f`.
    fn with_dmabuf<F, R>(&self, py: Python<'_>, f: F) -> PyResult<R>
    where
        F: FnOnce(&edgefirst_tflite::dmabuf::DmaBuf<'_>) -> PyResult<R>,
    {
        let interp = self.interp.bind(py).borrow();
        interp.with_delegate(self.delegate_index, |delegate| {
            let dmabuf = delegate
                .dmabuf()
                .ok_or_else(|| TfLiteError::new_err("DmaBuf not available on this delegate"))?;
            f(&dmabuf)
        })
    }
}

#[pymethods]
impl PyDmaBuf {
    // =======================================================================
    // Primary API (HAL Delegate DMA-BUF)
    // =======================================================================

    /// Check if DMA-BUF zero-copy is supported.
    fn is_supported(&self, py: Python<'_>) -> PyResult<bool> {
        self.with_dmabuf(py, |d| Ok(d.is_supported()))
    }

    /// Get DMA-BUF tensor information for a given tensor index.
    ///
    /// Returns a dict with keys: ``fd``, ``size``, ``offset``, ``shape``, ``dtype``.
    /// The ``fd`` is borrowed from the delegate and must NOT be closed.
    ///
    /// Requires the HAL Delegate DMA-BUF API.
    fn tensor_info<'py>(&self, py: Python<'py>, tensor_index: i32) -> PyResult<Bound<'py, PyDict>> {
        self.with_dmabuf(py, |d| {
            let info = d.tensor_info(tensor_index).map_err(error::to_py_err)?;
            let dict = PyDict::new(py);
            dict.set_item("fd", info.fd)?;
            dict.set_item("size", info.size)?;
            dict.set_item("offset", info.offset)?;
            dict.set_item("shape", info.shape)?;
            dict.set_item("dtype", info.dtype.to_string())?;
            Ok(dict)
        })
    }

    /// Sync tensor buffer for device (NPU) access by tensor index.
    ///
    /// Flushes CPU caches so the device can read the buffer contents.
    fn sync_for_device(&self, py: Python<'_>, tensor_index: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.sync_for_device(tensor_index).map_err(error::to_py_err)
        })
    }

    /// Sync tensor buffer for CPU access by tensor index.
    ///
    /// Invalidates CPU caches so the CPU sees device-written data.
    fn sync_for_cpu(&self, py: Python<'_>, tensor_index: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.sync_for_cpu(tensor_index).map_err(error::to_py_err)
        })
    }

    // =======================================================================
    // Legacy VxDelegate API (deprecated)
    // =======================================================================

    /// Register an externally-allocated DMA-BUF.
    ///
    /// Deprecated: `VxDelegate`-specific, will be removed in a future release.
    #[allow(deprecated)]
    #[pyo3(signature = (fd, size, sync_mode="none"))]
    fn register(&self, py: Python<'_>, fd: i32, size: usize, sync_mode: &str) -> PyResult<i32> {
        let mode = parse_sync_mode(sync_mode)?;
        self.with_dmabuf(py, |d| {
            let handle = d.register(fd, size, mode).map_err(error::to_py_err)?;
            Ok(handle.raw())
        })
    }

    /// Unregister a previously registered DMA-BUF.
    ///
    /// Deprecated: `VxDelegate`-specific, will be removed in a future release.
    #[allow(deprecated)]
    fn unregister(&self, py: Python<'_>, handle: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.unregister(BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Request the delegate to allocate a DMA-BUF for a tensor.
    ///
    /// Returns (handle, `desc_dict`) where `desc_dict` has keys: fd, size, `map_ptr`.
    ///
    /// Deprecated: `VxDelegate`-specific, will be removed in a future release.
    #[allow(deprecated)]
    #[pyo3(signature = (tensor_index, ownership="client", size=0))]
    fn request<'py>(
        &self,
        py: Python<'py>,
        tensor_index: i32,
        ownership: &str,
        size: usize,
    ) -> PyResult<(i32, Bound<'py, PyDict>)> {
        let own = parse_ownership(ownership)?;
        self.with_dmabuf(py, |d| {
            let (handle, desc) = d
                .request(tensor_index, own, size)
                .map_err(error::to_py_err)?;
            let dict = PyDict::new(py);
            dict.set_item("fd", desc.fd)?;
            dict.set_item("size", desc.size)?;
            dict.set_item("map_ptr", desc.map_ptr.map_or(0usize, |p| p as usize))?;
            Ok((handle.raw(), dict))
        })
    }

    /// Release a delegate-allocated DMA-BUF.
    ///
    /// Deprecated: `VxDelegate`-specific, will be removed in a future release.
    #[allow(deprecated)]
    fn release(&self, py: Python<'_>, handle: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.release(BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Bind a DMA-BUF to a tensor for zero-copy inference.
    ///
    /// Deprecated: `VxDelegate`-specific, will be removed in a future release.
    #[allow(deprecated)]
    fn bind_to_tensor(&self, py: Python<'_>, handle: i32, tensor_index: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.bind_to_tensor(BufferHandle::from_raw(handle), tensor_index)
                .map_err(error::to_py_err)
        })
    }

    /// Get the file descriptor for a buffer handle.
    ///
    /// Deprecated: `VxDelegate`-specific, use `tensor_info()` instead.
    #[allow(deprecated)]
    fn fd(&self, py: Python<'_>, handle: i32) -> PyResult<i32> {
        self.with_dmabuf(py, |d| {
            d.buffer_fd(BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Begin CPU access to a DMA-BUF.
    ///
    /// Deprecated: `VxDelegate`-specific, use `sync_for_cpu()` instead.
    #[allow(deprecated)]
    #[pyo3(signature = (handle, mode="read"))]
    fn begin_cpu_access(&self, py: Python<'_>, handle: i32, mode: &str) -> PyResult<()> {
        let sync = parse_sync_mode(mode)?;
        self.with_dmabuf(py, |d| {
            d.begin_cpu_access(BufferHandle::from_raw(handle), sync)
                .map_err(error::to_py_err)
        })
    }

    /// End CPU access to a DMA-BUF.
    ///
    /// Deprecated: `VxDelegate`-specific, use `sync_for_device()` instead.
    #[allow(deprecated)]
    #[pyo3(signature = (handle, mode="read"))]
    fn end_cpu_access(&self, py: Python<'_>, handle: i32, mode: &str) -> PyResult<()> {
        let sync = parse_sync_mode(mode)?;
        self.with_dmabuf(py, |d| {
            d.end_cpu_access(BufferHandle::from_raw(handle), sync)
                .map_err(error::to_py_err)
        })
    }

    /// Sync buffer for device (NPU) access by buffer handle.
    ///
    /// Deprecated: `VxDelegate`-specific, use `sync_for_device(tensor_index)` instead.
    #[allow(deprecated)]
    fn sync_for_device_by_handle(&self, py: Python<'_>, handle: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.sync_for_device_by_handle(BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Sync buffer for CPU access by buffer handle.
    ///
    /// Deprecated: `VxDelegate`-specific, use `sync_for_cpu(tensor_index)` instead.
    #[allow(deprecated)]
    fn sync_for_cpu_by_handle(&self, py: Python<'_>, handle: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.sync_for_cpu_by_handle(BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Set the active DMA-BUF for a tensor (buffer pool cycling).
    ///
    /// Deprecated: `VxDelegate`-specific, will be removed in a future release.
    #[allow(deprecated)]
    fn set_active(&self, py: Python<'_>, tensor_index: i32, handle: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.set_active(tensor_index, BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Get the currently active buffer for a tensor.
    ///
    /// Deprecated: `VxDelegate`-specific, will be removed in a future release.
    #[allow(deprecated)]
    fn active_buffer(&self, py: Python<'_>, tensor_index: i32) -> PyResult<Option<i32>> {
        self.with_dmabuf(py, |d| {
            Ok(d.active_buffer(tensor_index).map(BufferHandle::raw))
        })
    }

    /// Invalidate the compiled graph (forces recompilation on next invoke).
    ///
    /// Deprecated: `VxDelegate`-specific, will be removed in a future release.
    #[allow(deprecated)]
    fn invalidate_graph(&self, py: Python<'_>) -> PyResult<()> {
        self.with_dmabuf(py, |d| d.invalidate_graph().map_err(error::to_py_err))
    }

    /// Check if the graph has been compiled.
    ///
    /// Deprecated: `VxDelegate`-specific, will be removed in a future release.
    #[allow(deprecated)]
    fn is_graph_compiled(&self, py: Python<'_>) -> PyResult<bool> {
        self.with_dmabuf(py, |d| Ok(d.is_graph_compiled()))
    }
}
