// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Python `DmaBuf` class wrapping the `VxDelegate` DMA-BUF zero-copy API.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use edgefirst_tflite::dmabuf::{BufferHandle, Ownership, SyncMode};

use crate::error::{self, InvalidArgumentError, TfLiteError};
use crate::interpreter::PyInterpreter;

/// Parse a sync mode string into the Rust enum.
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
fn parse_ownership(s: &str) -> PyResult<Ownership> {
    match s {
        "client" => Ok(Ownership::Client),
        "delegate" => Ok(Ownership::Delegate),
        _ => Err(InvalidArgumentError::new_err(format!(
            "invalid ownership: {s:?} (expected: client, delegate)"
        ))),
    }
}

/// DMA-BUF zero-copy interface for `VxDelegate`.
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
    /// Check if DMA-BUF zero-copy is supported.
    fn is_supported(&self, py: Python<'_>) -> PyResult<bool> {
        self.with_dmabuf(py, |d| Ok(d.is_supported()))
    }

    /// Register an externally-allocated DMA-BUF.
    #[pyo3(signature = (fd, size, sync_mode="none"))]
    fn register(&self, py: Python<'_>, fd: i32, size: usize, sync_mode: &str) -> PyResult<i32> {
        let mode = parse_sync_mode(sync_mode)?;
        self.with_dmabuf(py, |d| {
            let handle = d.register(fd, size, mode).map_err(error::to_py_err)?;
            Ok(handle.raw())
        })
    }

    /// Unregister a previously registered DMA-BUF.
    fn unregister(&self, py: Python<'_>, handle: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.unregister(BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Request the delegate to allocate a DMA-BUF for a tensor.
    ///
    /// Returns (handle, `desc_dict`) where `desc_dict` has keys: fd, size, `map_ptr`.
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
    fn release(&self, py: Python<'_>, handle: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.release(BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Bind a DMA-BUF to a tensor for zero-copy inference.
    fn bind_to_tensor(&self, py: Python<'_>, handle: i32, tensor_index: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.bind_to_tensor(BufferHandle::from_raw(handle), tensor_index)
                .map_err(error::to_py_err)
        })
    }

    /// Get the file descriptor for a buffer handle.
    fn fd(&self, py: Python<'_>, handle: i32) -> PyResult<i32> {
        self.with_dmabuf(py, |d| {
            d.fd(BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Begin CPU access to a DMA-BUF.
    #[pyo3(signature = (handle, mode="read"))]
    fn begin_cpu_access(&self, py: Python<'_>, handle: i32, mode: &str) -> PyResult<()> {
        let sync = parse_sync_mode(mode)?;
        self.with_dmabuf(py, |d| {
            d.begin_cpu_access(BufferHandle::from_raw(handle), sync)
                .map_err(error::to_py_err)
        })
    }

    /// End CPU access to a DMA-BUF.
    #[pyo3(signature = (handle, mode="read"))]
    fn end_cpu_access(&self, py: Python<'_>, handle: i32, mode: &str) -> PyResult<()> {
        let sync = parse_sync_mode(mode)?;
        self.with_dmabuf(py, |d| {
            d.end_cpu_access(BufferHandle::from_raw(handle), sync)
                .map_err(error::to_py_err)
        })
    }

    /// Sync buffer for device (NPU) access.
    fn sync_for_device(&self, py: Python<'_>, handle: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.sync_for_device(BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Sync buffer for CPU access.
    fn sync_for_cpu(&self, py: Python<'_>, handle: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.sync_for_cpu(BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Set the active DMA-BUF for a tensor (buffer pool cycling).
    fn set_active(&self, py: Python<'_>, tensor_index: i32, handle: i32) -> PyResult<()> {
        self.with_dmabuf(py, |d| {
            d.set_active(tensor_index, BufferHandle::from_raw(handle))
                .map_err(error::to_py_err)
        })
    }

    /// Get the currently active buffer for a tensor.
    fn active_buffer(&self, py: Python<'_>, tensor_index: i32) -> PyResult<Option<i32>> {
        self.with_dmabuf(py, |d| {
            Ok(d.active_buffer(tensor_index).map(BufferHandle::raw))
        })
    }

    /// Invalidate the compiled graph (forces recompilation on next invoke).
    fn invalidate_graph(&self, py: Python<'_>) -> PyResult<()> {
        self.with_dmabuf(py, |d| d.invalidate_graph().map_err(error::to_py_err))
    }

    /// Check if the graph has been compiled.
    fn is_graph_compiled(&self, py: Python<'_>) -> PyResult<bool> {
        self.with_dmabuf(py, |d| Ok(d.is_graph_compiled()))
    }
}
