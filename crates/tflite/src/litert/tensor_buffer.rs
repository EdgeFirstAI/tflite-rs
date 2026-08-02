// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! `LiteRT` tensor buffers and the buffer requirements that describe them.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::slice;

use edgefirst_tflite_sys::litert::{
    kLiteRtTensorBufferLockModeRead, kLiteRtTensorBufferLockModeReadWrite,
    kLiteRtTensorBufferLockModeWrite, kLiteRtTensorBufferTypeHostMemory, LiteRtFunctions,
    RankedTensorType, TensorBuffer as RawBuffer, TensorBufferLockMode,
    TensorBufferRequirements as RawReqs, TensorBufferType,
};

use crate::error::{litert_status_to_result, Error, Result};
use crate::litert::{CompiledModel, Environment};

/// How a compiled model needs one of its tensors to be backed.
///
/// Different accelerators want different memory: an NPU may require DMA-able
/// or device-local memory, while the CPU path is happy with ordinary host
/// memory. Rather than guessing, ask the compiled model
/// ([`CompiledModel::input_buffer_requirements`]) and hand the answer to
/// [`TensorBuffer::managed_from_requirements`], which allocates memory the
/// accelerator can actually reach.
///
/// # Lifetime
///
/// `BufferRequirements<'cm>` **borrows** the [`CompiledModel`] it came from.
/// The C API is explicit that the underlying object is *"still owned by the
/// `LiteRtCompiledModel` and is only valid during \[its\] lifetime"*, so this
/// type carries that lifetime rather than being a free-standing value. Using
/// requirements after their compiled model has been dropped is a use-after-free
/// — and one the borrow checker now rejects at compile time:
///
/// ```compile_fail
/// # use edgefirst_tflite::{litert, Library};
/// # let lib = Library::new().unwrap();
/// # let env = litert::Environment::new(&lib).unwrap();
/// # let model = litert::Model::from_file(&env, "m.tflite").unwrap();
/// # let opts = litert::Options::new(&lib).unwrap();
/// let reqs = {
///     let compiled = litert::CompiledModel::create(&env, &model, &opts).unwrap();
///     compiled.input_buffer_requirements(0, 0).unwrap()
/// }; // `compiled` dropped here — `reqs` would dangle
/// println!("{}", reqs.size());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BufferRequirements<'cm> {
    buffer_type: TensorBufferType,
    size: usize,
    raw: RawReqs,
    /// Ties these requirements to the compiled model that owns them.
    _compiled: PhantomData<&'cm CompiledModel<'cm>>,
}

impl BufferRequirements<'_> {
    /// The preferred backing store for this tensor.
    ///
    /// This is the first entry of the runtime's supported-type list, which the
    /// C API orders by preference. Compare against
    /// `edgefirst_tflite_sys::litert::kLiteRtTensorBufferTypeHostMemory` and
    /// friends.
    #[must_use]
    pub const fn buffer_type(self) -> TensorBufferType {
        self.buffer_type
    }

    /// The minimum buffer size in bytes required for this tensor.
    ///
    /// A buffer allocated from these requirements may be *larger* than this —
    /// backends can add alignment padding. Use [`TensorBuffer::size`] for the
    /// size actually allocated.
    #[must_use]
    pub const fn size(self) -> usize {
        self.size
    }

    /// Query the requirements object behind a raw handle.
    ///
    /// # Safety contract
    ///
    /// The returned value borrows `'cm`; callers must only construct it from a
    /// pointer obtained from a `CompiledModel` living at least that long.
    pub(crate) fn from_raw(fns: &LiteRtFunctions, raw: RawReqs) -> Result<Self> {
        let mut num_types = 0;
        // SAFETY: `raw` is a live requirements handle owned by a compiled
        // model; `num_types` is a valid out-parameter.
        let status = unsafe {
            (fns.get_num_tensor_buffer_requirements_supported_buffer_types)(raw, &raw mut num_types)
        };
        litert_status_to_result(status).map_err(|e| {
            e.with_context("LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes")
        })?;
        if num_types <= 0 {
            return Err(Error::invalid_argument(
                "buffer requirements list zero supported types",
            ));
        }

        let mut buffer_type = kLiteRtTensorBufferTypeHostMemory;
        // SAFETY: index 0 is in range because `num_types > 0` was just checked.
        let status = unsafe {
            (fns.get_tensor_buffer_requirements_supported_tensor_buffer_type)(
                raw,
                0,
                &raw mut buffer_type,
            )
        };
        litert_status_to_result(status).map_err(|e| {
            e.with_context("LiteRtGetTensorBufferRequirementsSupportedTensorBufferType")
        })?;

        let mut size = 0usize;
        // SAFETY: `raw` is live; `size` is a valid out-parameter.
        let status =
            unsafe { (fns.get_tensor_buffer_requirements_buffer_size)(raw, &raw mut size) };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetTensorBufferRequirementsBufferSize"))?;

        Ok(Self {
            buffer_type,
            size,
            raw,
            _compiled: PhantomData,
        })
    }

    pub(crate) fn as_raw(self) -> RawReqs {
        self.raw
    }
}

/// How a [`TensorBuffer`] is mapped into host address space.
///
/// The mode is a hint to the runtime about whether existing contents need to be
/// made visible to the host (`Read`) and whether host writes need to be flushed
/// back to the device (`Write`). Choosing the narrowest mode that fits avoids
/// unnecessary cache maintenance on accelerators with non-coherent memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorBufferLock {
    /// Host reads only; device contents are made visible.
    Read,
    /// Host writes only; existing contents need not be fetched.
    Write,
    /// Host reads and writes.
    ReadWrite,
}

impl TensorBufferLock {
    fn to_raw(self) -> TensorBufferLockMode {
        match self {
            Self::Read => kLiteRtTensorBufferLockModeRead,
            Self::Write => kLiteRtTensorBufferLockModeWrite,
            Self::ReadWrite => kLiteRtTensorBufferLockModeReadWrite,
        }
    }
}

/// An active host mapping of a tensor buffer.
///
/// Unlocks on drop, so an early return or a panic between lock and unlock
/// cannot leave the buffer mapped. Call [`Mapping::unlock`] to observe the
/// unlock status instead of discarding it.
struct Mapping<'a> {
    fns: &'a LiteRtFunctions,
    ptr: RawBuffer,
    host: *mut u8,
    released: bool,
}

impl<'a> Mapping<'a> {
    /// Lock `ptr` and return the mapping, guaranteeing an unlock on every exit
    /// path once this returns `Ok`.
    fn lock(fns: &'a LiteRtFunctions, ptr: RawBuffer, mode: TensorBufferLock) -> Result<Self> {
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a live tensor buffer; `host` is a valid
        // out-parameter for the mapped address.
        let status = unsafe { (fns.lock_tensor_buffer)(ptr, &raw mut host, mode.to_raw()) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtLockTensorBuffer"))?;

        // The lock succeeded, so the buffer is mapped from here on: build the
        // guard *before* validating `host` so that the null-pointer path below
        // still unlocks.
        let mapping = Self {
            fns,
            ptr,
            host: host.cast::<u8>(),
            released: false,
        };
        if mapping.host.is_null() {
            return Err(Error::null_pointer(
                "LiteRtLockTensorBuffer returned a null host pointer",
            ));
        }
        Ok(mapping)
    }

    /// Release the mapping, propagating any unlock failure.
    fn unlock(mut self) -> Result<()> {
        self.released = true;
        // SAFETY: the buffer was locked by `Self::lock` and is unlocked exactly
        // once — `released` is set before the call and checked by `Drop`.
        let status = unsafe { (self.fns.unlock_tensor_buffer)(self.ptr) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtUnlockTensorBuffer"))
    }
}

impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        if !self.released {
            // SAFETY: as in `unlock`. The status is unobservable on this path
            // (we are already unwinding or returning an error), but unlocking
            // still has to happen or the buffer stays mapped forever.
            unsafe {
                let _ = (self.fns.unlock_tensor_buffer)(self.ptr);
            }
        }
    }
}

/// An owned `LiteRT` tensor buffer — the memory inference reads from and writes
/// to.
///
/// `LiteRT` decouples buffers from the model so that the runtime can place them
/// where the target accelerator can reach them. Obtain buffers the easy way,
/// via [`CompiledModel::create_input_buffer`] /
/// [`CompiledModel::create_output_buffer`], which query the model's
/// requirements for you.
///
/// Host access goes through [`TensorBuffer::write_bytes`] and
/// [`TensorBuffer::read_bytes`]. Both lock the buffer, copy, and unlock; the
/// mapping is never exposed, so there is no way to hold a stale pointer.
///
/// # Examples
///
/// ```no_run
/// # use edgefirst_tflite::{litert, Library};
/// # let lib = Library::new()?;
/// # let env = litert::Environment::new(&lib)?;
/// # let model = litert::Model::from_file(&env, "model.tflite")?;
/// # let opts = litert::Options::new(&lib)?;
/// # let mut compiled = litert::CompiledModel::create(&env, &model, &opts)?;
/// let mut input = compiled.create_input_buffer(0, 0)?;
/// input.write_bytes(&vec![0u8; input.size()])?;
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
///
/// # Thread safety
///
/// `TensorBuffer` is [`Send`] but not [`Sync`]. Locking mutates buffer state,
/// and the methods that lock take `&mut self`, so exclusive access is required
/// for host mapping — but a buffer can be moved to whichever thread runs
/// inference.
#[derive(Debug)]
pub struct TensorBuffer<'env> {
    ptr: RawBuffer,
    fns: &'env LiteRtFunctions,
    /// Size actually allocated, read back from `LiteRtGetTensorBufferSize`
    /// rather than assumed from the requested size. This is the bound every
    /// host mapping is sliced with, so it must reflect the real allocation.
    size: usize,
}

impl<'env> TensorBuffer<'env> {
    /// Allocate a runtime-managed buffer satisfying `requirements`.
    ///
    /// This is the allocation path that respects accelerator placement — the
    /// runtime chooses the backing store named by
    /// [`BufferRequirements::buffer_type`].
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if allocation fails or the requirements cannot
    /// be satisfied on this device.
    pub fn managed_from_requirements(
        env: &'env Environment<'_>,
        tensor_type: &RankedTensorType,
        requirements: BufferRequirements<'_>,
    ) -> Result<Self> {
        let fns = env.functions();
        let mut buf: RawBuffer = std::ptr::null_mut();
        // SAFETY: `env` is live; `tensor_type` outlives the call; the
        // requirements handle is live because `BufferRequirements` borrows the
        // compiled model that owns it.
        let status = unsafe {
            (fns.create_managed_tensor_buffer_from_requirements)(
                env.as_raw(),
                tensor_type,
                requirements.as_raw(),
                &raw mut buf,
            )
        };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtCreateManagedTensorBufferFromRequirements"))?;
        Self::from_raw(fns, buf, "LiteRtCreateManagedTensorBufferFromRequirements")
    }

    /// Allocate a runtime-managed buffer in ordinary host memory.
    ///
    /// Use this when you need a plain CPU-visible buffer of a known size.
    /// Prefer [`TensorBuffer::managed_from_requirements`] for buffers that will
    /// be handed to a compiled model, since host memory may not be reachable by
    /// the selected accelerator.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if allocation fails.
    pub fn managed_host(
        env: &'env Environment<'_>,
        tensor_type: &RankedTensorType,
        size: usize,
    ) -> Result<Self> {
        let fns = env.functions();
        let mut buf: RawBuffer = std::ptr::null_mut();
        // SAFETY: `env` is live and `tensor_type` outlives the call; `buf` is a
        // valid out-parameter.
        let status = unsafe {
            (fns.create_managed_tensor_buffer)(
                env.as_raw(),
                kLiteRtTensorBufferTypeHostMemory,
                tensor_type,
                size,
                &raw mut buf,
            )
        };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtCreateManagedTensorBuffer"))?;
        Self::from_raw(fns, buf, "LiteRtCreateManagedTensorBuffer")
    }

    /// Wrap a freshly created raw buffer, reading back its true size.
    fn from_raw(fns: &'env LiteRtFunctions, ptr: RawBuffer, context: &'static str) -> Result<Self> {
        if ptr.is_null() {
            return Err(Error::null_pointer(format!("{context} returned null")));
        }
        // Take ownership before anything else can fail, so the `?` below
        // destroys the buffer instead of leaking it.
        //
        // NOTE: mutate `buffer` in place and move it out. Do *not* build a
        // second value with struct-update syntax (`Self { size, ..buffer }`):
        // every field here is `Copy`, so that duplicates the owning handle
        // rather than moving it, and the original's `Drop` then frees a buffer
        // the caller still holds.
        let mut buffer = Self { ptr, fns, size: 0 };

        // Ask the runtime for the size it actually allocated rather than
        // trusting the size we requested: backends are free to pad, and this
        // value bounds every host mapping we hand out.
        let mut size = 0usize;
        // SAFETY: `ptr` is a live buffer (null-checked above) owned by
        // `buffer`; `size` is a valid out-parameter.
        let status = unsafe { (fns.get_tensor_buffer_size)(ptr, &raw mut size) };
        litert_status_to_result(status).map_err(|e| e.with_context("LiteRtGetTensorBufferSize"))?;

        buffer.size = size;
        Ok(buffer)
    }

    /// The buffer's allocated size in bytes, as reported by the runtime.
    ///
    /// May exceed [`BufferRequirements::size`] when the backend pads for
    /// alignment. This is the exact number of bytes
    /// [`TensorBuffer::read_bytes`] returns and the maximum
    /// [`TensorBuffer::write_bytes`] accepts.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// The element type and shape the runtime associates with this buffer.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if the buffer cannot be queried.
    pub fn tensor_type(&self) -> Result<RankedTensorType> {
        // SAFETY: `LiteRtRankedTensorType` is plain data, so zeroed is a valid
        // bit pattern for an out-parameter.
        let mut ty = unsafe { std::mem::zeroed::<RankedTensorType>() };
        // SAFETY: `self.ptr` is a live buffer; `ty` is a valid out-parameter.
        let status = unsafe { (self.fns.get_tensor_buffer_tensor_type)(self.ptr, &raw mut ty) };
        litert_status_to_result(status)
            .map_err(|e| e.with_context("LiteRtGetTensorBufferTensorType"))?;
        Ok(ty)
    }

    /// Copy `src` into the buffer.
    ///
    /// Takes a write-only lock, so existing contents are not fetched from the
    /// device. `src` may be shorter than [`TensorBuffer::size`]; bytes beyond
    /// `src.len()` are left as-is and should be treated as undefined.
    ///
    /// # Errors
    ///
    /// - [`Error::is_invalid_argument`](crate::Error::is_invalid_argument) if
    ///   `src` is longer than the buffer.
    /// - A `LiteRT` status error if the buffer cannot be locked or unlocked.
    pub fn write_bytes(&mut self, src: &[u8]) -> Result<()> {
        if src.len() > self.size {
            return Err(Error::invalid_argument(format!(
                "write of {} bytes exceeds buffer size {}",
                src.len(),
                self.size
            )));
        }
        let mapping = Mapping::lock(self.fns, self.ptr, TensorBufferLock::Write)?;
        // SAFETY: the lock returned a host mapping of at least `self.size`
        // writable bytes (`self.size` came from `LiteRtGetTensorBufferSize`),
        // and `src.len() <= self.size` was checked above. The regions cannot
        // overlap: `src` is Rust-owned memory, the mapping is runtime-owned.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), mapping.host, src.len());
        }
        mapping.unlock()
    }

    /// Copy the buffer's contents into `dst`.
    ///
    /// Reads exactly `dst.len()` bytes from the start of the buffer, so a
    /// caller that knows the logical tensor size can avoid reading alignment
    /// padding.
    ///
    /// # Errors
    ///
    /// - [`Error::is_invalid_argument`](crate::Error::is_invalid_argument) if
    ///   `dst` is longer than the buffer.
    /// - A `LiteRT` status error if the buffer cannot be locked or unlocked.
    pub fn read_bytes_into(&mut self, dst: &mut [u8]) -> Result<()> {
        if dst.len() > self.size {
            return Err(Error::invalid_argument(format!(
                "read of {} bytes exceeds buffer size {}",
                dst.len(),
                self.size
            )));
        }
        let mapping = Mapping::lock(self.fns, self.ptr, TensorBufferLock::Read)?;
        // SAFETY: the lock returned a host mapping of at least `self.size`
        // readable bytes, and `dst.len() <= self.size`. `dst` is Rust-owned and
        // cannot overlap the runtime's mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(mapping.host, dst.as_mut_ptr(), dst.len());
        }
        mapping.unlock()
    }

    /// Copy the whole buffer into a new `Vec`.
    ///
    /// Returns exactly [`TensorBuffer::size`] bytes, which may include
    /// alignment padding beyond the logical tensor. Use
    /// [`TensorBuffer::read_bytes_into`] when you know the logical size.
    ///
    /// # Errors
    ///
    /// A `LiteRT` status error if the buffer cannot be locked or unlocked.
    pub fn read_bytes(&mut self) -> Result<Vec<u8>> {
        let mapping = Mapping::lock(self.fns, self.ptr, TensorBufferLock::Read)?;
        // SAFETY: the lock returned a readable host mapping of at least
        // `self.size` bytes, which the runtime reported as the allocation size.
        // The slice lives only until `to_vec` returns, well inside the lock.
        let data = unsafe { slice::from_raw_parts(mapping.host.cast_const(), self.size).to_vec() };
        mapping.unlock()?;
        Ok(data)
    }

    pub(crate) fn as_raw(&self) -> RawBuffer {
        self.ptr
    }
}

// SAFETY: `TensorBuffer` owns a `LiteRtTensorBuffer`, an opaque heap handle
// with no thread affinity. Every host mapping is created and released inside a
// single `&mut self` method, so no mapping can be in flight across a move.
unsafe impl Send for TensorBuffer<'_> {}

// Deliberately NOT `Sync`: locking mutates buffer state, so shared concurrent
// access would be unsound. The locking methods take `&mut self`.

impl Drop for TensorBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from a `LiteRtCreate*TensorBuffer` call, was
        // null-checked in `from_raw`, and is destroyed exactly once. No mapping
        // can be outstanding: `Mapping` unlocks on drop and never escapes the
        // method that created it.
        unsafe {
            (self.fns.destroy_tensor_buffer)(self.ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_modes_map_to_distinct_raw_values() {
        assert_eq!(
            TensorBufferLock::Read.to_raw(),
            kLiteRtTensorBufferLockModeRead
        );
        assert_eq!(
            TensorBufferLock::Write.to_raw(),
            kLiteRtTensorBufferLockModeWrite
        );
        assert_eq!(
            TensorBufferLock::ReadWrite.to_raw(),
            kLiteRtTensorBufferLockModeReadWrite
        );
        assert_ne!(
            TensorBufferLock::Read.to_raw(),
            TensorBufferLock::Write.to_raw()
        );
    }
}
