// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Op-level profiler using the `TFLite` telemetry profiler C API.
//!
//! The [`Profiler`] collects per-operator timing events during inference.
//! Attach it to an [`InterpreterBuilder`](crate::InterpreterBuilder) before
//! building, then read events after [`Interpreter::invoke`](crate::Interpreter::invoke).
//!
//! # Example
//!
//! ```no_run
//! use edgefirst_tflite::{Library, Model, Interpreter, Profiler};
//!
//! let lib = Library::new()?;
//! let model = Model::from_file(&lib, "model.tflite")?;
//! let profiler = Profiler::new();
//!
//! let mut interp = Interpreter::builder(&lib)?
//!     .profiler(&profiler)?
//!     .build(&model)?;
//!
//! interp.invoke()?;
//!
//! for event in profiler.events() {
//!     println!("{}: {}us (op={}, subgraph={})",
//!         event.op_name, event.duration_us, event.op_idx, event.subgraph_idx);
//! }
//! # Ok::<(), edgefirst_tflite::Error>(())
//! ```

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

// ---------------------------------------------------------------------------
// OpEvent
// ---------------------------------------------------------------------------

/// A recorded per-op timing event from a single inference invocation.
#[derive(Debug, Clone)]
pub struct OpEvent {
    /// Operator name (e.g., `NeutronDelegate`, `SOFTMAX`, `Transpose`).
    pub op_name: String,
    /// Operator index in the subgraph.
    pub op_idx: i64,
    /// Subgraph index.
    pub subgraph_idx: i64,
    /// Duration in microseconds.
    pub duration_us: u64,
}

// ---------------------------------------------------------------------------
// TfLiteTelemetryProfilerStruct (C ABI)
// ---------------------------------------------------------------------------

/// C-compatible telemetry profiler struct matching
/// `TfLiteTelemetryProfilerStruct` from
/// `tensorflow/lite/profiling/telemetry/c/profiler.h`.
///
/// The `data` field points to user-owned state. Function pointers are
/// called by the `TFLite` runtime during inference to report events.
#[repr(C)]
struct TfLiteTelemetryProfilerStruct {
    data: *mut c_void,

    report_telemetry_event: Option<
        unsafe extern "C" fn(
            profiler: *mut TfLiteTelemetryProfilerStruct,
            event_name: *const c_char,
            status: u64,
        ),
    >,

    report_telemetry_op_event: Option<
        unsafe extern "C" fn(
            profiler: *mut TfLiteTelemetryProfilerStruct,
            event_name: *const c_char,
            op_idx: i64,
            subgraph_idx: i64,
            status: u64,
        ),
    >,

    report_settings: Option<
        unsafe extern "C" fn(
            profiler: *mut TfLiteTelemetryProfilerStruct,
            setting_name: *const c_char,
            settings: *const c_void,
        ),
    >,

    report_begin_op_invoke_event: Option<
        unsafe extern "C" fn(
            profiler: *mut TfLiteTelemetryProfilerStruct,
            op_name: *const c_char,
            op_idx: i64,
            subgraph_idx: i64,
        ) -> u32,
    >,

    report_end_op_invoke_event: Option<
        unsafe extern "C" fn(profiler: *mut TfLiteTelemetryProfilerStruct, event_handle: u32),
    >,

    report_op_invoke_event: Option<
        unsafe extern "C" fn(
            profiler: *mut TfLiteTelemetryProfilerStruct,
            op_name: *const c_char,
            elapsed_time: u64,
            op_idx: i64,
            subgraph_idx: i64,
        ),
    >,
}

// ---------------------------------------------------------------------------
// C callbacks
// ---------------------------------------------------------------------------

/// No-op callback for `ReportTelemetryEvent`.
unsafe extern "C" fn report_telemetry_event_noop(
    _profiler: *mut TfLiteTelemetryProfilerStruct,
    _event_name: *const c_char,
    _status: u64,
) {
}

/// No-op callback for `ReportTelemetryOpEvent`.
unsafe extern "C" fn report_telemetry_op_event_noop(
    _profiler: *mut TfLiteTelemetryProfilerStruct,
    _event_name: *const c_char,
    _op_idx: i64,
    _subgraph_idx: i64,
    _status: u64,
) {
}

/// No-op callback for `ReportSettings`.
unsafe extern "C" fn report_settings_noop(
    _profiler: *mut TfLiteTelemetryProfilerStruct,
    _setting_name: *const c_char,
    _settings: *const c_void,
) {
}

/// Recover an `&Arc<Mutex<ProfilerInner>>` from the C struct's `data` field.
///
/// # Safety
///
/// The `data` field of `*profiler` must point to a live, heap-allocated
/// `Arc<Mutex<ProfilerInner>>` (created via `Box::into_raw` in
/// [`Profiler::new`]).
unsafe fn inner_from_profiler(
    profiler: *mut TfLiteTelemetryProfilerStruct,
) -> &'static Arc<Mutex<ProfilerInner>> {
    // SAFETY: Caller guarantees the pointer is valid and the pointee is alive.
    unsafe { &*((*profiler).data.cast::<Arc<Mutex<ProfilerInner>>>()) }
}

/// Called at the start of each op invocation. Records the start time and
/// returns a handle that `TFLite` passes back to `report_end_op_invoke`.
///
/// # Safety
///
/// `profiler` must be a valid pointer to a `TfLiteTelemetryProfilerStruct`
/// whose `data` field points to a live `Arc<Mutex<ProfilerInner>>`.
/// `op_name` must be a valid, NUL-terminated C string.
unsafe extern "C" fn report_begin_op_invoke(
    profiler: *mut TfLiteTelemetryProfilerStruct,
    op_name: *const c_char,
    op_idx: i64,
    subgraph_idx: i64,
) -> u32 {
    // SAFETY: Caller (TFLite runtime) upholds the data-pointer invariant.
    let inner = unsafe { inner_from_profiler(profiler) };
    let mut guard = inner.lock().unwrap_or_else(PoisonError::into_inner);
    let handle = guard.next_handle;
    guard.next_handle = guard.next_handle.wrapping_add(1);
    // SAFETY: `op_name` is a valid C string provided by the TFLite runtime.
    let name = unsafe { CStr::from_ptr(op_name) }
        .to_string_lossy()
        .into_owned();
    guard
        .pending
        .insert(handle, (name, op_idx, subgraph_idx, Instant::now()));
    handle
}

/// Called at the end of each op invocation. Computes elapsed time from the
/// corresponding begin event and records a completed [`OpEvent`].
///
/// # Safety
///
/// `profiler` must be a valid pointer to a `TfLiteTelemetryProfilerStruct`
/// whose `data` field points to a live `Arc<Mutex<ProfilerInner>>`.
unsafe extern "C" fn report_end_op_invoke(
    profiler: *mut TfLiteTelemetryProfilerStruct,
    event_handle: u32,
) {
    // SAFETY: Caller (TFLite runtime) upholds the data-pointer invariant.
    let inner = unsafe { inner_from_profiler(profiler) };
    let mut guard = inner.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((op_name, op_idx, subgraph_idx, start)) = guard.pending.remove(&event_handle) {
        #[allow(clippy::cast_possible_truncation)]
        let duration_us = start.elapsed().as_micros() as u64;
        guard.events.push(OpEvent {
            op_name,
            op_idx,
            subgraph_idx,
            duration_us,
        });
    }
}

/// Called for ops that self-report their timing (`elapsed_time` in
/// microseconds).
///
/// # Safety
///
/// `profiler` must be a valid pointer to a `TfLiteTelemetryProfilerStruct`
/// whose `data` field points to a live `Arc<Mutex<ProfilerInner>>`.
/// `op_name` must be a valid, NUL-terminated C string.
unsafe extern "C" fn report_op_invoke_event(
    profiler: *mut TfLiteTelemetryProfilerStruct,
    op_name: *const c_char,
    elapsed_time: u64,
    op_idx: i64,
    subgraph_idx: i64,
) {
    // SAFETY: Caller (TFLite runtime) upholds the data-pointer invariant.
    let inner = unsafe { inner_from_profiler(profiler) };
    let mut guard = inner.lock().unwrap_or_else(PoisonError::into_inner);
    // SAFETY: `op_name` is a valid C string provided by the TFLite runtime.
    let name = unsafe { CStr::from_ptr(op_name) }
        .to_string_lossy()
        .into_owned();
    guard.events.push(OpEvent {
        op_name: name,
        op_idx,
        subgraph_idx,
        duration_us: elapsed_time,
    });
}

// ---------------------------------------------------------------------------
// ProfilerInner
// ---------------------------------------------------------------------------

/// Shared mutable state for the profiler, protected by a `Mutex`.
struct ProfilerInner {
    /// Completed op timing events.
    events: Vec<OpEvent>,
    /// In-flight events keyed by handle.
    pending: HashMap<u32, (String, i64, i64, Instant)>,
    /// Monotonically increasing handle counter.
    next_handle: u32,
}

// ---------------------------------------------------------------------------
// Profiler
// ---------------------------------------------------------------------------

/// Collects per-op timing events during `TFLite` inference.
///
/// Created via [`Profiler::new`], attached to an interpreter via
/// [`InterpreterBuilder::profiler`](crate::InterpreterBuilder::profiler),
/// then events are read after
/// [`Interpreter::invoke`](crate::Interpreter::invoke).
///
/// The `Profiler` must outlive the [`Interpreter`](crate::Interpreter) it
/// is attached to. This is guaranteed when the `Profiler` is declared
/// before the `Interpreter` in the same scope, or when it is stored in a
/// longer-lived struct.
///
/// # Example
///
/// ```no_run
/// use edgefirst_tflite::{Library, Model, Interpreter, Profiler};
///
/// let lib = Library::new()?;
/// let model = Model::from_file(&lib, "model.tflite")?;
/// let profiler = Profiler::new();
///
/// let mut interp = Interpreter::builder(&lib)?
///     .profiler(&profiler)?
///     .build(&model)?;
///
/// interp.invoke()?;
///
/// for event in profiler.events() {
///     println!("{}: {}us", event.op_name, event.duration_us);
/// }
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
pub struct Profiler {
    /// Shared state holding completed and in-flight events.
    inner: Arc<Mutex<ProfilerInner>>,
    /// Boxed C struct that `TFLite` holds a pointer to. Must not move after
    /// the pointer is handed to `TfLiteInterpreterOptionsSetTelemetryProfiler`.
    c_struct: Box<TfLiteTelemetryProfilerStruct>,
    /// Raw pointer to a heap-allocated `Arc<Mutex<ProfilerInner>>` created
    /// via `Box::into_raw`. Freed on drop.
    data_ptr: *mut Arc<Mutex<ProfilerInner>>,
}

// SAFETY: The `c_struct` contains a `*mut c_void` data pointer to an
// `Arc<Mutex<ProfilerInner>>`, which is itself `Send + Sync`. The C struct
// is only mutated through the `Mutex`-protected inner state. The raw
// `data_ptr` is never dereferenced outside the Mutex-guarded callbacks.
unsafe impl Send for Profiler {}
// SAFETY: All mutable access goes through the `Mutex` inside the `Arc`.
unsafe impl Sync for Profiler {}

impl Profiler {
    /// Create a new profiler ready to be attached to an interpreter.
    #[must_use]
    pub fn new() -> Self {
        let inner = Arc::new(Mutex::new(ProfilerInner {
            events: Vec::new(),
            pending: HashMap::new(),
            next_handle: 0,
        }));

        // Heap-allocate a clone of the Arc so we have a stable pointer that
        // the C callbacks can cast back to `&Arc<Mutex<ProfilerInner>>`.
        let data_box = Box::new(inner.clone());
        let data_ptr = Box::into_raw(data_box);

        let c_struct = Box::new(TfLiteTelemetryProfilerStruct {
            data: data_ptr.cast::<c_void>(),
            report_telemetry_event: Some(report_telemetry_event_noop),
            report_telemetry_op_event: Some(report_telemetry_op_event_noop),
            report_settings: Some(report_settings_noop),
            report_begin_op_invoke_event: Some(report_begin_op_invoke),
            report_end_op_invoke_event: Some(report_end_op_invoke),
            report_op_invoke_event: Some(report_op_invoke_event),
        });

        Self {
            inner,
            c_struct,
            data_ptr,
        }
    }

    /// Get a snapshot of all collected events since the last drain or clear.
    #[must_use]
    pub fn events(&self) -> Vec<OpEvent> {
        let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        guard.events.clone()
    }

    /// Drain and return all collected events, leaving the internal list empty.
    #[must_use]
    pub fn drain_events(&self) -> Vec<OpEvent> {
        let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        std::mem::take(&mut guard.events)
    }

    /// Clear all collected events without returning them.
    pub fn clear(&self) {
        let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        guard.events.clear();
        guard.pending.clear();
        guard.next_handle = 0;
    }

    /// Returns the number of completed events collected so far.
    #[must_use]
    pub fn event_count(&self) -> usize {
        let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        guard.events.len()
    }

    /// Returns a raw mutable pointer to the C profiler struct for FFI.
    ///
    /// The returned pointer is valid for the lifetime of this `Profiler`.
    /// It points to the `TfLiteTelemetryProfilerStruct` but is returned as
    /// `*mut c_void` to avoid exposing the private C struct type.
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        (self.c_struct.as_ref() as *const TfLiteTelemetryProfilerStruct)
            .cast_mut()
            .cast()
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        // SAFETY: `data_ptr` was created via `Box::into_raw` in `new()`.
        // We reconstruct the Box so the inner `Arc` is dropped properly,
        // decrementing its refcount.
        unsafe {
            drop(Box::from_raw(self.data_ptr));
        }
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Profiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        f.debug_struct("Profiler")
            .field("events", &guard.events.len())
            .field("pending", &guard.pending.len())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: return a typed pointer to the C struct for direct
    /// callback invocation in tests.
    fn c_struct_ptr(profiler: &Profiler) -> *mut TfLiteTelemetryProfilerStruct {
        profiler.as_ptr().cast()
    }

    #[test]
    fn new_profiler_has_no_events() {
        let profiler = Profiler::new();
        assert!(profiler.events().is_empty());
        assert_eq!(profiler.event_count(), 0);
    }

    #[test]
    fn default_matches_new() {
        let profiler = Profiler::default();
        assert!(profiler.events().is_empty());
    }

    #[test]
    fn clear_resets_state() {
        let profiler = Profiler::new();
        // Manually push an event through the inner state.
        {
            let mut guard = profiler.inner.lock().unwrap();
            guard.events.push(OpEvent {
                op_name: "TEST_OP".to_string(),
                op_idx: 0,
                subgraph_idx: 0,
                duration_us: 100,
            });
        }
        assert_eq!(profiler.event_count(), 1);
        profiler.clear();
        assert_eq!(profiler.event_count(), 0);
    }

    #[test]
    fn drain_events_empties_list() {
        let profiler = Profiler::new();
        {
            let mut guard = profiler.inner.lock().unwrap();
            guard.events.push(OpEvent {
                op_name: "OP_A".to_string(),
                op_idx: 1,
                subgraph_idx: 0,
                duration_us: 50,
            });
            guard.events.push(OpEvent {
                op_name: "OP_B".to_string(),
                op_idx: 2,
                subgraph_idx: 0,
                duration_us: 75,
            });
        }
        let drained = profiler.drain_events();
        assert_eq!(drained.len(), 2);
        assert!(profiler.events().is_empty());
    }

    #[test]
    fn events_returns_snapshot() {
        let profiler = Profiler::new();
        {
            let mut guard = profiler.inner.lock().unwrap();
            guard.events.push(OpEvent {
                op_name: "CONV2D".to_string(),
                op_idx: 0,
                subgraph_idx: 0,
                duration_us: 200,
            });
        }
        let events = profiler.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].op_name, "CONV2D");
        assert_eq!(events[0].duration_us, 200);
        // Original events still present (snapshot, not drain).
        assert_eq!(profiler.event_count(), 1);
    }

    #[test]
    fn debug_format() {
        let profiler = Profiler::new();
        let debug = format!("{profiler:?}");
        assert!(debug.contains("Profiler"));
        assert!(debug.contains("events"));
    }

    #[test]
    fn op_event_debug_clone() {
        let event = OpEvent {
            op_name: "SOFTMAX".to_string(),
            op_idx: 3,
            subgraph_idx: 0,
            duration_us: 42,
        };
        let cloned = event.clone();
        assert_eq!(cloned.op_name, "SOFTMAX");
        assert_eq!(cloned.op_idx, 3);
        assert_eq!(cloned.duration_us, 42);
        let debug = format!("{event:?}");
        assert!(debug.contains("SOFTMAX"));
    }

    #[test]
    fn profiler_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Profiler>();
    }

    #[test]
    fn c_struct_pointer_is_stable() {
        let profiler = Profiler::new();
        let ptr1 = profiler.as_ptr();
        let ptr2 = profiler.as_ptr();
        assert_eq!(ptr1, ptr2, "C struct pointer must be stable (boxed)");
    }

    #[test]
    fn begin_end_callback_round_trip() {
        let profiler = Profiler::new();
        let c_ptr = c_struct_ptr(&profiler);

        let op_name = c"TEST_OP";

        // Simulate what TFLite does: call begin, then end.
        // SAFETY: We own the profiler and the C struct is valid.
        let handle = unsafe {
            ((*c_ptr).report_begin_op_invoke_event.unwrap())(c_ptr, op_name.as_ptr(), 5, 0)
        };
        // Small delay to get a nonzero duration.
        std::thread::sleep(std::time::Duration::from_micros(10));
        unsafe {
            ((*c_ptr).report_end_op_invoke_event.unwrap())(c_ptr, handle);
        }

        let events = profiler.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].op_name, "TEST_OP");
        assert_eq!(events[0].op_idx, 5);
        assert_eq!(events[0].subgraph_idx, 0);
        // Duration should be at least a few microseconds.
        assert!(events[0].duration_us > 0);
    }

    #[test]
    fn self_reported_op_invoke_callback() {
        let profiler = Profiler::new();
        let c_ptr = c_struct_ptr(&profiler);

        let op_name = c"DELEGATE_OP";

        // SAFETY: We own the profiler and the C struct is valid.
        unsafe {
            ((*c_ptr).report_op_invoke_event.unwrap())(c_ptr, op_name.as_ptr(), 1234, 2, 1);
        }

        let events = profiler.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].op_name, "DELEGATE_OP");
        assert_eq!(events[0].duration_us, 1234);
        assert_eq!(events[0].op_idx, 2);
        assert_eq!(events[0].subgraph_idx, 1);
    }

    #[test]
    fn end_with_unknown_handle_is_ignored() {
        let profiler = Profiler::new();
        let c_ptr = c_struct_ptr(&profiler);

        // SAFETY: We own the profiler and the C struct is valid.
        unsafe {
            ((*c_ptr).report_end_op_invoke_event.unwrap())(c_ptr, 999);
        }

        assert!(profiler.events().is_empty());
    }
}
