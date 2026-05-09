// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Python `Profiler` and `OpEvent` classes for the telemetry profiler API.

use std::sync::Arc;

use pyo3::prelude::*;

/// One recorded operator timing event.
#[pyclass(name = "OpEvent")]
#[derive(Debug, Clone)]
pub struct PyOpEvent {
    pub(crate) inner: edgefirst_tflite::OpEvent,
}

#[pymethods]
impl PyOpEvent {
    #[getter]
    fn op_name(&self) -> &str {
        &self.inner.op_name
    }

    #[getter]
    fn op_idx(&self) -> i64 {
        self.inner.op_idx
    }

    #[getter]
    fn subgraph_idx(&self) -> i64 {
        self.inner.subgraph_idx
    }

    #[getter]
    fn duration_us(&self) -> u64 {
        self.inner.duration_us
    }

    fn __repr__(&self) -> String {
        format!(
            "OpEvent(op_name={:?}, op_idx={}, subgraph_idx={}, duration_us={})",
            self.inner.op_name, self.inner.op_idx, self.inner.subgraph_idx, self.inner.duration_us
        )
    }
}

/// Op-level telemetry profiler.
///
/// Construct one, pass it as ``profiler=`` to ``Interpreter(...)``,
/// then read events after each ``invoke()``.
#[pyclass(name = "Profiler")]
#[derive(Debug)]
pub struct PyProfiler {
    pub(crate) inner: Arc<edgefirst_tflite::Profiler>,
}

impl PyProfiler {
    pub(crate) fn arc(&self) -> Arc<edgefirst_tflite::Profiler> {
        Arc::clone(&self.inner)
    }
}

#[pymethods]
impl PyProfiler {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(edgefirst_tflite::Profiler::new()),
        }
    }

    /// Snapshot of all collected events since the last drain or clear.
    fn events(&self) -> Vec<PyOpEvent> {
        self.inner
            .events()
            .into_iter()
            .map(|e| PyOpEvent { inner: e })
            .collect()
    }

    /// Drain and return all collected events, leaving the internal list empty.
    fn drain_events(&self) -> Vec<PyOpEvent> {
        self.inner
            .drain_events()
            .into_iter()
            .map(|e| PyOpEvent { inner: e })
            .collect()
    }

    /// Clear all collected events without returning them.
    fn clear(&self) {
        self.inner.clear();
    }

    /// Number of completed events collected so far.
    fn event_count(&self) -> usize {
        self.inner.event_count()
    }

    fn __len__(&self) -> usize {
        self.inner.event_count()
    }

    fn __repr__(&self) -> String {
        format!("Profiler(events={})", self.inner.event_count())
    }
}
