// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Inlining of offset-stored model buffers.
//!
//! `TFLite`'s flatbuffer schema lets a `Buffer` store its bytes *outside*
//! the flatbuffer — appended after it and referenced by `Buffer.offset`/
//! `size` rather than an inline `data` vector. ai-edge / `LiteRT` exporters
//! (and the standard Ultralytics int8 `TFLite` export) use this for large
//! weight and bias constants. The `TFLite` **C API** (`TfLiteModelCreate` /
//! `TfLiteModelCreateFromFile`) does not resolve these offset-stored
//! buffers, so those constants have null data at inference time and the
//! interpreter aborts with "Input tensor N lacks data" — even though the
//! same model loads and runs through the C++ `InterpreterBuilder`.
//!
//! [`inline_offset_buffers`] rewrites such a model in memory so every
//! buffer is stored inline, which the C API reads correctly. The rewrite is
//! used only as the runtime's model buffer; the caller keeps the original
//! bytes for [`Model::data`](crate::Model::data), so any trailing content
//! the exporter appended (an `EdgeFirst` / Ultralytics ZIP metadata trailer,
//! whose entries carry absolute file offsets) stays intact for metadata
//! readers.

use crate::error::{Error, Result};
use crate::schema_generated::tflite;

/// Rewrites `data` so every buffer is stored inline.
///
/// Returns `Ok(None)` when there is nothing to do — the model stores all its
/// buffers inline already (or is not a parseable flatbuffer, which the runtime
/// loader will report) — so the caller uses the original bytes unchanged,
/// paying nothing. Returns `Ok(Some(bytes))` with the rewritten model when
/// offset-stored buffers were inlined. Returns `Err` when the model *does*
/// declare offset-stored buffers but one cannot be resolved (its
/// `offset`/`size` lies outside the model bytes); surfacing that here fails
/// the load cleanly instead of deferring to a runtime "Input tensor N lacks
/// data" abort.
pub(crate) fn inline_offset_buffers(data: &[u8]) -> Result<Option<Vec<u8>>> {
    // A model we cannot parse is not our concern; the runtime loader reports
    // it. Only a parseable model with offset-stored buffers is rewritten.
    let Ok(model) = tflite::root_as_model(data) else {
        return Ok(None);
    };

    let has_offset = model
        .buffers()
        .is_some_and(|bufs| bufs.iter().any(|b| b.offset() > 0));
    if !has_offset {
        return Ok(None);
    }

    let mut model_t = model.unpack();
    if let Some(buffers) = model_t.buffers.as_mut() {
        for buf in buffers.iter_mut() {
            if buf.offset > 0 {
                let slice = usize::try_from(buf.offset).ok().and_then(|start| {
                    let size = usize::try_from(buf.size).ok()?;
                    let end = start.checked_add(size)?;
                    data.get(start..end)
                });
                let Some(slice) = slice else {
                    return Err(Error::invalid_argument(format!(
                        "offset-stored buffer [offset={}, size={}] lies outside the {}-byte model",
                        buf.offset,
                        buf.size,
                        data.len(),
                    )));
                };
                buf.data = Some(slice.to_vec());
                buf.offset = 0;
                buf.size = 0;
            }
        }
    }

    // The rewrite is roughly the size of the input, so pre-size the builder to
    // avoid repeated growth reallocations on large (multi-MB) models.
    let mut builder = flatbuffers::FlatBufferBuilder::with_capacity(data.len());
    let root = model_t.pack(&mut builder);
    builder.finish(root, Some("TFL3"));
    Ok(Some(builder.finished_data().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const INLINE_ONLY: &[u8] = include_bytes!("../../../testdata/minimal.tflite");
    const OFFSET_BUFFERS: &[u8] = include_bytes!("../../../testdata/minimal_offset_buffers.tflite");

    fn offset_buffer_count(model: &[u8]) -> usize {
        let m = tflite::root_as_model(model).expect("valid tflite model");
        m.buffers()
            .map_or(0, |bufs| bufs.iter().filter(|b| b.offset() > 0).count())
    }

    /// Buffer `i`'s inline bytes, or empty if absent.
    fn inline_bytes(model: &[u8], i: usize) -> Vec<u8> {
        let m = tflite::root_as_model(model).expect("valid tflite model");
        m.buffers()
            .and_then(|bufs| bufs.get(i).data())
            .map(|d| d.bytes().to_vec())
            .unwrap_or_default()
    }

    #[test]
    fn inline_only_model_is_left_unchanged() {
        // The stock minimal.tflite stores every buffer inline, so there is
        // nothing to rewrite and the fast path returns Ok(None).
        assert_eq!(offset_buffer_count(INLINE_ONLY), 0, "fixture precondition");
        assert!(inline_offset_buffers(INLINE_ONLY).unwrap().is_none());
    }

    #[test]
    fn offset_buffer_outside_model_bytes_is_an_error() {
        // Truncating the fixture to just its flatbuffer leaves the buffer
        // offsets pointing past the end. Such a model declares offset buffers
        // it cannot resolve, so inlining must fail the load rather than fall
        // back to bytes the C API would abort on at invoke time.
        assert_eq!(
            offset_buffer_count(OFFSET_BUFFERS),
            3,
            "fixture precondition"
        );
        let flatbuffer_only = &OFFSET_BUFFERS[..848];
        let err = inline_offset_buffers(flatbuffer_only)
            .expect_err("unresolvable offset buffer must be an error");
        assert!(err.is_invalid_argument(), "{err}");
    }

    #[test]
    fn offset_buffers_are_inlined_with_data_preserved() {
        // Precondition: the fixture really uses offset-stored buffers, and
        // the reference inline model has the same buffer contents.
        assert_eq!(
            offset_buffer_count(OFFSET_BUFFERS),
            3,
            "fixture precondition"
        );

        let rewritten = inline_offset_buffers(OFFSET_BUFFERS)
            .expect("rewrite must not error")
            .expect("model with offset buffers is rewritten");

        // Every buffer is now inline.
        assert_eq!(offset_buffer_count(&rewritten), 0);

        // Each buffer's bytes match the original inline-only model exactly.
        for i in 0..6 {
            assert_eq!(
                inline_bytes(&rewritten, i),
                inline_bytes(INLINE_ONLY, i),
                "buffer {i} contents differ after inlining"
            );
        }
    }
}
