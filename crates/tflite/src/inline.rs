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

use crate::schema_generated::tflite;

/// If `data` contains any offset-stored buffers, returns a rewritten model
/// with every buffer inlined; otherwise returns `None` (the caller uses the
/// original bytes unchanged, paying nothing).
pub(crate) fn inline_offset_buffers(data: &[u8]) -> Option<Vec<u8>> {
    let model = tflite::root_as_model(data).ok()?;

    // Fast path: nothing to do unless a buffer is offset-stored.
    let has_offset = model
        .buffers()
        .is_some_and(|bufs| bufs.iter().any(|b| b.offset() > 0));
    if !has_offset {
        return None;
    }

    let mut model_t = model.unpack();
    if let Some(buffers) = model_t.buffers.as_mut() {
        for buf in buffers.iter_mut() {
            if buf.offset > 0 {
                let start = usize::try_from(buf.offset).ok()?;
                let end = start.checked_add(usize::try_from(buf.size).ok()?)?;
                // A malformed offset/size that does not lie within the model
                // bytes: give up on rewriting rather than panic; the caller
                // falls back to the original bytes.
                let slice = data.get(start..end)?;
                buf.data = Some(slice.to_vec());
                buf.offset = 0;
                buf.size = 0;
            }
        }
    }

    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let root = model_t.pack(&mut builder);
    builder.finish(root, Some("TFL3"));
    Some(builder.finished_data().to_vec())
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
        // nothing to rewrite and the fast path returns None.
        assert_eq!(offset_buffer_count(INLINE_ONLY), 0, "fixture precondition");
        assert!(inline_offset_buffers(INLINE_ONLY).is_none());
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

        let rewritten =
            inline_offset_buffers(OFFSET_BUFFERS).expect("model with offset buffers is rewritten");

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
