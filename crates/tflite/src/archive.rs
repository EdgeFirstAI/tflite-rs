// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Embedded ZIP-archive metadata for `TFLite` model files.
//!
//! The EdgeFirst tflite-converter appends a standard ZIP archive (typically
//! containing `edgefirst.json`, `labels.txt`, and `metadata.json`) to the
//! end of the FlatBuffer payload. The TFLite runtime ignores the trailing
//! bytes, so the model still loads with any standards-compliant interpreter.
//! ZIP readers locate the central directory by scanning backwards from the
//! end of the file for the `PK\x05\x06` end-of-central-directory marker, so
//! both formats coexist in the same file.
//!
//! This module exposes the embedded archive without forcing callers to take
//! a direct dependency on the `zip` crate. The high-level [`ModelArchive`]
//! type wraps a model byte slice and lets you read entries by name; the
//! free helpers [`edgefirst_json`], [`labels`], and [`has_archive`] are
//! one-shot conveniences for the common cases.
//!
//! # Quick start
//!
//! ```no_run
//! use edgefirst_tflite::archive::ModelArchive;
//!
//! let bytes = std::fs::read("model.tflite")?;
//! let mut archive = ModelArchive::new(&bytes)?;
//! let json = archive.edgefirst_json()?;
//! let labels = archive.labels()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Usage with `Model`
//!
//! ```no_run
//! use edgefirst_tflite::{Library, Model};
//! use edgefirst_tflite::archive::ModelArchive;
//!
//! let lib = Library::new()?;
//! let model = Model::from_file(&lib, "model.tflite")?;
//! let mut archive = ModelArchive::new(model.data())?;
//! let json = archive.edgefirst_json()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::io::{Cursor, Read};

use crate::error::{Error, Result};

/// File name of the EdgeFirst schema JSON inside the archive.
pub const EDGEFIRST_JSON: &str = "edgefirst.json";
/// File name of the line-delimited class labels inside the archive.
pub const LABELS_TXT: &str = "labels.txt";
/// File name of the supplementary human-readable metadata inside the archive.
pub const METADATA_JSON: &str = "metadata.json";

/// Read-only view of the ZIP archive embedded at the end of a `.tflite`
/// model file.
///
/// Construction reads the central directory once; subsequent calls to
/// [`ModelArchive::read`] / [`ModelArchive::read_to_string`] decompress
/// individual entries on demand.
pub struct ModelArchive<'a> {
    inner: zip::ZipArchive<Cursor<&'a [u8]>>,
}

impl std::fmt::Debug for ModelArchive<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelArchive")
            .field("len", &self.inner.len())
            .finish()
    }
}

impl<'a> ModelArchive<'a> {
    /// Open the archive that is appended to a `.tflite` model byte slice.
    ///
    /// Returns an error if `data` does not end with a valid ZIP central
    /// directory. Use [`has_archive`] for a non-allocating probe.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let inner = zip::ZipArchive::new(Cursor::new(data))
            .map_err(|e| Error::invalid_argument(format!("no embedded ZIP archive: {e}")))?;
        Ok(Self { inner })
    }

    /// Number of entries present in the archive.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the archive has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate over the names of all entries in the archive.
    pub fn entry_names(&self) -> impl Iterator<Item = &str> {
        self.inner.file_names()
    }

    /// Read an entry by name into an owned byte buffer.
    pub fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        let mut file = self
            .inner
            .by_name(name)
            .map_err(|e| Error::invalid_argument(format!("archive entry {name:?}: {e}")))?;
        let mut buf = Vec::with_capacity(usize::try_from(file.size()).unwrap_or(0));
        file.read_to_end(&mut buf)
            .map_err(|e| Error::invalid_argument(format!("read archive entry {name:?}: {e}")))?;
        Ok(buf)
    }

    /// Read an entry by name and decode as UTF-8.
    pub fn read_to_string(&mut self, name: &str) -> Result<String> {
        let mut file = self
            .inner
            .by_name(name)
            .map_err(|e| Error::invalid_argument(format!("archive entry {name:?}: {e}")))?;
        let mut s = String::with_capacity(usize::try_from(file.size()).unwrap_or(0));
        file.read_to_string(&mut s).map_err(|e| {
            Error::invalid_argument(format!("read archive entry {name:?} as utf-8: {e}"))
        })?;
        Ok(s)
    }

    /// Returns `true` if the archive contains an entry with the given name.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.inner.index_for_name(name).is_some()
    }

    /// Read [`EDGEFIRST_JSON`] as a UTF-8 string.
    ///
    /// The returned string is suitable for
    /// [`SchemaV2::parse_json`](https://docs.rs/edgefirst-decoder/latest/edgefirst_decoder/schema/struct.SchemaV2.html#method.parse_json)
    /// or
    /// [`DecoderBuilder::with_config_json_str`](https://docs.rs/edgefirst-decoder/latest/edgefirst_decoder/struct.DecoderBuilder.html#method.with_config_json_str).
    pub fn edgefirst_json(&mut self) -> Result<String> {
        self.read_to_string(EDGEFIRST_JSON)
    }

    /// Read [`LABELS_TXT`] and split into one label per line.
    ///
    /// Trailing whitespace on each line is trimmed; blank lines are skipped.
    pub fn labels(&mut self) -> Result<Vec<String>> {
        let raw = self.read_to_string(LABELS_TXT)?;
        Ok(raw
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect())
    }
}

/// Probe whether `data` ends with a valid ZIP archive.
///
/// Returns `false` for plain `.tflite` files without an embedded archive,
/// for truncated files, and for files whose trailing bytes are not a
/// well-formed central directory. Does not allocate.
#[must_use]
pub fn has_archive(data: &[u8]) -> bool {
    zip::ZipArchive::new(Cursor::new(data)).is_ok()
}

/// One-shot read of `edgefirst.json` from a model byte slice.
///
/// Equivalent to `ModelArchive::new(data)?.edgefirst_json()`. Prefer
/// [`ModelArchive`] when reading more than one entry.
pub fn edgefirst_json(data: &[u8]) -> Result<String> {
    ModelArchive::new(data)?.edgefirst_json()
}

/// One-shot read of `labels.txt` from a model byte slice.
pub fn labels(data: &[u8]) -> Result<Vec<String>> {
    ModelArchive::new(data)?.labels()
}

#[cfg(test)]
mod tests {
    use super::*;

    static MODEL_WITH_ARCHIVE: &[u8] =
        include_bytes!("../../../testdata/yolov8n-seg-int8.tflite");
    static MINIMAL_MODEL: &[u8] = include_bytes!("../../../testdata/minimal.tflite");

    /// Schema v2 fixtures, one per converter output layout. Each model
    /// embeds an `edgefirst.json` that exercises a different decoder
    /// dispatch path:
    /// - `combined`: single fused detection tensor + protos.
    /// - `logical`: separate boxes/scores/mask_coefs/protos.
    /// - `smart`: per-scale FPN-split outputs (DFL + sub-stride children).
    static SCHEMA_V2_COMBINED: &[u8] =
        include_bytes!("../../../testdata/yolov8n-seg-combined-int8.tflite");
    static SCHEMA_V2_LOGICAL: &[u8] =
        include_bytes!("../../../testdata/yolov8n-seg-logical-int8.tflite");
    static SCHEMA_V2_SMART: &[u8] =
        include_bytes!("../../../testdata/yolov8n-seg-smart-int8.tflite");

    fn schema_v2_models() -> [(&'static str, &'static [u8]); 3] {
        [
            ("combined", SCHEMA_V2_COMBINED),
            ("logical", SCHEMA_V2_LOGICAL),
            ("smart", SCHEMA_V2_SMART),
        ]
    }

    #[test]
    fn detects_archive_presence() {
        assert!(has_archive(MODEL_WITH_ARCHIVE));
        assert!(!has_archive(MINIMAL_MODEL));
        assert!(!has_archive(&[]));
        assert!(!has_archive(&[0u8; 16]));
    }

    #[test]
    fn lists_expected_entries() {
        let archive = ModelArchive::new(MODEL_WITH_ARCHIVE).unwrap();
        assert!(archive.contains(EDGEFIRST_JSON));
        assert!(archive.contains(LABELS_TXT));
        assert!(archive.contains(METADATA_JSON));
        assert!(!archive.contains("missing.txt"));
        assert!(!archive.is_empty());
    }

    #[test]
    fn reads_edgefirst_json() {
        let mut archive = ModelArchive::new(MODEL_WITH_ARCHIVE).unwrap();
        let json = archive.edgefirst_json().unwrap();
        // Both schema v1 and v2 carry a `decoder_version` field, so use that
        // as a stable existence check across versions.
        assert!(json.contains("\"decoder_version\""));
        assert!(json.starts_with('{') && json.trim_end().ends_with('}'));
    }

    #[test]
    fn reads_labels() {
        let mut archive = ModelArchive::new(MODEL_WITH_ARCHIVE).unwrap();
        let labels = archive.labels().unwrap();
        assert_eq!(labels.len(), 80);
        assert_eq!(labels[0], "person");
    }

    #[test]
    fn missing_entry_is_invalid_argument() {
        let mut archive = ModelArchive::new(MODEL_WITH_ARCHIVE).unwrap();
        let err = archive.read("does-not-exist").unwrap_err();
        assert!(err.is_invalid_argument());
    }

    #[test]
    fn one_shot_helpers_roundtrip() {
        let json = edgefirst_json(MODEL_WITH_ARCHIVE).unwrap();
        assert!(json.contains("decoder_version"));
        let labels = labels(MODEL_WITH_ARCHIVE).unwrap();
        assert_eq!(labels.len(), 80);
    }

    #[test]
    fn no_archive_is_invalid_argument() {
        let err = ModelArchive::new(MINIMAL_MODEL).unwrap_err();
        assert!(err.is_invalid_argument());
    }

    /// Every schema v2 fixture must round-trip the archive open + entry
    /// reads. This guards the API against ZIP-format regressions for all
    /// three converter layouts produced by the EdgeFirst tflite-converter.
    #[test]
    fn schema_v2_fixtures_open_cleanly() {
        for (name, bytes) in schema_v2_models() {
            assert!(
                has_archive(bytes),
                "{name}: no embedded archive detected"
            );
            let mut archive = ModelArchive::new(bytes)
                .unwrap_or_else(|e| panic!("{name}: open failed: {e}"));
            assert!(
                archive.contains(EDGEFIRST_JSON),
                "{name}: missing edgefirst.json"
            );
            assert!(archive.contains(LABELS_TXT), "{name}: missing labels.txt");
            let labels = archive
                .labels()
                .unwrap_or_else(|e| panic!("{name}: labels read failed: {e}"));
            assert_eq!(labels.len(), 80, "{name}: expected 80 COCO labels");
            assert_eq!(labels[0], "person", "{name}: first label");
        }
    }

    /// Schema v2 metadata advertises `"schema_version": 2` and
    /// `"decoder_version": "yolov8"`, regardless of which physical output
    /// layout the converter chose. We assert this contract here so a
    /// breaking change in the converter or the schema is caught locally.
    #[test]
    fn schema_v2_fixtures_advertise_schema_v2() {
        for (name, bytes) in schema_v2_models() {
            let mut archive = ModelArchive::new(bytes).expect("open");
            let json = archive
                .edgefirst_json()
                .unwrap_or_else(|e| panic!("{name}: edgefirst.json read: {e}"));
            assert!(
                json.contains("\"schema_version\": 2") || json.contains("\"schema_version\":2"),
                "{name}: edgefirst.json is not schema v2:\n{}",
                &json[..json.len().min(200)],
            );
            assert!(
                json.contains("\"decoder_version\": \"yolov8\"")
                    || json.contains("\"decoder_version\":\"yolov8\""),
                "{name}: decoder_version != yolov8"
            );
        }
    }

    /// The "combined" layout fuses boxes+scores+mask_coefs into a single
    /// detection tensor. The "logical" layout exposes separate logical
    /// outputs. The "smart" layout exposes per-scale FPN children with
    /// `scale_index`/`stride`. This test pins the layout signature for
    /// each fixture so we notice if a converter rev silently changes it.
    #[test]
    fn schema_v2_fixtures_match_expected_layout() {
        let cases: &[(&str, &[u8], &[&str], &[&str])] = &[
            (
                "combined",
                SCHEMA_V2_COMBINED,
                &["\"type\": \"detection\"", "\"type\": \"protos\""],
                &["\"type\": \"boxes\"", "\"scale_index\""],
            ),
            (
                "logical",
                SCHEMA_V2_LOGICAL,
                &[
                    "\"type\": \"boxes\"",
                    "\"type\": \"scores\"",
                    "\"type\": \"mask_coefs\"",
                    "\"type\": \"protos\"",
                ],
                &["\"scale_index\""],
            ),
            (
                "smart",
                SCHEMA_V2_SMART,
                &["\"scale_index\"", "\"type\": \"protos\""],
                &[],
            ),
        ];
        for (name, bytes, must_contain, must_not_contain) in cases {
            let mut archive = ModelArchive::new(bytes).expect("open");
            let json = archive.edgefirst_json().expect("read");
            for needle in *must_contain {
                assert!(
                    json.contains(needle),
                    "{name}: expected {needle:?} in edgefirst.json"
                );
            }
            for needle in *must_not_contain {
                assert!(
                    !json.contains(needle),
                    "{name}: did not expect {needle:?} in edgefirst.json"
                );
            }
        }
    }
}
