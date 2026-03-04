// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! `TFLite` model metadata extraction.
//!
//! Extracts human-readable metadata (name, version, author, description,
//! license) from `TFLite` model files using the embedded `FlatBuffer`
//! metadata buffer.

#[allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    unused_imports,
    unused_lifetimes,
    redundant_lifetimes,
    mismatched_lifetime_syntaxes,
    dead_code,
    non_camel_case_types,
    non_snake_case,
    missing_debug_implementations,
    unreachable_pub
)]
mod metadata_schema_generated {
    include!("metadata_schema_generated.rs");
}

#[allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    unused_imports,
    unused_lifetimes,
    redundant_lifetimes,
    mismatched_lifetime_syntaxes,
    dead_code,
    non_camel_case_types,
    non_snake_case,
    missing_debug_implementations,
    unreachable_pub
)]
mod schema_generated {
    include!("schema_generated.rs");
}

use metadata_schema_generated::tflite::root_as_model_metadata;
use schema_generated::tflite::root_as_model;
use std::fmt;

/// Metadata buffer name used by the `TFLite` metadata specification.
const METADATA_BUFFER_NAME: &str = "TFLITE_METADATA";

/// Extracted model metadata from a `TFLite` model file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Metadata {
    /// Model name.
    pub name: Option<String>,
    /// Model version.
    pub version: Option<String>,
    /// Model description.
    pub description: Option<String>,
    /// Model author.
    pub author: Option<String>,
    /// Model license.
    pub license: Option<String>,
    /// Minimum parser version required.
    pub min_parser_version: Option<String>,
}

impl Metadata {
    /// Extract metadata from raw model bytes.
    ///
    /// Returns a `Metadata` struct with all fields set to `None` if the model
    /// does not contain a `TFLITE_METADATA` buffer or the buffer cannot be
    /// parsed.
    #[must_use]
    pub fn from_model_bytes(model: &[u8]) -> Self {
        let mut metadata = Self::default();
        let Ok(m) = root_as_model(model) else {
            return metadata;
        };
        let model_desc = m.description();

        let Some(model_meta) = m.metadata() else {
            return metadata;
        };

        for entry in model_meta {
            if entry.name() != Some(METADATA_BUFFER_NAME) {
                continue;
            }

            let buffer_index = entry.buffer();
            let Some(buffers) = m.buffers() else {
                return metadata;
            };
            #[allow(clippy::cast_sign_loss)]
            let Some(data) = buffers.get(buffer_index as usize).data() else {
                return metadata;
            };
            let Ok(model_metadata) = root_as_model_metadata(data.bytes()) else {
                return metadata;
            };

            metadata.name = model_metadata.name().map(str::to_owned);
            metadata.description = match (model_desc, model_metadata.description()) {
                (Some(s1), Some(s2)) => Some(format!("{s1} {s2}")),
                (Some(s1), None) => Some(s1.to_owned()),
                (None, Some(s2)) => Some(s2.to_owned()),
                (None, None) => None,
            };
            metadata.author = model_metadata.author().map(str::to_owned);
            metadata.license = model_metadata.license().map(str::to_owned);
            metadata.min_parser_version = model_metadata.min_parser_version().map(str::to_owned);
            metadata.version = model_metadata.version().map(str::to_owned);
        }

        metadata
    }
}

impl fmt::Display for Metadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            writeln!(f, "Name: {name}")?;
        }
        if let Some(version) = &self.version {
            writeln!(f, "Version: {version}")?;
        }
        if let Some(description) = &self.description {
            writeln!(f, "Description: {description}")?;
        }
        if let Some(author) = &self.author {
            writeln!(f, "Author: {author}")?;
        }
        if let Some(license) = &self.license {
            writeln!(f, "License: {license}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_all_none() {
        let m = Metadata::default();
        assert_eq!(m.name, None);
        assert_eq!(m.version, None);
        assert_eq!(m.description, None);
        assert_eq!(m.author, None);
        assert_eq!(m.license, None);
        assert_eq!(m.min_parser_version, None);
    }

    #[test]
    fn from_model_bytes_empty() {
        let m = Metadata::from_model_bytes(&[]);
        assert_eq!(m, Metadata::default());
    }

    #[test]
    fn from_model_bytes_garbage() {
        let m = Metadata::from_model_bytes(&[0xFF; 16]);
        assert_eq!(m, Metadata::default());
    }

    #[test]
    fn from_model_bytes_minimal_model() {
        static MINIMAL_MODEL: &[u8] = include_bytes!("../../../testdata/minimal.tflite");
        let m = Metadata::from_model_bytes(MINIMAL_MODEL);
        assert_eq!(m, Metadata::default());
    }

    #[test]
    fn display_all_fields_set() {
        let m = Metadata {
            name: Some("TestModel".into()),
            version: Some("1.0".into()),
            description: Some("A test model".into()),
            author: Some("Tester".into()),
            license: Some("MIT".into()),
            min_parser_version: Some("1.5".into()),
        };
        let output = m.to_string();
        assert!(output.contains("Name: TestModel"));
        assert!(output.contains("Version: 1.0"));
        assert!(output.contains("Description: A test model"));
        assert!(output.contains("Author: Tester"));
        assert!(output.contains("License: MIT"));
    }

    #[test]
    fn display_no_fields_set() {
        let m = Metadata::default();
        let output = m.to_string();
        assert!(output.is_empty());
    }

    #[test]
    fn display_partial_fields() {
        let m = Metadata {
            name: Some("PartialModel".into()),
            version: None,
            description: Some("Only name and description".into()),
            author: None,
            license: None,
            min_parser_version: None,
        };
        let output = m.to_string();
        assert!(output.contains("Name: PartialModel"));
        assert!(output.contains("Description: Only name and description"));
        assert!(!output.contains("Version:"));
        assert!(!output.contains("Author:"));
        assert!(!output.contains("License:"));
    }
}
