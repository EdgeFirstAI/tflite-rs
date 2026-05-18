// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Concrete error type for the `yolov8` example pipeline.

use std::fmt;

/// All failure modes the `yolov8` example can encounter.
///
/// Keeping the variants concrete lets the example bubble up errors via
/// `?` without resorting to `Box<dyn Error>` and lets a future
/// integration test pattern-match on the failure mode.
#[derive(Debug)]
pub enum Error {
    /// Filesystem or other `std::io` error reading model/image.
    Io(std::io::Error),
    /// `edgefirst-tflite` reported an inference-runtime failure.
    Tflite(edgefirst_tflite::Error),
    /// HAL `Decoder` build or decode failure.
    Decoder(edgefirst_hal::decoder::DecoderError),
    /// HAL `codec` (image decode / peek) failure.
    Codec(edgefirst_hal::codec::CodecError),
    /// HAL `image` (load / convert / save / overlay) failure.
    Image(edgefirst_hal::image::Error),
    /// HAL `tensor` (allocation / quantization / map) failure.
    Tensor(edgefirst_hal::tensor::Error),
    /// Unsupported configuration the example explicitly rejects, e.g. an
    /// output dtype the post-processing path doesn't handle.
    Unsupported(String),
}

impl Error {
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Tflite(e) => write!(f, "tflite error: {e}"),
            Self::Decoder(e) => write!(f, "decoder error: {e}"),
            Self::Codec(e) => write!(f, "codec error: {e}"),
            Self::Image(e) => write!(f, "image error: {e}"),
            Self::Tensor(e) => write!(f, "tensor error: {e}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Tflite(e) => Some(e),
            Self::Decoder(e) => Some(e),
            Self::Codec(e) => Some(e),
            Self::Image(e) => Some(e),
            Self::Tensor(e) => Some(e),
            Self::Unsupported(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<edgefirst_tflite::Error> for Error {
    fn from(e: edgefirst_tflite::Error) -> Self {
        Self::Tflite(e)
    }
}

impl From<edgefirst_hal::decoder::DecoderError> for Error {
    fn from(e: edgefirst_hal::decoder::DecoderError) -> Self {
        Self::Decoder(e)
    }
}

impl From<edgefirst_hal::image::Error> for Error {
    fn from(e: edgefirst_hal::image::Error) -> Self {
        Self::Image(e)
    }
}

impl From<edgefirst_hal::codec::CodecError> for Error {
    fn from(e: edgefirst_hal::codec::CodecError) -> Self {
        Self::Codec(e)
    }
}

impl From<edgefirst_hal::tensor::Error> for Error {
    fn from(e: edgefirst_hal::tensor::Error) -> Self {
        Self::Tensor(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
