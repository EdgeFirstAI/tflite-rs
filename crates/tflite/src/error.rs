// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Error types for the `edgefirst-tflite` crate.
//!
//! This module follows the canonical error struct pattern: a public [`Error`]
//! struct wrapping a private `ErrorKind` enum. Callers inspect errors through
//! [`Error::is_library_error`], [`Error::is_delegate_error`],
//! [`Error::is_null_pointer`], and [`Error::status_code`] rather than matching
//! on variants directly.

use std::fmt;

// ---------------------------------------------------------------------------
// StatusCode
// ---------------------------------------------------------------------------

/// Status codes returned by the TensorFlow Lite C API.
///
/// Each variant maps to a `kTfLite*` constant defined in the C header
/// `common.h`. The numeric value is accessible via `as u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusCode {
    /// Generic runtime error (`kTfLiteError = 1`).
    RuntimeError = 1,
    /// Delegate returned an error (`kTfLiteDelegateError = 2`).
    DelegateError = 2,
    /// Application-level error (`kTfLiteApplicationError = 3`).
    ApplicationError = 3,
    /// Delegate data not found (`kTfLiteDelegateDataNotFound = 4`).
    DelegateDataNotFound = 4,
    /// Delegate data write error (`kTfLiteDelegateDataWriteError = 5`).
    DelegateDataWriteError = 5,
    /// Delegate data read error (`kTfLiteDelegateDataReadError = 6`).
    DelegateDataReadError = 6,
    /// Model contains unresolved ops (`kTfLiteUnresolvedOps = 7`).
    UnresolvedOps = 7,
    /// Operation was cancelled (`kTfLiteCancelled = 8`).
    Cancelled = 8,
    /// Output tensor shape is not yet known (`kTfLiteOutputShapeNotKnown = 9`).
    OutputShapeNotKnown = 9,
}

impl StatusCode {
    /// Attempt to convert a raw C API status value into a `StatusCode`.
    ///
    /// Returns `None` for `kTfLiteOk` (0) or any unknown value.
    fn from_raw(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::RuntimeError),
            2 => Some(Self::DelegateError),
            3 => Some(Self::ApplicationError),
            4 => Some(Self::DelegateDataNotFound),
            5 => Some(Self::DelegateDataWriteError),
            6 => Some(Self::DelegateDataReadError),
            7 => Some(Self::UnresolvedOps),
            8 => Some(Self::Cancelled),
            9 => Some(Self::OutputShapeNotKnown),
            _ => None,
        }
    }
}

impl fmt::Display for StatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeError => f.write_str("runtime error"),
            Self::DelegateError => f.write_str("delegate error"),
            Self::ApplicationError => f.write_str("application error"),
            Self::DelegateDataNotFound => f.write_str("delegate data not found"),
            Self::DelegateDataWriteError => f.write_str("delegate data write error"),
            Self::DelegateDataReadError => f.write_str("delegate data read error"),
            Self::UnresolvedOps => f.write_str("unresolved ops"),
            Self::Cancelled => f.write_str("cancelled"),
            Self::OutputShapeNotKnown => f.write_str("output shape not known"),
        }
    }
}

// ---------------------------------------------------------------------------
// ErrorKind (private)
// ---------------------------------------------------------------------------

/// Internal error classification. Not exposed to consumers.
#[derive(Debug)]
enum ErrorKind {
    /// The TensorFlow Lite C API returned a non-OK status.
    Status(StatusCode),
    /// A C API function returned a null pointer.
    NullPointer,
    /// Library loading or symbol resolution failed.
    Library(libloading::Error),
    /// An invalid argument was passed to the API.
    InvalidArgument(String),
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// The error type for all fallible operations in `edgefirst-tflite`.
///
/// `Error` wraps a private `ErrorKind` enum so that the set of failure modes
/// can grow without breaking callers. Use the `is_*()` inspection methods and
/// [`Error::status_code`] to classify an error programmatically.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    context: Option<String>,
}

// -- Public inspection API --------------------------------------------------

impl Error {
    /// Returns `true` if this error originated from library loading or symbol
    /// resolution (i.e. a [`libloading::Error`]).
    #[must_use]
    pub fn is_library_error(&self) -> bool {
        matches!(self.kind, ErrorKind::Library(_))
    }

    /// Returns `true` if the underlying `TFLite` status is one of the delegate
    /// error codes: [`StatusCode::DelegateError`],
    /// [`StatusCode::DelegateDataNotFound`],
    /// [`StatusCode::DelegateDataWriteError`], or
    /// [`StatusCode::DelegateDataReadError`].
    #[must_use]
    pub fn is_delegate_error(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::Status(
                StatusCode::DelegateError
                    | StatusCode::DelegateDataNotFound
                    | StatusCode::DelegateDataWriteError
                    | StatusCode::DelegateDataReadError
            )
        )
    }

    /// Returns `true` if a C API call returned a null pointer.
    #[must_use]
    pub fn is_null_pointer(&self) -> bool {
        matches!(self.kind, ErrorKind::NullPointer)
    }

    /// Returns the `TFLite` [`StatusCode`] when the error originated from a
    /// non-OK C API status, or `None` otherwise.
    #[must_use]
    pub fn status_code(&self) -> Option<StatusCode> {
        if let ErrorKind::Status(code) = self.kind {
            Some(code)
        } else {
            None
        }
    }

    /// Attach additional human-readable context to this error.
    ///
    /// The context string is appended in parentheses when the error is
    /// displayed.
    #[must_use]
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

// -- Crate-internal constructors --------------------------------------------

impl Error {
    /// Create an error from a `TFLite` [`StatusCode`].
    #[must_use]
    pub(crate) fn status(code: StatusCode) -> Self {
        Self {
            kind: ErrorKind::Status(code),
            context: None,
        }
    }

    /// Create a null-pointer error with a description of which pointer was
    /// null.
    #[must_use]
    pub(crate) fn null_pointer(context: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::NullPointer,
            context: Some(context.into()),
        }
    }

    /// Create an invalid-argument error.
    #[must_use]
    pub(crate) fn invalid_argument(msg: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidArgument(msg.into()),
            context: None,
        }
    }
}

// -- Display ----------------------------------------------------------------

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::Status(code) => write!(f, "TFLite status: {code}")?,
            ErrorKind::NullPointer => f.write_str("null pointer from C API")?,
            ErrorKind::Library(inner) => write!(f, "library loading error: {inner}")?,
            ErrorKind::InvalidArgument(msg) => write!(f, "invalid argument: {msg}")?,
        }
        if let Some(ctx) = &self.context {
            write!(f, " ({ctx})")?;
        }
        Ok(())
    }
}

// -- std::error::Error ------------------------------------------------------

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ErrorKind::Library(inner) => Some(inner),
            _ => None,
        }
    }
}

// -- From conversions -------------------------------------------------------

impl From<libloading::Error> for Error {
    fn from(err: libloading::Error) -> Self {
        Self {
            kind: ErrorKind::Library(err),
            context: None,
        }
    }
}

// ---------------------------------------------------------------------------
// status_to_result
// ---------------------------------------------------------------------------

/// Convert a raw `TFLite` C API status code to a [`Result`].
///
/// `kTfLiteOk` (0) maps to `Ok(())`. All other known values map to the
/// corresponding [`StatusCode`]. Unknown non-zero values are treated as
/// [`StatusCode::RuntimeError`].
pub(crate) fn status_to_result(status: u32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let code = StatusCode::from_raw(status).unwrap_or(StatusCode::RuntimeError);
    Err(Error::status(code))
}

// ---------------------------------------------------------------------------
// Result type alias
// ---------------------------------------------------------------------------

/// A [`Result`](std::result::Result) type alias using [`Error`] as the error
/// variant.
pub type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_ok_is_ok() {
        assert!(status_to_result(0).is_ok());
    }

    #[test]
    fn status_error_maps_correctly() {
        let err = status_to_result(1).unwrap_err();
        assert_eq!(err.status_code(), Some(StatusCode::RuntimeError));
    }

    #[test]
    fn status_delegate_codes() {
        for (raw, expected) in [
            (2, StatusCode::DelegateError),
            (4, StatusCode::DelegateDataNotFound),
            (5, StatusCode::DelegateDataWriteError),
            (6, StatusCode::DelegateDataReadError),
        ] {
            let err = status_to_result(raw).unwrap_err();
            assert_eq!(err.status_code(), Some(expected));
            assert!(err.is_delegate_error());
        }
    }

    #[test]
    fn status_all_known_codes() {
        for raw in 1..=9 {
            let err = status_to_result(raw).unwrap_err();
            assert!(err.status_code().is_some());
        }
    }

    #[test]
    fn unknown_status_falls_back_to_runtime_error() {
        let err = status_to_result(42).unwrap_err();
        assert_eq!(err.status_code(), Some(StatusCode::RuntimeError));
    }

    #[test]
    fn null_pointer_error() {
        let err = Error::null_pointer("TfLiteModelCreate");
        assert!(err.is_null_pointer());
        assert!(!err.is_library_error());
        assert!(!err.is_delegate_error());
        assert!(err.status_code().is_none());
        assert!(err.to_string().contains("null pointer"));
        assert!(err.to_string().contains("TfLiteModelCreate"));
    }

    #[test]
    fn invalid_argument_error() {
        let err = Error::invalid_argument("tensor index out of range");
        assert!(!err.is_null_pointer());
        assert!(err.to_string().contains("tensor index out of range"));
    }

    #[test]
    fn with_context_appends_message() {
        let err = Error::status(StatusCode::RuntimeError).with_context("during AllocateTensors");
        let msg = err.to_string();
        assert!(msg.contains("runtime error"));
        assert!(msg.contains("during AllocateTensors"));
    }

    #[test]
    fn from_libloading_error() {
        // Attempt to load a library that does not exist to obtain a
        // `libloading::Error`.
        let lib_err = unsafe { libloading::Library::new("__nonexistent__.so") }.unwrap_err();
        let err = Error::from(lib_err);
        assert!(err.is_library_error());
        assert!(err.status_code().is_none());
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn display_includes_status_code_name() {
        let err = Error::status(StatusCode::Cancelled);
        assert!(err.to_string().contains("cancelled"));
    }

    #[test]
    fn non_delegate_status_is_not_delegate_error() {
        let err = Error::status(StatusCode::RuntimeError);
        assert!(!err.is_delegate_error());
    }

    #[test]
    fn status_code_discriminant_values() {
        assert_eq!(StatusCode::RuntimeError as u32, 1);
        assert_eq!(StatusCode::DelegateError as u32, 2);
        assert_eq!(StatusCode::ApplicationError as u32, 3);
        assert_eq!(StatusCode::DelegateDataNotFound as u32, 4);
        assert_eq!(StatusCode::DelegateDataWriteError as u32, 5);
        assert_eq!(StatusCode::DelegateDataReadError as u32, 6);
        assert_eq!(StatusCode::UnresolvedOps as u32, 7);
        assert_eq!(StatusCode::Cancelled as u32, 8);
        assert_eq!(StatusCode::OutputShapeNotKnown as u32, 9);
    }

    #[test]
    fn status_code_display_all_variants() {
        let cases = [
            (StatusCode::RuntimeError, "runtime error"),
            (StatusCode::DelegateError, "delegate error"),
            (StatusCode::ApplicationError, "application error"),
            (StatusCode::DelegateDataNotFound, "delegate data not found"),
            (
                StatusCode::DelegateDataWriteError,
                "delegate data write error",
            ),
            (
                StatusCode::DelegateDataReadError,
                "delegate data read error",
            ),
            (StatusCode::UnresolvedOps, "unresolved ops"),
            (StatusCode::Cancelled, "cancelled"),
            (StatusCode::OutputShapeNotKnown, "output shape not known"),
        ];
        for (code, expected) in cases {
            assert_eq!(code.to_string(), expected);
        }
    }

    #[test]
    fn error_debug_format() {
        let err = Error::status(StatusCode::RuntimeError);
        let debug = format!("{err:?}");
        assert!(debug.contains("Error"));
        assert!(debug.contains("Status"));
    }
}
