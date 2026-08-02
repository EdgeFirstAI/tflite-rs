// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Error types for the `edgefirst-tflite` crate.
//!
//! This module follows the canonical error struct pattern: a public [`Error`]
//! struct wrapping a private `ErrorKind` enum. Callers inspect errors through
//! [`Error::is_library_error`], [`Error::is_delegate_error`],
//! [`Error::is_null_pointer`], [`Error::is_litert_unavailable`],
//! [`Error::status_code`], [`Error::litert_status_code`], and
//! [`Error::litert_missing_symbol`] rather than matching on variants directly.

use std::fmt;

use edgefirst_tflite_sys::litert::MissingSymbol;

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

/// A raw `LiteRtStatus` value returned by the `LiteRT` C API.
///
/// The wrapped `u32` is the value defined by `LiteRtStatus` in
/// `litert_common.h`. Obtain one from [`Error::litert_status_code`]; compare it
/// against the constants re-exported by
/// [`edgefirst_tflite_sys::litert::status`] rather than against integer
/// literals, so that a header re-vendor cannot silently change the meaning of a
/// comparison.
///
/// # Examples
///
/// ```no_run
/// use edgefirst_tflite::{Library, litert};
/// use edgefirst_tflite_sys::litert::status;
///
/// let lib = Library::new()?;
/// let env = litert::Environment::new(&lib)?;
/// match litert::Model::from_file(&env, "missing.tflite") {
///     Err(e) if e.litert_status_code().map(|c| c.raw()) == Some(status::ERROR_FILE_IO) => {
///         eprintln!("model file could not be read");
///     }
///     Err(e) => eprintln!("load failed: {e}"),
///     Ok(_) => {}
/// }
/// # Ok::<(), edgefirst_tflite::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiteRtStatusCode(pub u32);

impl LiteRtStatusCode {
    /// The raw `LiteRtStatus` value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for LiteRtStatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use edgefirst_tflite_sys::litert::status;

        // Mapped from the sys constants rather than literals so the table
        // cannot drift from the vendored headers.
        let name = match self.0 {
            status::OK => "ok",
            status::ERROR_INVALID_ARGUMENT => "invalid argument",
            status::ERROR_MEMORY_ALLOCATION_FAILURE => "memory allocation failure",
            status::ERROR_RUNTIME_FAILURE => "runtime failure",
            status::ERROR_MISSING_INPUT_TENSOR => "missing input tensor",
            status::ERROR_UNSUPPORTED => "unsupported",
            status::ERROR_NOT_FOUND => "not found",
            status::ERROR_TIMEOUT_EXPIRED => "timeout expired",
            status::ERROR_WRONG_VERSION => "wrong version",
            status::ERROR_UNKNOWN => "unknown",
            status::ERROR_ALREADY_EXISTS => "already exists",
            status::CANCELLED => "cancelled",
            status::ERROR_FILE_IO => "file io",
            status::ERROR_INVALID_FLATBUFFER => "invalid flatbuffer",
            status::ERROR_DYNAMIC_LOADING => "dynamic loading",
            status::ERROR_SERIALIZATION => "serialization",
            status::ERROR_COMPILATION => "compilation",
            status::ERROR_INDEX_OOB => "index out of bounds",
            status::ERROR_INVALID_IR_TYPE => "invalid IR type",
            status::ERROR_INVALID_GRAPH_INVARIANT => "invalid graph invariant",
            status::ERROR_GRAPH_MODIFICATION => "graph modification",
            status::ERROR_INVALID_TOOL_CONFIG => "invalid tool config",
            status::LEGALIZE_NO_MATCH => "legalize: no match",
            status::ERROR_INVALID_LEGALIZATION => "invalid legalization",
            status::PATTERN_NO_MATCH => "pattern: no match",
            status::INVALID_TRANSFORMATION => "invalid transformation",
            status::ERROR_UNSUPPORTED_RUNTIME_VERSION => "unsupported runtime version",
            status::ERROR_UNSUPPORTED_COMPILER_VERSION => "unsupported compiler version",
            status::ERROR_INCOMPATIBLE_BYTE_CODE_VERSION => "incompatible byte code version",
            status::ERROR_UNSUPPORTED_OP_SHAPE_INFERER => "unsupported op shape inferer",
            other => return write!(f, "status {other}"),
        };
        f.write_str(name)
    }
}

/// Internal error classification. Not exposed to consumers.
#[derive(Debug)]
enum ErrorKind {
    /// The TensorFlow Lite C API returned a non-OK status.
    Status(StatusCode),
    /// The `LiteRT` C API returned a non-OK status.
    LiteRtStatus(LiteRtStatusCode),
    /// `LiteRT` symbols are not available in the loaded shared library.
    ///
    /// Carries the name of the first symbol that failed to resolve, which
    /// distinguishes "this is a classic `TFLite` library" from "this is a
    /// partial or version-skewed `LiteRT` build".
    LiteRtUnavailable(&'static str),
    /// A C API function returned a null pointer.
    NullPointer,
    /// Library loading or symbol resolution failed.
    Library(libloading::Error),
    /// An invalid argument was passed to the API.
    InvalidArgument(String),
    /// The loaded runtime does not export an optional API the call needs.
    ///
    /// Distinct from [`ErrorKind::InvalidArgument`]: the call was well-formed,
    /// but this build of `TFLite` cannot service it. Carries the API's name.
    Unsupported(&'static str),
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

    /// Returns `true` if this error is an invalid-argument error.
    #[must_use]
    pub fn is_invalid_argument(&self) -> bool {
        matches!(self.kind, ErrorKind::InvalidArgument(_))
    }

    /// Returns `true` if the loaded runtime does not export an optional API the
    /// call required.
    ///
    /// The call itself was well-formed — this build of `TFLite` simply cannot
    /// service it. [`Error::unsupported_api`] names which one.
    #[must_use]
    pub fn is_unsupported(&self) -> bool {
        matches!(self.kind, ErrorKind::Unsupported(_))
    }

    /// The name of the unavailable API, when this is an unsupported-API error.
    ///
    /// For example `"TfLiteInterpreterSetCustomAllocationForTensor"` from
    /// [`Interpreter::set_custom_allocation_for_input`](crate::Interpreter::set_custom_allocation_for_input)
    /// on a runtime built without the experimental C API.
    #[must_use]
    pub fn unsupported_api(&self) -> Option<&'static str> {
        if let ErrorKind::Unsupported(api) = self.kind {
            Some(api)
        } else {
            None
        }
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

    /// Returns `true` if `LiteRT` symbols are missing from the loaded library.
    ///
    /// This is the expected outcome on a classic TensorFlow Lite build and is
    /// not a defect — [`crate::Interpreter`] remains fully available. Use
    /// [`Error::litert_missing_symbol`] to tell "no `LiteRT` at all" apart from
    /// "partial `LiteRT`".
    #[must_use]
    pub fn is_litert_unavailable(&self) -> bool {
        matches!(self.kind, ErrorKind::LiteRtUnavailable(_))
    }

    /// Returns the name of the `LiteRT` symbol that failed to resolve.
    ///
    /// `Some` only for errors where [`Error::is_litert_unavailable`] is `true`.
    /// A value of `"LiteRtCreateEnvironment"` — the first symbol probed — means
    /// the library exports no `LiteRT` surface at all. Any later symbol means
    /// the library is a partial or version-skewed `LiteRT` build, which is
    /// usually a packaging problem worth reporting.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use edgefirst_tflite::{Library, litert};
    ///
    /// let lib = Library::new()?;
    /// if let Err(e) = litert::Environment::new(&lib) {
    ///     match e.litert_missing_symbol() {
    ///         Some("LiteRtCreateEnvironment") => println!("classic TFLite library"),
    ///         Some(sym) => println!("partial LiteRT build; missing {sym}"),
    ///         None => println!("LiteRT present, but creation failed: {e}"),
    ///     }
    /// }
    /// # Ok::<(), edgefirst_tflite::Error>(())
    /// ```
    #[must_use]
    pub fn litert_missing_symbol(&self) -> Option<&'static str> {
        if let ErrorKind::LiteRtUnavailable(symbol) = self.kind {
            Some(symbol)
        } else {
            None
        }
    }

    /// Returns the `LiteRT` [`LiteRtStatusCode`] when the error originated from
    /// a non-OK `LiteRT` C API status, or `None` otherwise.
    #[must_use]
    pub fn litert_status_code(&self) -> Option<LiteRtStatusCode> {
        if let ErrorKind::LiteRtStatus(code) = self.kind {
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

    /// Create an error for an optional API the loaded runtime does not export.
    ///
    /// `api` names the missing entry point; `context` should say what the
    /// caller can do instead.
    #[must_use]
    pub(crate) fn unsupported(api: &'static str, context: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Unsupported(api),
            context: Some(context.into()),
        }
    }

    /// Create an error indicating `LiteRT` is not present in the loaded library.
    ///
    /// `missing` is the first `LiteRt*` symbol that failed to resolve.
    #[must_use]
    pub(crate) fn litert_unavailable(missing: MissingSymbol) -> Self {
        let symbol = missing.name();
        let context = if symbol == "LiteRtCreateEnvironment" {
            format!(
                "no LiteRt* symbols found ({symbol} unresolved); this is a classic \
                 TensorFlow Lite library — use Interpreter instead"
            )
        } else {
            format!(
                "incomplete LiteRT build: {symbol} unresolved; the library exports \
                 some LiteRt* symbols but not the full required set"
            )
        };
        Self {
            kind: ErrorKind::LiteRtUnavailable(symbol),
            context: Some(context),
        }
    }

    /// Create an error from a raw `LiteRT` status code.
    #[must_use]
    pub(crate) fn litert_status(code: u32) -> Self {
        Self {
            kind: ErrorKind::LiteRtStatus(LiteRtStatusCode(code)),
            context: None,
        }
    }
}

// -- Display ----------------------------------------------------------------

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::Status(code) => write!(f, "TFLite status: {code}")?,
            ErrorKind::LiteRtStatus(code) => write!(f, "LiteRT status: {code}")?,
            ErrorKind::LiteRtUnavailable(_) => f.write_str("LiteRT unavailable")?,
            ErrorKind::NullPointer => f.write_str("null pointer from C API")?,
            ErrorKind::Library(inner) => write!(f, "library loading error: {inner}")?,
            ErrorKind::InvalidArgument(msg) => write!(f, "invalid argument: {msg}")?,
            ErrorKind::Unsupported(api) => write!(f, "unsupported by this runtime: {api}")?,
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
// hal_to_result
// ---------------------------------------------------------------------------

/// Convert a HAL DMA-BUF return code to a [`Result`].
///
/// HAL functions return `0` on success and `-1` on error (with `errno` set).
/// On failure, the errno value is captured via [`std::io::Error::last_os_error`]
/// and included in the error context.
pub(crate) fn hal_to_result(ret: std::ffi::c_int, context: &str) -> Result<()> {
    if ret == 0 {
        return Ok(());
    }
    let os_err = std::io::Error::last_os_error();
    Err(Error::status(StatusCode::DelegateError).with_context(format!("{context}: {os_err}")))
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

/// Convert a raw `LiteRT` C API status code to a [`Result`].
///
/// `kLiteRtStatusOk` maps to `Ok(())`; every other value is preserved verbatim
/// in a [`LiteRtStatusCode`] so that callers can distinguish, say, a missing
/// file from a compilation failure.
pub(crate) fn litert_status_to_result(status: u32) -> Result<()> {
    if status == edgefirst_tflite_sys::litert::status::OK {
        return Ok(());
    }
    Err(Error::litert_status(status))
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

    #[test]
    fn litert_unavailable_error() {
        let err = Error::litert_unavailable(MissingSymbol("LiteRtCreateEnvironment"));
        assert!(err.is_litert_unavailable());
        assert!(!err.is_library_error());
        assert!(err.litert_status_code().is_none());
        assert_eq!(err.litert_missing_symbol(), Some("LiteRtCreateEnvironment"));
        assert!(err.to_string().contains("LiteRT unavailable"));
        // The first symbol probed means "no LiteRT at all", not a partial build.
        assert!(err.to_string().contains("classic"));
    }

    #[test]
    fn litert_partial_build_error_names_symbol() {
        let err = Error::litert_unavailable(MissingSymbol("LiteRtCreateCompiledModel"));
        assert_eq!(
            err.litert_missing_symbol(),
            Some("LiteRtCreateCompiledModel")
        );
        let text = err.to_string();
        assert!(text.contains("incomplete LiteRT build"), "{text}");
        assert!(text.contains("LiteRtCreateCompiledModel"), "{text}");
    }

    #[test]
    fn litert_status_error() {
        use edgefirst_tflite_sys::litert::status;

        let err = litert_status_to_result(status::ERROR_RUNTIME_FAILURE).unwrap_err();
        assert_eq!(
            err.litert_status_code(),
            Some(LiteRtStatusCode(status::ERROR_RUNTIME_FAILURE))
        );
        assert_eq!(
            err.litert_status_code().map(LiteRtStatusCode::raw),
            Some(status::ERROR_RUNTIME_FAILURE)
        );
        assert!(err.to_string().contains("runtime failure"));
        assert!(litert_status_to_result(status::OK).is_ok());
        assert!(err.litert_missing_symbol().is_none());
    }

    #[test]
    fn litert_status_display_covers_extended_codes() {
        use edgefirst_tflite_sys::litert::status;

        // Codes above the 500 block must still render by name, and genuinely
        // unknown values must not panic.
        assert_eq!(
            LiteRtStatusCode(status::ERROR_INDEX_OOB).to_string(),
            "index out of bounds"
        );
        assert_eq!(
            LiteRtStatusCode(status::ERROR_COMPILATION).to_string(),
            "compilation"
        );
        assert_eq!(LiteRtStatusCode(9_999).to_string(), "status 9999");
    }
}
