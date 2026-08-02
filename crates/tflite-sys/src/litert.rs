// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Soft-optional `LiteRT` Next (`LiteRt*`) bindings.
//!
//! `LiteRT` Next symbols live in the *same* shared object as the classic
//! TensorFlow Lite C API on hosts that ship it (the Android `libLiteRt.so`
//! exports both `LiteRt*` and `TfLite*`). This module therefore resolves
//! `LiteRt*` from an already-open [`libloading::Library`] handle rather than
//! opening a second object.
//!
//! Call [`LiteRtFunctions::try_load`] to resolve the surface used by the
//! high-level `edgefirst-tflite` `litert` module. It reports the *name* of the
//! first unresolved symbol instead of collapsing every failure into a single
//! "unavailable", so a partial `LiteRT` build can be diagnosed.
//!
//! # Relationship to the generated bindings
//!
//! Types and constants come from bindgen (`litert_ffi.rs`, generated from the
//! vendored headers under `litert/`). The function-pointer table below is
//! written by hand — the same pattern as [`crate::xnnpack_ffi`] — because a
//! bindgen `--dynamic-loading` table resolves every symbol eagerly and cannot
//! express "optional". To stop the two from drifting, the
//! `litert_functions!` macro emits a compile-time cross-check asserting that
//! each hand-written signature is *identical* to the bindgen-generated one.
//! A mismatch after re-vendoring headers is a build error, not a silent ABI
//! bug.

use std::ffi::{c_char, c_int, c_void};
use std::fmt;

use crate::litert_ffi::{
    LiteRtAccelerator, LiteRtAcceleratorId, LiteRtCompiledModel, LiteRtEnvOption,
    LiteRtEnvironment, LiteRtHwAcceleratorSet, LiteRtModel, LiteRtOptions, LiteRtParamIndex,
    LiteRtRankedTensorType, LiteRtSignature, LiteRtStatus, LiteRtTensor, LiteRtTensorBuffer,
    LiteRtTensorBufferLockMode, LiteRtTensorBufferRequirements, LiteRtTensorBufferType,
};

pub use crate::litert_ffi::{
    LiteRtElementType,
    LiteRtEnvOption as EnvOption,
    LiteRtHwAccelerators,
    LiteRtHwAccelerators_kLiteRtHwAcceleratorCpu as kLiteRtHwAcceleratorCpu,
    LiteRtHwAccelerators_kLiteRtHwAcceleratorGpu as kLiteRtHwAcceleratorGpu,
    LiteRtHwAccelerators_kLiteRtHwAcceleratorNone as kLiteRtHwAcceleratorNone,
    // `kLiteRtHwAcceleratorWebNn` (bit 3) is guarded by `#if
    // defined(__EMSCRIPTEN__)` upstream and so is absent from these bindings on
    // every target this crate supports. It is deliberately not re-exported.
    LiteRtHwAccelerators_kLiteRtHwAcceleratorNpu as kLiteRtHwAcceleratorNpu,
    LiteRtLayout,
    LiteRtStatus_kLiteRtStatusOk as kLiteRtStatusOk,
    LiteRtTensorBufferLockMode_kLiteRtTensorBufferLockModeRead as kLiteRtTensorBufferLockModeRead,
    LiteRtTensorBufferLockMode_kLiteRtTensorBufferLockModeReadWrite as kLiteRtTensorBufferLockModeReadWrite,
    LiteRtTensorBufferLockMode_kLiteRtTensorBufferLockModeWrite as kLiteRtTensorBufferLockModeWrite,
    LiteRtTensorBufferType_kLiteRtTensorBufferTypeHostMemory as kLiteRtTensorBufferTypeHostMemory,
};

// Opaque handle / status type re-exports for downstream crates.
pub type Environment = LiteRtEnvironment;
pub type Model = LiteRtModel;
pub type Options = LiteRtOptions;
pub type CompiledModel = LiteRtCompiledModel;
pub type TensorBuffer = LiteRtTensorBuffer;
pub type TensorBufferRequirements = LiteRtTensorBufferRequirements;
pub type Accelerator = LiteRtAccelerator;
pub type AcceleratorId = LiteRtAcceleratorId;
pub type ParamIndex = LiteRtParamIndex;
pub type HwAcceleratorSet = LiteRtHwAcceleratorSet;
pub type RankedTensorType = LiteRtRankedTensorType;
pub type Status = LiteRtStatus;
pub type TensorBufferType = LiteRtTensorBufferType;
pub type TensorBufferLockMode = LiteRtTensorBufferLockMode;
pub type Signature = LiteRtSignature;
pub type Tensor = LiteRtTensor;

/// The name of a `LiteRT` symbol that could not be resolved.
///
/// Returned by [`LiteRtFunctions::try_load`]. Reporting *which* symbol is
/// absent distinguishes a classic TensorFlow Lite library (the very first
/// symbol fails) from a partial or version-skewed `LiteRT` build (a later
/// symbol fails), which are very different diagnoses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingSymbol(pub &'static str);

impl MissingSymbol {
    /// The unresolved symbol name, e.g. `"LiteRtCreateCompiledModel"`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for MissingSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unresolved LiteRT symbol: {}", self.0)
    }
}

impl std::error::Error for MissingSymbol {}

/// Declares the `LiteRT` function table, its loader, and an ABI cross-check.
///
/// Each entry binds a Rust field name to a C symbol name and a signature. The
/// macro emits three things from that single source:
///
/// 1. the [`LiteRtFunctions`] struct,
/// 2. [`LiteRtFunctions::try_load`], which resolves each symbol in order and
///    returns [`MissingSymbol`] naming the first failure, and
/// 3. `abi_cross_check`, a never-called function whose body fails to compile
///    unless every signature matches the bindgen-generated table exactly.
macro_rules! litert_functions {
    ($(
        $(#[$attr:meta])*
        $field:ident => $sym:ident : unsafe extern "C" fn($($arg:ident : $arg_ty:ty),* $(,)?) $(-> $ret:ty)?
    ),* $(,)?) => {
        /// Resolved `LiteRT` function pointers.
        ///
        /// Construct with [`LiteRtFunctions::try_load`]. Fields are public so
        /// that downstream crates can call the C API directly; every call is
        /// `unsafe` and carries the usual FFI obligations.
        ///
        /// The table is [`Copy`], but the pointers are only valid while the
        /// originating shared library remains loaded. Prefer holding it behind
        /// a reference borrowed from the object that owns the library handle
        /// (this is what `edgefirst_tflite::Library::litert` does).
        #[derive(Debug, Clone, Copy)]
        #[allow(clippy::struct_field_names)]
        pub struct LiteRtFunctions {
            $(
                $(#[$attr])*
                pub $field: unsafe extern "C" fn($($arg: $arg_ty),*) $(-> $ret)?,
            )*
        }

        impl LiteRtFunctions {
            /// Resolve the `LiteRT` surface used by `edgefirst-tflite` from `lib`.
            ///
            /// Symbols are resolved in declaration order and the first failure
            /// short-circuits, so the returned [`MissingSymbol`] names the
            /// earliest unresolved entry. A classic TensorFlow Lite library
            /// fails on `LiteRtCreateEnvironment`.
            ///
            /// # Errors
            ///
            /// Returns [`MissingSymbol`] when any required symbol is absent.
            /// This is the expected outcome on TFLite-only libraries and is not
            /// an error condition for callers that treat `LiteRT` as optional.
            ///
            /// # Safety
            ///
            /// `lib` must remain loaded for as long as the returned table (or
            /// any copy of it) is used. The caller must also guarantee that
            /// `lib` really is a `LiteRT` build — a library that exports these
            /// names with different signatures would produce undefined
            /// behaviour on call.
            pub unsafe fn try_load(lib: &libloading::Library) -> Result<Self, MissingSymbol> {
                // SAFETY: each `get` resolves one symbol from the loaded
                // library; the `?` short-circuits before any pointer is stored
                // if a symbol is absent. Signatures are checked against the
                // bindgen-generated table by `abi_cross_check` below.
                unsafe {
                    Ok(Self {
                        $(
                            $field: *lib
                                .get(concat!(stringify!($sym), "\0").as_bytes())
                                .map_err(|_| MissingSymbol(stringify!($sym)))?,
                        )*
                    })
                }
            }

            /// Number of symbols [`LiteRtFunctions::try_load`] must resolve.
            #[must_use]
            pub const fn required_symbol_count() -> usize {
                [$(stringify!($sym)),*].len()
            }

            /// Names of every symbol [`LiteRtFunctions::try_load`] requires.
            ///
            /// Useful for diagnostics — e.g. reporting which of these a given
            /// shared library actually exports.
            #[must_use]
            pub const fn required_symbols() -> &'static [&'static str] {
                &[$(stringify!($sym)),*]
            }
        }

        /// Compile-time proof that every hand-written signature above is
        /// identical to the bindgen-generated one.
        ///
        /// Never called. The array type annotation forces the generated field
        /// type and ours to unify; if a re-vendored header changes an argument
        /// or return type, this stops compiling.
        #[allow(dead_code, clippy::used_underscore_items)]
        fn abi_cross_check(generated: &crate::litert_ffi::litert, ours: &LiteRtFunctions) {
            $({
                let _pair: [Option<&unsafe extern "C" fn($($arg_ty),*) $(-> $ret)?>; 2] =
                    [generated.$sym.as_ref().ok(), Some(&ours.$field)];
            })*
        }
    };
}

litert_functions! {
    // -- Environment --------------------------------------------------------
    /// `LiteRtCreateEnvironment` — create an owned environment.
    create_environment => LiteRtCreateEnvironment: unsafe extern "C" fn(
        num_options: c_int,
        options: *const LiteRtEnvOption,
        environment: *mut LiteRtEnvironment,
    ) -> LiteRtStatus,
    /// `LiteRtDestroyEnvironment` — destroy an environment created above.
    destroy_environment => LiteRtDestroyEnvironment: unsafe extern "C" fn(
        environment: LiteRtEnvironment,
    ),

    // -- Model --------------------------------------------------------------
    /// `LiteRtCreateModelFromFile` — load a `.tflite` file.
    create_model_from_file => LiteRtCreateModelFromFile: unsafe extern "C" fn(
        environment: LiteRtEnvironment,
        filename: *const c_char,
        model: *mut LiteRtModel,
    ) -> LiteRtStatus,
    /// `LiteRtCreateModelFromBuffer` — load from memory the caller keeps alive.
    create_model_from_buffer => LiteRtCreateModelFromBuffer: unsafe extern "C" fn(
        environment: LiteRtEnvironment,
        buffer_addr: *const c_void,
        buffer_size: usize,
        model: *mut LiteRtModel,
    ) -> LiteRtStatus,
    /// `LiteRtDestroyModel` — destroy a model.
    destroy_model => LiteRtDestroyModel: unsafe extern "C" fn(model: LiteRtModel),
    /// `LiteRtGetNumModelSignatures` — signature count.
    get_num_model_signatures => LiteRtGetNumModelSignatures: unsafe extern "C" fn(
        model: LiteRtModel,
        num_signatures: *mut LiteRtParamIndex,
    ) -> LiteRtStatus,
    /// `LiteRtGetModelSignature` — borrow a signature (valid while the model is).
    get_model_signature => LiteRtGetModelSignature: unsafe extern "C" fn(
        model: LiteRtModel,
        signature_index: LiteRtParamIndex,
        signature: *mut LiteRtSignature,
    ) -> LiteRtStatus,
    /// `LiteRtGetNumSignatureInputs` — input count for a signature.
    get_num_signature_inputs => LiteRtGetNumSignatureInputs: unsafe extern "C" fn(
        signature: LiteRtSignature,
        num_inputs: *mut LiteRtParamIndex,
    ) -> LiteRtStatus,
    /// `LiteRtGetNumSignatureOutputs` — output count for a signature.
    get_num_signature_outputs => LiteRtGetNumSignatureOutputs: unsafe extern "C" fn(
        signature: LiteRtSignature,
        num_outputs: *mut LiteRtParamIndex,
    ) -> LiteRtStatus,
    /// `LiteRtGetSignatureInputTensorByIndex` — borrow an input tensor.
    get_signature_input_tensor_by_index => LiteRtGetSignatureInputTensorByIndex:
        unsafe extern "C" fn(
            signature: LiteRtSignature,
            input_idx: LiteRtParamIndex,
            tensor: *mut LiteRtTensor,
        ) -> LiteRtStatus,
    /// `LiteRtGetSignatureOutputTensorByIndex` — borrow an output tensor.
    get_signature_output_tensor_by_index => LiteRtGetSignatureOutputTensorByIndex:
        unsafe extern "C" fn(
            signature: LiteRtSignature,
            output_idx: LiteRtParamIndex,
            tensor: *mut LiteRtTensor,
        ) -> LiteRtStatus,
    /// `LiteRtGetRankedTensorType` — element type plus layout for a tensor.
    get_ranked_tensor_type => LiteRtGetRankedTensorType: unsafe extern "C" fn(
        tensor: LiteRtTensor,
        ranked_tensor_type: *mut LiteRtRankedTensorType,
    ) -> LiteRtStatus,

    // -- Options ------------------------------------------------------------
    /// `LiteRtCreateOptions` — create owned compilation options.
    create_options => LiteRtCreateOptions: unsafe extern "C" fn(
        options: *mut LiteRtOptions,
    ) -> LiteRtStatus,
    /// `LiteRtDestroyOptions` — destroy compilation options.
    destroy_options => LiteRtDestroyOptions: unsafe extern "C" fn(options: LiteRtOptions),
    /// `LiteRtSetOptionsHardwareAccelerators` — set the accelerator bitmask.
    set_options_hardware_accelerators => LiteRtSetOptionsHardwareAccelerators:
        unsafe extern "C" fn(
            options: LiteRtOptions,
            hardware_accelerators: LiteRtHwAcceleratorSet,
        ) -> LiteRtStatus,
    /// `LiteRtGetOptionsHardwareAccelerators` — read the accelerator bitmask.
    get_options_hardware_accelerators => LiteRtGetOptionsHardwareAccelerators:
        unsafe extern "C" fn(
            options: LiteRtOptions,
            hardware_accelerators: *mut LiteRtHwAcceleratorSet,
        ) -> LiteRtStatus,

    // -- CompiledModel ------------------------------------------------------
    /// `LiteRtCreateCompiledModel` — compile a model; the model must outlive it.
    create_compiled_model => LiteRtCreateCompiledModel: unsafe extern "C" fn(
        environment: LiteRtEnvironment,
        model: LiteRtModel,
        compilation_options: LiteRtOptions,
        compiled_model: *mut LiteRtCompiledModel,
    ) -> LiteRtStatus,
    /// `LiteRtDestroyCompiledModel` — destroy a compiled model.
    destroy_compiled_model => LiteRtDestroyCompiledModel: unsafe extern "C" fn(
        compiled_model: LiteRtCompiledModel,
    ),
    /// `LiteRtRunCompiledModel` — synchronous inference.
    run_compiled_model => LiteRtRunCompiledModel: unsafe extern "C" fn(
        compiled_model: LiteRtCompiledModel,
        signature_index: LiteRtParamIndex,
        num_input_buffers: usize,
        input_buffers: *mut LiteRtTensorBuffer,
        num_output_buffers: usize,
        output_buffers: *mut LiteRtTensorBuffer,
    ) -> LiteRtStatus,
    /// `LiteRtGetCompiledModelInputBufferRequirements` — borrowed requirements.
    get_compiled_model_input_buffer_requirements =>
        LiteRtGetCompiledModelInputBufferRequirements: unsafe extern "C" fn(
            compiled_model: LiteRtCompiledModel,
            signature_index: LiteRtParamIndex,
            input_index: LiteRtParamIndex,
            buffer_requirements: *mut LiteRtTensorBufferRequirements,
        ) -> LiteRtStatus,
    /// `LiteRtGetCompiledModelOutputBufferRequirements` — borrowed requirements.
    get_compiled_model_output_buffer_requirements =>
        LiteRtGetCompiledModelOutputBufferRequirements: unsafe extern "C" fn(
            compiled_model: LiteRtCompiledModel,
            signature_index: LiteRtParamIndex,
            output_index: LiteRtParamIndex,
            buffer_requirements: *mut LiteRtTensorBufferRequirements,
        ) -> LiteRtStatus,
    /// `LiteRtCompiledModelIsFullyAccelerated` — dispatch-completeness predicate.
    compiled_model_is_fully_accelerated => LiteRtCompiledModelIsFullyAccelerated:
        unsafe extern "C" fn(
            compiled_model: LiteRtCompiledModel,
            fully_accelerated: *mut bool,
        ) -> LiteRtStatus,

    // -- TensorBuffer -------------------------------------------------------
    /// `LiteRtCreateManagedTensorBuffer` — allocate a runtime-managed buffer.
    create_managed_tensor_buffer => LiteRtCreateManagedTensorBuffer: unsafe extern "C" fn(
        env: LiteRtEnvironment,
        buffer_type: LiteRtTensorBufferType,
        tensor_type: *const LiteRtRankedTensorType,
        buffer_size: usize,
        buffer: *mut LiteRtTensorBuffer,
    ) -> LiteRtStatus,
    /// `LiteRtCreateManagedTensorBufferFromRequirements` — allocate to spec.
    create_managed_tensor_buffer_from_requirements =>
        LiteRtCreateManagedTensorBufferFromRequirements: unsafe extern "C" fn(
            env: LiteRtEnvironment,
            tensor_type: *const LiteRtRankedTensorType,
            requirements: LiteRtTensorBufferRequirements,
            buffer: *mut LiteRtTensorBuffer,
        ) -> LiteRtStatus,
    /// `LiteRtDestroyTensorBuffer` — destroy a tensor buffer.
    destroy_tensor_buffer => LiteRtDestroyTensorBuffer: unsafe extern "C" fn(
        buffer: LiteRtTensorBuffer,
    ),
    /// `LiteRtLockTensorBuffer` — map a buffer into host memory.
    lock_tensor_buffer => LiteRtLockTensorBuffer: unsafe extern "C" fn(
        tensor_buffer: LiteRtTensorBuffer,
        host_mem_addr: *mut *mut c_void,
        lock_mode: LiteRtTensorBufferLockMode,
    ) -> LiteRtStatus,
    /// `LiteRtUnlockTensorBuffer` — release a mapping taken above.
    unlock_tensor_buffer => LiteRtUnlockTensorBuffer: unsafe extern "C" fn(
        buffer: LiteRtTensorBuffer,
    ) -> LiteRtStatus,
    /// `LiteRtGetTensorBufferSize` — the buffer's actual allocated size.
    get_tensor_buffer_size => LiteRtGetTensorBufferSize: unsafe extern "C" fn(
        tensor_buffer: LiteRtTensorBuffer,
        size: *mut usize,
    ) -> LiteRtStatus,
    /// `LiteRtGetTensorBufferTensorType` — element type and layout.
    get_tensor_buffer_tensor_type => LiteRtGetTensorBufferTensorType: unsafe extern "C" fn(
        tensor_buffer: LiteRtTensorBuffer,
        tensor_type: *mut LiteRtRankedTensorType,
    ) -> LiteRtStatus,

    // -- Buffer requirements ------------------------------------------------
    /// `LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes` — type count.
    get_num_tensor_buffer_requirements_supported_buffer_types =>
        LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes: unsafe extern "C" fn(
            requirements: LiteRtTensorBufferRequirements,
            num_types: *mut c_int,
        ) -> LiteRtStatus,
    /// `LiteRtGetTensorBufferRequirementsSupportedTensorBufferType` — nth type.
    get_tensor_buffer_requirements_supported_tensor_buffer_type =>
        LiteRtGetTensorBufferRequirementsSupportedTensorBufferType: unsafe extern "C" fn(
            requirements: LiteRtTensorBufferRequirements,
            type_index: c_int,
            buffer_type: *mut LiteRtTensorBufferType,
        ) -> LiteRtStatus,
    /// `LiteRtGetTensorBufferRequirementsBufferSize` — required byte size.
    get_tensor_buffer_requirements_buffer_size =>
        LiteRtGetTensorBufferRequirementsBufferSize: unsafe extern "C" fn(
            requirements: LiteRtTensorBufferRequirements,
            buffer_size: *mut usize,
        ) -> LiteRtStatus,

    // -- Accelerator registry -----------------------------------------------
    /// `LiteRtGetNumAccelerators` — accelerators registered to an environment.
    get_num_accelerators => LiteRtGetNumAccelerators: unsafe extern "C" fn(
        environment: LiteRtEnvironment,
        num_accelerators: *mut LiteRtParamIndex,
    ) -> LiteRtStatus,
    /// `LiteRtGetAccelerator` — borrow the nth registered accelerator.
    get_accelerator => LiteRtGetAccelerator: unsafe extern "C" fn(
        environment: LiteRtEnvironment,
        index: LiteRtParamIndex,
        accelerator: *mut LiteRtAccelerator,
    ) -> LiteRtStatus,
    /// `LiteRtGetAcceleratorId` — runtime-assigned identifier.
    get_accelerator_id => LiteRtGetAcceleratorId: unsafe extern "C" fn(
        accelerator: LiteRtAccelerator,
        id: *mut LiteRtAcceleratorId,
    ) -> LiteRtStatus,
    /// `LiteRtGetAcceleratorName` — borrowed, NUL-terminated name.
    get_accelerator_name => LiteRtGetAcceleratorName: unsafe extern "C" fn(
        accelerator: LiteRtAccelerator,
        name: *mut *const c_char,
    ) -> LiteRtStatus,
    /// `LiteRtGetAcceleratorHardwareSupport` — supported hardware bitmask.
    get_accelerator_hardware_support => LiteRtGetAcceleratorHardwareSupport:
        unsafe extern "C" fn(
            accelerator: LiteRtAccelerator,
            supported_hardware: *mut LiteRtHwAcceleratorSet,
        ) -> LiteRtStatus,
}

/// `LiteRtStatus` constants, re-exported under short names.
///
/// These are the canonical numeric values; prefer them over literals so that a
/// header re-vendor cannot silently change a mapping.
pub mod status {
    pub use crate::litert_ffi::{
        LiteRtStatus_kLiteRtStatusCancelled as CANCELLED,
        LiteRtStatus_kLiteRtStatusErrorAlreadyExists as ERROR_ALREADY_EXISTS,
        LiteRtStatus_kLiteRtStatusErrorCompilation as ERROR_COMPILATION,
        LiteRtStatus_kLiteRtStatusErrorDynamicLoading as ERROR_DYNAMIC_LOADING,
        LiteRtStatus_kLiteRtStatusErrorFileIO as ERROR_FILE_IO,
        LiteRtStatus_kLiteRtStatusErrorGraphModification as ERROR_GRAPH_MODIFICATION,
        LiteRtStatus_kLiteRtStatusErrorIncompatibleByteCodeVersion as ERROR_INCOMPATIBLE_BYTE_CODE_VERSION,
        LiteRtStatus_kLiteRtStatusErrorIndexOOB as ERROR_INDEX_OOB,
        LiteRtStatus_kLiteRtStatusErrorInvalidArgument as ERROR_INVALID_ARGUMENT,
        LiteRtStatus_kLiteRtStatusErrorInvalidFlatbuffer as ERROR_INVALID_FLATBUFFER,
        LiteRtStatus_kLiteRtStatusErrorInvalidGraphInvariant as ERROR_INVALID_GRAPH_INVARIANT,
        LiteRtStatus_kLiteRtStatusErrorInvalidIrType as ERROR_INVALID_IR_TYPE,
        LiteRtStatus_kLiteRtStatusErrorInvalidLegalization as ERROR_INVALID_LEGALIZATION,
        LiteRtStatus_kLiteRtStatusErrorInvalidToolConfig as ERROR_INVALID_TOOL_CONFIG,
        LiteRtStatus_kLiteRtStatusErrorMemoryAllocationFailure as ERROR_MEMORY_ALLOCATION_FAILURE,
        LiteRtStatus_kLiteRtStatusErrorMissingInputTensor as ERROR_MISSING_INPUT_TENSOR,
        LiteRtStatus_kLiteRtStatusErrorNotFound as ERROR_NOT_FOUND,
        LiteRtStatus_kLiteRtStatusErrorRuntimeFailure as ERROR_RUNTIME_FAILURE,
        LiteRtStatus_kLiteRtStatusErrorSerialization as ERROR_SERIALIZATION,
        LiteRtStatus_kLiteRtStatusErrorTimeoutExpired as ERROR_TIMEOUT_EXPIRED,
        LiteRtStatus_kLiteRtStatusErrorUnknown as ERROR_UNKNOWN,
        LiteRtStatus_kLiteRtStatusErrorUnsupported as ERROR_UNSUPPORTED,
        LiteRtStatus_kLiteRtStatusErrorUnsupportedCompilerVersion as ERROR_UNSUPPORTED_COMPILER_VERSION,
        LiteRtStatus_kLiteRtStatusErrorUnsupportedOpShapeInferer as ERROR_UNSUPPORTED_OP_SHAPE_INFERER,
        LiteRtStatus_kLiteRtStatusErrorUnsupportedRuntimeVersion as ERROR_UNSUPPORTED_RUNTIME_VERSION,
        LiteRtStatus_kLiteRtStatusErrorWrongVersion as ERROR_WRONG_VERSION,
        LiteRtStatus_kLiteRtStatusInvalidTransformation as INVALID_TRANSFORMATION,
        LiteRtStatus_kLiteRtStatusLegalizeNoMatch as LEGALIZE_NO_MATCH,
        LiteRtStatus_kLiteRtStatusOk as OK,
        LiteRtStatus_kLiteRtStatusPatternNoMatch as PATTERN_NO_MATCH,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_symbols_are_consistent() {
        let names = LiteRtFunctions::required_symbols();
        assert_eq!(names.len(), LiteRtFunctions::required_symbol_count());
        assert!(names.iter().all(|n| n.starts_with("LiteRt")));
        // Environment creation is resolved first, so it is the symbol a classic
        // TFLite library reports as missing.
        assert_eq!(names[0], "LiteRtCreateEnvironment");
    }

    #[test]
    fn missing_symbol_reports_name() {
        let missing = MissingSymbol("LiteRtCreateCompiledModel");
        assert_eq!(missing.name(), "LiteRtCreateCompiledModel");
        assert!(missing.to_string().contains("LiteRtCreateCompiledModel"));
    }

    #[test]
    fn status_constants_match_header_values() {
        assert_eq!(status::OK, 0);
        assert_eq!(status::ERROR_INVALID_ARGUMENT, 1);
        assert_eq!(status::ERROR_RUNTIME_FAILURE, 3);
        assert_eq!(status::CANCELLED, 100);
        assert_eq!(status::ERROR_FILE_IO, 500);
        assert_eq!(status::ERROR_COMPILATION, 504);
    }
}
