// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.
//
// Bindgen entry point for the LiteRT C API (LiteRT v2.1.6 headers).
// Intentionally excludes:
// - litert_dispatch_delegate.h (depends on TFLite opaque delegate headers)
// - litert_runtime_c_api.h (ABI method table, not dlsym LiteRt* exports)

#include "litert/c/litert_common.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_profiler.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_op_options.h"
#include "litert/c/litert_builder.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/c/options/litert_runtime_options.h"
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/c/options/litert_samsung_options.h"
#include "litert/c/options/litert_webnn_options.h"
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/c/internal/litert_accelerator.h"
