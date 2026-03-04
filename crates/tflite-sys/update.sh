#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

BINDGEN_EXTRA_CLANG_ARGS="-I./" bindgen --dynamic-loading tensorflowlite_c --wrap-unsafe-ops --allowlist-function 'TfLite.*' wrapper.h > src/ffi.rs

# Fix C code examples in doc comments to prevent doc-test failures.
# Bindgen copies C API doc comments verbatim. Rustdoc treats 4+-space
# indented blocks as Rust code and tries to compile them. We wrap these
# blocks in ```text fences.

# Opening fences: triple-newline-indent (2 doc comments)
sed -i 's/\\n\\n\\n     /\\n\\n```text\\n     /g' src/ffi.rs
# Opening fence: GetExecutionPlan second code block
sed -i 's/undefined\.\\n\\n     void/undefined.\\n\\n```text\\n     void/' src/ffi.rs
# Opening fence: PreviewDelegatePartitioning code block
sed -i 's/usage:\\n\\n     /usage:\\n\\n```text\\n     /' src/ffi.rs
# Closing fences: indented } followed by non-indented paragraph
sed -i 's/\\n     }\\n\\n \([^ ]\)/\\n     }\\n```\\n\\n \1/g' src/ffi.rs
# Closing fence: indented } at end of doc string
sed -i 's/\\n     }"/\\n     }\\n```"/' src/ffi.rs

cargo clippy --fix --allow-dirty
