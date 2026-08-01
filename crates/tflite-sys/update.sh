#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

set -eu

# Portable in-place sed (GNU sed and BSD sed).
sedi() {
  if sed --version >/dev/null 2>&1; then
    sed -i "$@"
  else
    sed -i '' "$@"
  fi
}

# ---------------------------------------------------------------------------
# TensorFlow Lite C API
# ---------------------------------------------------------------------------
BINDGEN_EXTRA_CLANG_ARGS="-I./" bindgen \
  --dynamic-loading tensorflowlite_c \
  --wrap-unsafe-ops \
  --allowlist-function 'TfLite.*' \
  wrapper.h > src/ffi.rs

# libloading 0.9 sealed the path argument of `Library::new` behind the new
# `AsFilename` trait; bindgen still emits the 0.8-era `AsRef<OsStr>` bound.
# Rewrite the generated bound so the loader compiles against libloading 0.9+.
sedi 's/P: AsRef<::std::ffi::OsStr>,/P: ::libloading::AsFilename,/' src/ffi.rs

# Fix C code examples in doc comments to prevent doc-test failures.
# Bindgen copies C API doc comments verbatim. Rustdoc treats 4+-space
# indented blocks as Rust code and tries to compile them. We wrap these
# blocks in ```text fences.

# Opening fences: triple-newline-indent (2 doc comments)
sedi 's/\\n\\n\\n     /\\n\\n```text\\n     /g' src/ffi.rs
# Opening fence: GetExecutionPlan second code block
sedi 's/undefined\.\\n\\n     void/undefined.\\n\\n```text\\n     void/' src/ffi.rs
# Opening fence: PreviewDelegatePartitioning code block
sedi 's/usage:\\n\\n     /usage:\\n\\n```text\\n     /' src/ffi.rs
# Closing fences: indented } followed by non-indented paragraph
sedi 's/\\n     }\\n\\n \([^ ]\)/\\n     }\\n```\\n\\n \1/g' src/ffi.rs
# Closing fence: indented } at end of doc string
sedi 's/\\n     }"/\\n     }\\n```"/' src/ffi.rs

# ---------------------------------------------------------------------------
# LiteRT C API (vendored headers under ./litert/, pin in litert/VERSION)
#
# Run ./litert/vendor.sh first to refresh or re-pin the headers -- it reapplies
# the local patches under litert/patches/.
#
# `--dynamic-loading` is used even though the generated `litert` loader struct
# is never instantiated: src/litert.rs resolves LiteRt* symbols individually
# (so a missing one degrades gracefully and is named in the error), and uses
# this generated table only as a compile-time ABI cross-check against its
# hand-written signatures. Dropping the flag would remove that safety net.
# ---------------------------------------------------------------------------
BINDGEN_EXTRA_CLANG_ARGS="-I./" bindgen \
  --dynamic-loading litert \
  --wrap-unsafe-ops \
  --allowlist-function 'LiteRt.*' \
  --allowlist-type 'LiteRt.*' \
  --allowlist-var 'kLiteRt.*' \
  --allowlist-var 'LITERT_.*' \
  wrapper_litert.h > src/litert_ffi.rs

sedi 's/P: AsRef<::std::ffi::OsStr>,/P: ::libloading::AsFilename,/' src/litert_ffi.rs

cargo clippy --fix --allow-dirty

cat <<'EOF'

Bindings regenerated. If the LiteRT headers moved, `cargo check -p
edgefirst-tflite-sys` will fail inside `litert::abi_cross_check` for any symbol
whose signature changed shape. Review those changes and update the
`litert_functions!` table in src/litert.rs to match -- do not silence them.
EOF
