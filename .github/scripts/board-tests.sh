#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Au-Zone Technologies
#
# Run the edgefirst-tflite integration test binaries on a board against the
# BSP's TFLite runtime and NPU delegate.
#
# Usage: board-tests.sh <payload-dir> <npu-delegate.so>
#
# The payload holds bin/<test binaries> and testdata/. Each binary runs once,
# single-threaded: VxDelegate initialisation is not thread-safe.
#
# A skipped test passes, so exit codes alone could report a board run that
# tested nothing. TFLITE_TEST_NO_SKIP turns the runtime-capability skips (no
# library, no opaque delegate API, no telemetry profiler, no NPU delegate)
# into test failures. The log checks below are a second guard: the
# custom_delegate and edgefirst_delegate binaries must print no skips at all,
# and the integration binary must have loaded the library.

set -euo pipefail

payload="$(cd "${1:?payload dir}" && pwd)"
npu_delegate="${2:?NPU delegate path}"

if [[ -z "${TFLITE_TEST_LIB:-}" ]]; then
    TFLITE_TEST_LIB="$( (ls /usr/lib/libtensorflow-lite.so.2.* 2>/dev/null || true) | sort -V | tail -1)"
fi
[[ -f "$TFLITE_TEST_LIB" ]] || { echo "::error::no TFLite runtime found (TFLITE_TEST_LIB='$TFLITE_TEST_LIB')"; exit 1; }
[[ -f "$npu_delegate" ]] || { echo "::error::NPU delegate not found: $npu_delegate"; exit 1; }
export TFLITE_TEST_LIB TFLITE_TEST_NPU_DELEGATE="$npu_delegate" TFLITE_TEST_NO_SKIP=1
echo "TFLite runtime: $TFLITE_TEST_LIB"
echo "NPU delegate:   $TFLITE_TEST_NPU_DELEGATE"

# integration.rs opens ../../testdata relative to the working directory.
workdir="$payload/run/cwd"
mkdir -p "$workdir"
cd "$workdir"

status=0

# run <binary> <minimum passed> <forbidden output pattern>
run() {
    local bin="$payload/bin/$1" min="$2" forbid="$3" log="$payload/run/$1.log"
    echo "::group::$1"
    chmod +x "$bin"
    local rc=0
    "$bin" --test-threads=1 --nocapture 2>&1 | tee "$log" || rc=$?
    echo "::endgroup::"

    local passed
    passed="$(sed -n 's/^test result: [a-zA-Z]*\. \([0-9]*\) passed.*/\1/p' "$log" | tail -1)"
    if (( rc != 0 )); then
        echo "::error::$1 failed (exit $rc)"
        status=1
    elif (( ${passed:-0} < min )); then
        echo "::error::$1 passed ${passed:-0} tests, expected at least $min"
        status=1
    elif grep -q -E "$forbid" "$log"; then
        echo "::error::$1 skipped tests that must run on this board:"
        grep -E "$forbid" "$log" | sed 's/^/  /'
        status=1
    else
        echo "$1: $passed passed"
    fi
}

run custom_delegate 11 'SKIPPED'
run edgefirst_delegate 8 'SKIPPED'
run integration 40 'SKIPPED: TFLite shared library not available'

exit "$status"
