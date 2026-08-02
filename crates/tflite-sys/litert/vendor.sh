#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.
#
# Re-vendor the LiteRT C API headers, reapplying local patches.
#
#   ./vendor.sh            # re-vendor the tag in VERSION
#   ./vendor.sh v2.2.0     # re-vendor a different tag and update VERSION
#
# Afterwards run ../update.sh to regenerate src/litert_ffi.rs. The compile-time
# ABI cross-check in src/litert.rs will fail the build if any signature that
# LiteRtFunctions binds changed shape in the new headers -- that failure is the
# signal to review the diff, not something to silence.

set -euo pipefail

cd "$(dirname "$0")"

TAG="${1:-$(cat VERSION)}"
URL="https://github.com/google-ai-edge/LiteRT/archive/refs/tags/${TAG}.tar.gz"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Fetching LiteRT ${TAG} ..."
curl -fsSL "$URL" | tar -xz -C "$WORK" --strip-components=1

if [ ! -d "$WORK/litert/c" ]; then
  echo "error: ${TAG} tarball has no litert/c directory" >&2
  exit 1
fi

echo "Replacing vendored headers ..."
rm -rf c
cp -R "$WORK/litert/c" c
# Upstream ships no LICENSE inside litert/, so take the repository one.
cp "$WORK/LICENSE" LICENSE

echo "Applying local patches ..."
for patch in patches/*.patch; do
  [ -e "$patch" ] || continue
  echo "  $patch"
  patch -p1 --forward < "$patch"
done

echo "$TAG" > VERSION

cat <<EOF

Vendored LiteRT ${TAG}.

Note: build_common/build_config.h is a hand-written stub (upstream generates it
via CMake) and is intentionally not overwritten by this script.

Next: cd .. && ./update.sh
EOF
