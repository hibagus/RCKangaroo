#!/usr/bin/env bash
# Differential test for the CDNA field arithmetic.
#
# Ground truth is Python's arbitrary-precision integers, so the reference cannot
# share a bug with the code under test. Vectors lead with the full cross product
# of edge cases (0, 1, p-1, p, 2^256-1, values near 2^256 that stress the
# Solinas fold, and low-word patterns that stress the divsteps inverse), then
# bulk random.
#
# Usage: tests/run_diff.sh [num_vectors] [device] [gfx_arch]

set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-10000000}"
DEV="${2:-0}"
ARCH="${3:-gfx942}"
ROCM="${ROCM_PATH:-/opt/rocm/core-10}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> building for $ARCH"
"$ROCM/bin/hipcc" -O3 --offload-arch="$ARCH" -fno-strict-aliasing \
    -o "$TMP/modarith_diff" tests/modarith_diff.hip

echo "==> generating $N vectors"
python3 tests/gen_vectors.py "$TMP/vec.bin" "$N"

echo "==> running on device $DEV"
"$TMP/modarith_diff" "$TMP/vec.bin" "$TMP/out.bin" "$DEV"

echo "==> verifying against exact arithmetic"
python3 tests/check_results.py "$TMP/vec.bin" "$TMP/out.bin"
