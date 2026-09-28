#!/usr/bin/env bash
# Static codegen check for the CDNA field arithmetic.
#
# Asserts the multiply still lowers to the theoretical minimum 64 v_mad_u64_u32
# and that nothing spills. Cheap, and it catches codegen regressions that a
# throughput benchmark would otherwise blame on the algorithm.
#
# Usage: tests/isa_quality.sh [gfx_arch]

set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${1:-gfx942}"
ROCM="${ROCM_PATH:-/opt/rocm/core-10}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/probe.hip" <<'PROBE'
#include "RCGpuUtils_cdna.h"
extern "C" __global__ void k_mul512(u64* o, const u64* i){ u64 a[4],b[4],r[8]; int t=threadIdx.x+blockIdx.x*256;
  for(int k=0;k<4;k++){a[k]=i[t*8+k];b[k]=i[t*8+4+k];} mul_256_to_512(r,a,b); for(int k=0;k<8;k++) o[t*8+k]=r[k]; }
extern "C" __global__ void k_mul(u64* o, const u64* i){ u64 a[4],b[4],r[4]; int t=threadIdx.x+blockIdx.x*256;
  for(int k=0;k<4;k++){a[k]=i[t*8+k];b[k]=i[t*8+4+k];} MulModP(r,a,b); for(int k=0;k<4;k++) o[t*4+k]=r[k]; }
extern "C" __global__ void k_sqr(u64* o, const u64* i){ u64 a[4],r[4]; int t=threadIdx.x+blockIdx.x*256;
  for(int k=0;k<4;k++)a[k]=i[t*4+k]; SqrModP(r,a); for(int k=0;k<4;k++) o[t*4+k]=r[k]; }
extern "C" __global__ void k_sub(u64* o, const u64* i){ u64 a[4],b[4],r[4]; int t=threadIdx.x+blockIdx.x*256;
  for(int k=0;k<4;k++){a[k]=i[t*8+k];b[k]=i[t*8+4+k];} SubModP(r,a,b); for(int k=0;k<4;k++) o[t*4+k]=r[k]; }
// InvModP is probed under KernelA's real launch configuration - one 256-thread
// workgroup per CU, i.e. 1 wave/SIMD - because that is what gives the wave the
// full 256 architectural VGPRs. Compiled for default (higher) occupancy it
// spills 22 VGPRs to scratch; at 1 wave/SIMD it needs 139 and spills nothing.
extern "C" __global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_waves_per_eu(1,1), amdgpu_flat_work_group_size(256,256)))
void k_inv(u64* o, const u64* i){ __align__(16) u64 t[5]; int x=threadIdx.x+blockIdx.x*256;
  for(int k=0;k<4;k++)t[k]=i[x*4+k]; t[4]=0; InvModP((u32*)t); for(int k=0;k<4;k++) o[x*4+k]=t[k]; }
PROBE

"$ROCM/bin/hipcc" -O3 --offload-arch="$ARCH" -fno-strict-aliasing -I include -I include/cdna \
    --offload-device-only -S -o "$TMP/probe.s" "$TMP/probe.hip"

printf "%-12s %6s %8s %10s %7s\n" kernel VALU mad_u64 carry_ops VGPRs
for k in k_mul512 k_mul k_sqr k_sub k_inv; do
  seg=$(awk "/^${k}:/{f=1} f&&/s_endpgm/{print;exit} f" "$TMP/probe.s")
  valu=$(echo "$seg" | grep -cE '^[[:space:]]+v_' || true)
  mad=$(echo "$seg" | grep -cE '^[[:space:]]+v_mad_u64_u32' || true)
  carry=$(echo "$seg" | grep -cE '^[[:space:]]+v_(add_co|addc_co|sub_co|subb_co|subbrev_co)' || true)
  vg=$(grep -A8 "\.name: *${k}\$" "$TMP/probe.s" | grep -m1 'vgpr_count' | awk '{print $2}')
  printf "%-12s %6s %8s %10s %7s\n" "$k" "$valu" "$mad" "$carry" "${vg:-?}"
done
echo

rc=0

# The 8x8 32-bit-limb schoolbook needs exactly 64 MADs. More means the backend
# stopped fusing the accumulate; fewer means it eliminated real work.
mads=$(awk '/^k_mul512:/{f=1} f&&/s_endpgm/{print;exit} f' "$TMP/probe.s" | grep -cE '^[[:space:]]+v_mad_u64_u32')
if [ "$mads" -ne 64 ]; then
  echo "FAIL: mul_256_to_512 emitted $mads v_mad_u64_u32, expected exactly 64"
  rc=1
else
  echo "OK: mul_256_to_512 at the theoretical minimum of 64 v_mad_u64_u32"
fi

if grep -qE 'spill_count: *[1-9]' "$TMP/probe.s"; then
  echo "FAIL: register spills detected"
  grep -E 'spill_count: *[1-9]' "$TMP/probe.s"
  rc=1
else
  echo "OK: no register spills"
fi

exit $rc
