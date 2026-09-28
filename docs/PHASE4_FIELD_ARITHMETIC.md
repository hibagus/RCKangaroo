# Phase 4 field-arithmetic report

- Date: 2026-09-28
- ROCm: 7.2.0
- Runtime-tested GPU: AMD Instinct MI355X (`gfx950`)
- Compile/ISA-tested GPU: AMD Instinct MI300X (`gfx942`)

## Outcome

The production HIP field API now uses:

- one AMDGCN inline-assembly carry block for 256-bit addition;
- compiler-generated 8x32-bit Comba multiplication and squaring;
- the existing reduction specialized for
  `p = 2^256 - 2^32 - 977`; and
- a fixed `p - 2` inversion chain with 255 squarings and 15 multiplies.

LLVM selects `V_MAD_U64_U32` for the winning Comba idiom. A separate variant
forces `V_MUL_LO_U32`, `V_MUL_HI_U32`, and a dependent VCC carry chain in one
assembly block. The explicit version is correct and uses fewer VGPRs, but it is
slower on MI355X, so it remains a comparison/fallback path rather than the
production selection.

The code is separated by responsibility:

- `src/hip/field/portable.hpp`: portable correctness baseline and reduction;
- `src/hip/field/variants.hpp`: isolated compiler and assembly candidates; and
- `src/hip/field/api.hpp`: backend selected by the production kernels.

## Correctness gates

`rckangaroo_gpu_field_variant_tests` compares GPU results with an independent
Boost.Multiprecision oracle using:

- 4,096 deterministic edge/random vectors for carry-chain addition,
  multiplication, and squaring; and
- 256 deterministic edge/random vectors for every addition-chain inversion
  backend, including zero.

The original portable field suite remains in place. The integrated MI355X build
passed all six normal CPU/GPU tests and both opt-in known-key solver tests:

```text
100% tests passed, 0 tests failed out of 6
rckangaroo.gpu.end_to_end.private_2        Passed
rckangaroo.gpu.end_to_end.offset_private_3 Passed
```

## Isolated MI355X result

Build and run:

```sh
cmake --preset mi355x
cmake --build --preset mi355x --target rckangaroo_field_benchmark -j
build/mi355x/benchmarks/rckangaroo_field_benchmark --samples 5
```

To archive the result, metadata, full ISA, and resource summaries together:

```sh
python3 scripts/benchmark/run_field_variants.py --preset mi355x --device 0
```

The geometry was 256 workgroups x 256 threads. Multiply/square kernels performed
256 serial operations per lane. Inversion kernels performed one inversion per
lane. Values below are medians of five HIP-event samples after one warm-up.

| Operation | Portable baseline | Comba MAD / chain | Explicit MUL/carry | Winning speedup |
|---|---:|---:|---:|---:|
| Multiply | 5.630 Gop/s | 73.583 Gop/s | 59.544 Gop/s | 13.07x |
| Square | 5.448 Gop/s | 76.412 Gop/s | 60.262 Gop/s | 14.03x |
| Inverse | 0.010800 Gop/s | 0.265884 Gop/s | 0.214756 Gop/s | 24.62x |

The portable-multiply addition chain reached 0.019751 Gop/s but used an
84-byte private segment and 26 reported VGPR spills. It was rejected. The
selected Comba-MAD chain used 126 VGPRs with no private segment or reported
spills.

Standalone `gfx950` resources for the selected kernels:

| Kernel | VGPR | SGPR | Private bytes | Spills |
|---|---:|---:|---:|---:|
| `FieldMulCombaMad` | 56 | 36 | 0 | 0 |
| `FieldSquareCombaMad` | 42 | 36 | 0 | 0 |
| `FieldInverseChainCombaMad` | 126 | 46 | 0 | 0 |

The standalone multiply contains 77 `V_MAD_U64_U32` instructions, including
reduction work. The explicit candidate contains 64 each of `V_MUL_LO_U32` and
`V_MUL_HI_U32`, plus 193 `V_ADD_CO_U32`/`V_ADDC_CO_U32` instructions.

## Integrated MI355X result

The same warmed KernelA protocol as Phase 3 used a 2,000 ms warm-up followed by
five 2,000 ms samples:

| Metric | Phase 3 | Phase 4 | Change |
|---|---:|---:|---:|
| KernelA | 6.420 ms/step | 0.468 ms/step | 13.72x faster |
| KernelA throughput | 244.983 MKeys/s | 3,361.798 MKeys/s | 13.72x |

Phase 4 samples were 3,364.103, 3,360.607, 3,363.549, 3,360.005, and
3,361.798 MKeys/s, with a 1.751 MKeys/s MAD.

A separate 20-second diagnostic solver run reported:

| Metric | Phase 3 checkpoint | Phase 4 diagnostic |
|---|---:|---:|
| KernelGen | 36,896.801 ms | 1,503.317 ms |
| KernelA, 1,000 steps | 7,802.150 ms | 400.308 ms |
| KernelB | 9.044 ms | 8.997 ms |
| KernelC | 6.113 ms | 0.291 ms |
| End-to-end throughput | 201.211 MKeys/s | 3,836.254 MKeys/s |

The end-to-end comparison is directional rather than the final controlled
baseline because the Phase 4 diagnostic used a shorter run and did not capture
the full telemetry protocol. The focused KernelA comparison uses the matching
fixed-duration protocol.

Integrated hot-kernel metadata remains spill-free on both target builds:

| Kernel | Phase 3 VGPR/SGPR | Phase 4 VGPR/SGPR | Private bytes | Spills |
|---|---:|---:|---:|---:|
| KernelGen | 182 / 86 | 226 / 70 | 48 | 0 |
| KernelA | 144 / 84 | 142 / 80 | 0 | 0 |
| KernelB | 87 / 63 | 87 / 63 | 0 | 0 |
| KernelC | 130 / 74 | 176 / 66 | 0 | 0 |

KernelGen and KernelC register demand increased, but the end-to-end diagnostic
still improved substantially. Their occupancy and code-size tradeoffs should be
revisited during Phase 6 tuning.

## ISA artifacts and MI300X gate

Generate standalone field-kernel artifacts with:

```sh
python3 scripts/profile/extract_field_variants.py --preset mi355x
python3 scripts/profile/extract_field_variants.py --preset mi300x
```

Generate integrated hot-kernel artifacts with:

```sh
python3 scripts/profile/extract_code_object.py --preset mi355x
python3 scripts/profile/extract_code_object.py --preset mi300x
```

The current `gfx942` and `gfx950` field objects have matching instruction and
resource summaries, and both integrated objects report zero SGPR/VGPR spills.
Artifacts are stored under ignored `profiles/` directories because full
disassemblies and code objects are large.

The MI300X selection is provisional. It must pass the GPU differential tests,
known-key solver tests, isolated benchmark, warmed KernelA benchmark, and a
profiled solver run on physical `gfx942` hardware before the Phase 4 checkbox
can be closed for both target architectures.
