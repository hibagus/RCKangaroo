# Performance results

This file records reproducible performance checkpoints. Phase 3 will expand
the metadata collection, profiling artifacts, and multi-GPU methodology.

## Portable HIP baseline: MI355X

- Date: 2026-09-28
- Revision: `319cd94`
- ROCm: 7.2.0
- GPU: AMD Instinct MI355X, `gfx950`, 256 CUs
- Backend: portable HIP 8x32-bit field arithmetic

Build and run:

```sh
cmake --preset mi355x
cmake --build --preset mi355x --target rckangaroo_hip_benchmark -j
build/mi355x/benchmarks/rckangaroo_hip_benchmark 0
```

Result:

```text
GPU 0: AMD Instinct MI355X (gfx950:sramecc+:xnack-), 256 CUs
KernelA portable baseline: 13.807 ms, 113.918 MKeys/s (1572864 kangaroos, one step)
```

The value is the median of five HIP-event samples after one warm-up. Each
sample launches one workgroup per CU, 256 threads per workgroup, and 24
kangaroos per thread. Every kangaroo starts at `G` and uses a table containing
`2G`; host-to-device resets are outside the timed region.

This is a focused `KernelA` baseline, not an end-to-end solver throughput
claim. It excludes `KernelGen`, distance/loop processing in Kernels B/C, host
distinguished-point handling, and transfers. Those measurements belong to
Phase 3. No MI300X performance number has been measured yet.

## Phase 3 repeatable MI355X baseline

- Date: 2026-09-28
- Base revision: `0d8f0e4` plus the Phase 3 working tree (the artifacts record
  the dirty-file list)
- ROCm: 7.2.0
- GPU: AMD Instinct MI355X, `gfx950`, 256 CUs
- Workload: range 139, DP 32, seed 1, one GPU
- Geometry: 256 workgroups x 256 threads x 24 points, 1,572,864 kangaroos

Focused KernelA protocol: 2,000 ms warm-up followed by five 2,000 ms
fixed-kernel-time samples. State reset and host-to-device copies are outside the
HIP-event interval.

```text
KernelA: 6.420 ms/step, 244.983 MKeys/s median, 0.226 MKeys/s MAD
Samples: 244.757, 244.896, 244.983, 245.406, 245.331 MKeys/s
```

A telemetry validation run at the same warmed throughput observed a
representative GFX clock of 2,389-2,390 MHz, 442 W socket power, and 60-61 C
hotspot temperature while KernelA was active. Clocks were managed rather than
locked; future comparisons must retain and inspect their telemetry series.

The earlier Phase 2 number used a much shorter, cold single-launch protocol and
must not be compared directly with this warmed fixed-duration result.

After one 60-second solver warm-up, three independent 60-second solver samples
produced these process-level medians:

| Metric | Median | MAD | Min / max |
|---|---:|---:|---:|
| KernelGen | 36,896.801 ms | 4.207 ms | 36,890.160 / 36,901.008 ms |
| KernelA, 1,000 steps | 7,802.150 ms | 7.689 ms | 7,794.461 / 7,813.161 ms |
| KernelB | 9.044 ms | 0.002 ms | 9.033 / 9.046 ms |
| KernelC | 6.113 ms | 0.013 ms | 6.100 / 6.127 ms |
| End-to-end throughput | 201.211 MKeys/s | 0.180 MKeys/s | 200.928 / 201.391 MKeys/s |

One separate fixed 60-second rocprofv3 workload completed three steady-state
iterations after KernelGen and attributed traced GPU time as follows:

| Region | Median HIP event time | MAD | Share of traced GPU time |
|---|---:|---:|---:|
| KernelGen | 36,892.070 ms (one call) | n/a | 61.13% |
| KernelA, 1,000 steps | 7,805.894 ms | 1.122 ms | 38.80% |
| KernelB | 9.027 ms | 0.003 ms | 0.045% |
| KernelC | 6.095 ms | 0.002 ms | 0.030% |

The profiler run's end-to-end median was 201.108 MKeys/s with a 0.051 MKeys/s
within-process MAD, consistent with the unprofiled repeatability run. End-to-end
timing includes normal DP processing, copies, loop handling, and host-side
iteration work. It excludes the one-time KernelGen cost from the per-iteration
throughput calculation.

Compiler code-object metadata for the same `gfx950` build:

| Kernel | VGPR | SGPR | AGPR | Dynamic LDS | Private bytes | Spills |
|---|---:|---:|---:|---:|---:|---:|
| KernelGen | 182 | 86 | 0 | 0 KiB | 48 | 0 |
| KernelA | 144 | 84 | 0 | 32 KiB | 0 | 0 |
| KernelB | 87 | 63 | 0 | 48 KiB | 0 | 0 |
| KernelC | 130 | 74 | 0 | 48 KiB | 0 | 0 |

rocprofv3 observed a 24 MiB device scratch allocation associated with the code
object. KernelGen's 36.9-second startup and 48-byte private segment, followed by
KernelA's steady-state cost and register pressure, are the first Phase 4
optimization priorities. See [PROFILING.md](PROFILING.md) for the artifact and
comparison workflow. No MI300X runtime result has been measured.

## Phase 4 optimized field arithmetic

The selected Comba-MAD and fixed addition-chain implementation increased the
warmed MI355X KernelA result from 244.983 to 3,361.798 MKeys/s. A shorter
profiled solver diagnostic reached 3,836.254 MKeys/s and reduced KernelGen from
about 36.9 seconds to 1.50 seconds. See
[PHASE4_FIELD_ARITHMETIC.md](PHASE4_FIELD_ARITHMETIC.md) for the candidate
matrix, protocol, resource tables, correctness gates, and MI300X limitation.
