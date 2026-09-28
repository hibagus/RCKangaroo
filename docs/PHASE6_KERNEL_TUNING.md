# Phase 6: per-architecture kernel tuning

Status: MI355X (gfx950) tuning complete; MI300X (gfx942) defaults are
compile/ISA-validated and await physical-hardware measurement.

Date: 2026-09-28
ROCm: 7.2.0
Measured GPU: AMD Instinct MI355X, gfx950, 256 CUs

## Selected production defaults

| Target | Threads | Blocks/CU | Points/lane | Steps/launch | KernelA table | State layout |
|---|---:|---:|---:|---:|---|---|
| MI300X / gfx942 | 256 | 1 | 24 | 1,000 | split, 32 KiB LDS | group-major |
| MI355X / gfx950 | 256 | 1 | 32 | 2,048 | both point tables, 64 KiB LDS | workgroup-major |

The gfx950 row is selected from measurements on physical hardware. The gfx942
row deliberately retains the conservative Phase 5 geometry and stays within
the 64 KiB CDNA3 LDS limit. It is provisional until the same runners execute on
an MI300X.

The defaults live in the architecture tuning table at
include/rckangaroo/gpu/tuning.hpp. The solver selects them from the runtime
architecture name. The fat binary also compiles architecture-specific table
and state-layout branches into its gfx942 and gfx950 code objects.

Runtime overrides remain available:

~~~sh
build/mi355x/bin/rckangaroo --point-groups 24 --kernel-steps 1000 ...
~~~

## Implementation changes

Phase 6 removed the remaining hidden geometry assumptions from the HIP path:

- workgroup size, point-group count, steps per launch, and blocks per CU are
  CMake cache parameters;
- KernelA, KernelB, KernelC, and KernelGen use runtime group counts and the
  configured workgroup size instead of embedded 256/24 constants;
- KernelB processes complete ten-step chunks plus a tail, so launch lengths no
  longer need to be divisible by MD_LEN;
- jump-list, point-state, last-point, and temporary allocations derive their
  sizes from the selected geometry and launch length;
- point indexing is shared across all kernels and supports group-major and
  workgroup-major layouts;
- KernelA supports split LDS/constant, global-read-only, and 64 KiB LDS point
  table modes;
- benchmark JSON and solver profile JSON record the effective geometry,
  table/LDS selection, and state layout.

The public tuning header is in include/rckangaroo/gpu rather than the private
HIP source tree so applications, benchmarks, and tests all observe the same
architecture defaults.

## Correctness gates

Every compile-time candidate build ran the KernelGen and one-step KernelA
CPU/GPU comparison before its performance result was accepted. The test covers
8, 12, 16, 24, and 32 point groups. The selected gfx950 production build also
passed all nine CTest cases, including randomized field tests, the wave64
inversion test, the multi-geometry kernel test, and both known-key end-to-end
solver tests.

A separate end-to-end run used 8 groups and 512 steps to exercise KernelB's
non-multiple-of-ten tail path and recovered private key 2.

## Geometry matrix

The first sweep used a 500 ms warm-up and three 500 ms fixed-kernel-time
samples per candidate. The runner paired workgroup size with enough workgroups
to retain 256 resident workgroups: 64x4, 128x2, and 256x1.

| Threads / blocks per CU | 8 groups | 12 groups | 16 groups | 24 groups | 32 groups |
|---|---:|---:|---:|---:|---:|
| 64 / 4 | 1,626.725 | 2,188.912 | 2,623.950 | 3,326.750 | 3,706.992 |
| 128 / 2 | 1,198.744 | 1,617.305 | 1,967.658 | 2,529.934 | 2,928.288 |
| 256 / 1 | 1,642.613 | 2,197.054 | 2,638.441 | 3,354.227 | 3,722.509 |

Values are median MKeys/s. The winner was 256 threads, one workgroup per CU,
and 32 points per lane.

A longer 2,000 ms warm-up plus five 2,000 ms samples confirmed the point-group
choice:

| Groups | Median MKeys/s | MAD | Median ms/launch |
|---:|---:|---:|---:|
| 24 | 3,353.342 | 0.707 | 0.469044 |
| 32 | 3,751.144 | 0.668 | 0.559070 |

Thirty-two groups improve aggregate throughput by 11.86%.

## Steps per solver launch

The launch-length runner used three independent 15-second process samples after
a 10-second warm-up, with 32 point groups. It measures normal end-to-end solver
throughput rather than the isolated one-step KernelA benchmark.

| Steps | End-to-end MKeys/s | KernelA ms | KernelB ms | KernelC ms | Approx. allocation |
|---:|---:|---:|---:|---:|---:|
| 256 | 4,436.950 | 118.020 | 3.067 | 0.294 | 3,300 MiB |
| 512 | 4,436.950 | 235.211 | 6.092 | 0.291 | 4,324 MiB |
| 1,000 | 4,443.119 | 459.444 | 11.931 | 0.276 | 6,276 MiB |
| 2,048 | 4,455.360 | 938.616 | 24.576 | 0.274 | 10,468 MiB |

The 2,048-step result is 0.28% above 1,000 steps and is selected because the
project prioritizes peak performance. It also has the highest memory demand,
so the runtime override is useful when device memory must be shared.

A final paired run after selecting the 64 KiB table and workgroup-major layout
preserved the ordering: 4,453.052 MKeys/s for 2,048 steps versus 4,452.552 for
1,000 steps. The 0.01% difference is effectively a tie at this run length, so
1,000 steps is the practical memory-saving choice; the peak-first production
default remains 2,048.

## KernelA jump-table storage

KernelA needs the X/Y coordinates but not the distance portion of the primary
96-byte jump records. The 64 KiB candidate therefore stages two 32 KiB point
tables in LDS; it is not a 96 KiB allocation.

Initial sweep, 1,000 ms warm-up and five 1,000 ms samples:

| Mode | Dynamic LDS | Median MKeys/s | MAD |
|---|---:|---:|---:|
| split primary-LDS / secondary-constant | 32 KiB | 3,748.600 | 3.338 |
| primary-global / secondary-constant | 0 KiB | 3,678.603 | 0.454 |
| both point tables in LDS | 64 KiB | 3,797.580 | 3.726 |

Long confirmation, 2,000 ms warm-up and five 2,000 ms samples:

| Mode | Median MKeys/s | MAD | Median ms/launch |
|---|---:|---:|---:|
| split | 3,720.992 | 0.261 | 0.563600 |
| 64 KiB LDS | 3,790.978 | 5.113 | 0.553195 |

The 64 KiB mode improves the confirmation median by 1.88% and fits comfortably
within gfx950's LDS capacity. gfx942 retains the 32 KiB split mode pending real
hardware measurement.

## State layout

Both layouts passed the CPU/GPU kernel comparison.

| Protocol | Group-major MKeys/s | Workgroup-major MKeys/s | Selected gain |
|---|---:|---:|---:|
| 1,000 ms warm-up, 5 x 1,000 ms | 3,798.084 | 3,817.584 | 0.51% |
| 5,000 ms warm-up, 15 x 2,000 ms | 3,814.094 | 3,820.874 | 0.18% |

Workgroup-major wins both runs and is selected on gfx950. The gain is small, so
gfx942 stays group-major until it can be measured rather than assuming the same
cache/XCD behavior.

## A/B boundary decision

At the selected 2,048-step launch length, KernelB takes about 24.6 ms versus
938.6 ms for KernelA, roughly 2.6% of their combined time. Full A/B fusion would
need to absorb variable loop detection, distance history, and recovery state
into an already register-heavy KernelA. The recoverable upper bound did not
justify the spill and correctness risk in this phase. The split boundary and
existing cache-streaming jump-list store are retained.

This is an explicit pruning decision, not a claim that fusion can never win.
It should only be revisited with counter evidence showing jump-list traffic is
a dominant bottleneck or with a design that removes the history without
inflating live state.

## Selected code-object resources

| Target | Kernel | VGPR | SGPR | Private bytes | Dynamic LDS | Spills |
|---|---|---:|---:|---:|---:|---:|
| gfx950 | KernelA | 146 | 82 | 0 | 64 KiB | 0 |
| gfx950 | KernelB | 114 | 72 | 0 | 48 KiB | 0 |
| gfx950 | KernelC | 176 | 68 | 0 | 48 KiB | 0 |
| gfx950 | KernelGen | 226 | 70 | 48 | 0 | 0 |
| gfx942 | KernelA | 146 | 86 | 0 | 32 KiB | 0 |
| gfx942 | KernelB | 114 | 73 | 0 | 48 KiB | 0 |
| gfx942 | KernelC | 176 | 68 | 0 | 48 KiB | 0 |
| gfx942 | KernelGen | 226 | 72 | 48 | 0 | 0 |

All integrated hot kernels are spill-free. KernelGen still has a 48-byte
private segment and remains a future optimization target.

## Reproduction

~~~sh
python3 scripts/benchmark/tune_kernel_a_geometry.py \
  --preset mi355x --device 0 --output profiles/phase6-mi355x-geometry

python3 scripts/benchmark/tune_solver_steps.py \
  --preset mi355x --device 0 --output profiles/phase6-mi355x-steps

python3 scripts/benchmark/tune_kernel_a_tables.py \
  --preset mi355x --device 0 --point-groups 32 \
  --output profiles/phase6-mi355x-tables

python3 scripts/benchmark/tune_state_layout.py \
  --preset mi355x --device 0 --point-groups 32 \
  --output profiles/phase6-mi355x-layout

python3 scripts/profile/extract_code_object.py \
  --preset mi355x --output profiles/phase6-mi355x-selected-isa
python3 scripts/profile/extract_code_object.py \
  --preset mi300x --output profiles/phase6-mi300x-selected-isa
~~~

The profiles directory is intentionally ignored by Git. Each runner writes
metadata, command logs, machine-readable JSON, and a Markdown summary.

## Remaining MI300X work

Run the same geometry, step, table, and layout matrices on physical MI300X
hardware, followed by the complete CTest suite and a long end-to-end sample.
Do not promote the current gfx942 values from provisional to measured based only
on matching ISA or spill counts.
