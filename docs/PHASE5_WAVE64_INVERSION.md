# Phase 5: Wave64 inversion study

Date: 2026-09-28

Targets: MI300X (`gfx942`) and MI355X (`gfx950`)

## Outcome

Keep the existing per-lane fixed addition-chain inversion. The tested wave64
Montgomery-batch candidate is correct, but on MI355X it delivers only 39.0% of
the per-lane inversion rate and is 2.57 times slower. It also spills eight
VGPRs. The candidate is retained as an isolated experiment and is not wired
into `KernelA`.

This is a useful negative result: reducing the mathematical inversion count
from 64 per wave to one does not reduce the wave's issued vector instruction
count on CDNA. A fixed, branchless inversion executed by lane 0 still occupies
the wave's vector pipeline, while the other lanes are masked off. Prefix and
suffix scans then add work and register pressure.

## Candidates

### Per-lane baseline

Every lane independently evaluates the selected Phase 4 `p - 2` addition
chain with the Comba-MAD multiplication backend. `KernelA` already performs a
Montgomery batch over its 24 points per lane, so this means one field inversion
per lane for each 24-point update.

### Wave64 cooperative batch

`InvertWave64` treats zero and inactive lanes as the multiplicative identity,
computes inclusive prefix and suffix products using wave shuffles, inverts the
wave product in lane 0, broadcasts it, and reconstructs each active non-zero
lane's inverse. Active zero inputs return zero, and inactive lanes remain
unchanged.

The implementation deliberately supports active widths from 1 through 64 so
its semantics are explicit for partial waves. The current production geometry
uses complete waves.

### Persistent inversion workers

A producer/worker queue was not implemented after the cooperative primitive
failed. Both proposed topologies rely on moving fixed-chain inversion work to a
subset of waves. On a SIMD machine that does not reduce the number of issued
vector instructions for each inversion, and a persistent queue would add
atomics, cache traffic, synchronization, residency constraints, and progress
rules. It cannot recover the measured 2.57x deficit of the queue-free
cooperative primitive.

This decision should be revisited only if the inversion algorithm becomes
lane-divergent or variable-time, or if a future target provides an execution
mechanism that makes a single-lane inversion materially cheaper than a full
wave instruction stream.

## Correctness

The GPU test covers 2,048 deterministic field elements, injected zero values,
and active wave widths `1, 2, 3, 7, 16, 31, 32, 33, 47, 63, 64`, followed by
full waves. Results are checked independently with Boost.Multiprecision;
inactive values must remain unchanged.

```sh
cmake --preset mi355x
cmake --build --preset mi355x --target rckangaroo_gpu_wave64_inverse_tests -j
build/mi355x/tests/rckangaroo_gpu_wave64_inverse_tests
```

Result on MI355X:

```text
Wave64 inversion passed 2048 values across full and partial waves.
```

The same test target cross-compiles for `gfx942`. Physical MI300X correctness
execution is still required.

## MI355X benchmark

The benchmark uses 256 workgroups, 256 threads per workgroup, eight inversions
per timed launch, one warm-up launch, and five HIP-event samples. Each variant
processes 524,288 inversions per sample. State reset is outside the timed
region.

```sh
python3 scripts/benchmark/run_wave64_inversion.py \
  --preset mi355x --device 0 --samples 5 --iterations 8
```

| Variant | Median time | Median rate | Rate MAD | Relative rate |
|---|---:|---:|---:|---:|
| Per-lane addition chain | 1.901947 ms | 0.275659 Ginversion/s | 0.003470 | 1.000x |
| Wave64 batch | 4.880069 ms | 0.107435 Ginversion/s | 0.000269 | 0.390x |

The wave64 candidate is 2.566x slower by median elapsed time. Since it loses
decisively in isolation, integrating it into `KernelA` cannot improve
end-to-end performance and was intentionally skipped.

## Resource and ISA evidence

Compiler metadata is the same for `gfx942` and `gfx950`:

| Variant | VGPR | SGPR | Private bytes | VGPR spills |
|---|---:|---:|---:|---:|
| Per-lane addition chain | 126 | 46 | 0 | 0 |
| Wave64 batch | 128 | 76 | 36 | 8 |

Static disassembly counts:

| Target | Variant | Total | VALU | SALU | MAD | Wave shuffles | Scratch |
|---|---|---:|---:|---:|---:|---:|---:|
| `gfx950` | Per-lane | 54,885 | 46,792 | 8,089 | 5,455 | 0 | 0 |
| `gfx950` | Wave64 | 64,005 | 53,718 | 10,145 | 6,533 | 128 | 10 |
| `gfx942` | Per-lane | 54,885 | 46,792 | 8,089 | 5,455 | 0 | 0 |
| `gfx942` | Wave64 | 64,007 | 53,718 | 10,147 | 6,533 | 128 | 10 |

The wave candidate adds 14 field-multiplication sites to the wave instruction
stream: 12 for its two scans and two for reconstruction. It also adds 128
`ds_bpermute_b32` instructions in the unrolled code,
and scratch accesses caused by spills. The single lane-0 addition chain remains
present as a wave instruction stream, explaining why the nominal 64-to-1
reduction in scalar inversions does not translate to SIMD throughput.

## Reproducing artifacts

The benchmark runner records Git/toolchain/GPU metadata, raw output, parsed
results, compiler resource metadata, instruction counts, and disassembly under
an ignored `profiles/` directory. ISA-only extraction is also available:

```sh
python3 scripts/profile/extract_field_variants.py \
  --preset mi355x --suite wave64
python3 scripts/profile/extract_field_variants.py \
  --preset mi300x --suite wave64
```

MI300X performance remains unmeasured; matching static resources do not imply
matching runtime behavior. The selected per-lane architecture remains
provisional for `gfx942` until the correctness and benchmark commands run on
physical MI300X hardware.
