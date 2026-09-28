# Profiling and benchmark workflow

The Phase 3 tools produce self-contained, machine-readable artifacts under
`profiles/`. That directory is intentionally ignored by Git because traces and
disassemblies can be large. Copy or archive a result directory when it must be
retained outside the workstation.

## Reproducible baseline

Configure and build the target before the first run:

```sh
cmake --preset mi355x
cmake --build --preset mi355x --target rckangaroo rckangaroo_hip_benchmark -j
python3 scripts/benchmark/run_baseline.py --preset mi355x --device 0
```

The baseline runner records:

- the Git revision, dirty state, host, ROCm, HIP compiler, and profiler versions;
- static GPU identity plus clock, power, temperature, and memory readings before,
  during, and after the workload;
- the complete range, DP, seed, public-key workload, launch geometry, and timing
  protocol;
- five fixed-kernel-time KernelA samples after warm-up; and
- three independent fixed-duration solver samples with KernelGen/A/B/C HIP-event
  timings and end-to-end throughput.

Each timed command has a sibling `*_telemetry.json` file containing timestamped
AMD SMI samples. This makes loaded clocks and power visible instead of relying
only on idle readings before and after a run.

The default range-139 public-key workload is copied from `launch_kangaroo` and
uses seed 1. The Phase 3 portable MI355X baseline took about 37 seconds in
KernelGen before the steady-state loop. Phase 4 reduced that to about 1.5
seconds, but the default 60-second warm-up and 90-second samples are retained
for controlled comparisons. The runner rejects
a sample that does not complete a KernelA/B/C iteration. A shorter diagnostic
run can omit the solver portion:

```sh
python3 scripts/benchmark/run_baseline.py \
  --preset mi355x --solver-samples 0 --warmup-ms 100 --sample-ms 100 --samples 3
```

The focused benchmark resets state outside the timed region and times KernelA
only. Normal solver runs incur no event-timing overhead unless `RCK_PROFILE` is
set to a nonzero value. In profile mode, the executable emits one
`RCK_PROFILE_JSON=...` record when the GPU worker stops.

## rocprofv3 trace

Capture dispatch, copy, and scratch-allocation traces with:

```sh
python3 scripts/profile/run_rocprof.py --preset mi355x --device 0
```

The wrapper verifies that KernelGen, KernelA, KernelB, and KernelC all appear in
the trace. `hot_kernel_summary.json` is calculated from raw dispatch timestamps
and contains median, MAD, min/max, population standard deviation, launch
geometry, and profiler-reported allocated VGPR/SGPR/scratch counts. Raw CSVs are
retained beside the summary.

## Code objects and generated ISA

Archive the bundled AMD code object, metadata, disassembly, and static
instruction mix with:

```sh
python3 scripts/profile/extract_code_object.py --preset mi355x
python3 scripts/profile/extract_code_object.py --preset mi300x
```

Phase 4 field candidates have their own kernel-per-variant extractor:

```sh
python3 scripts/profile/extract_field_variants.py --preset mi355x
python3 scripts/profile/extract_field_variants.py --preset mi300x
```

It additionally counts the selected MAD, MUL, and VCC carry instructions.

Phase 5's per-lane versus wave64 inversion comparison uses the same extractor
with a suite selector. The benchmark wrapper records metadata and invokes the
extractor automatically:

```sh
python3 scripts/benchmark/run_wave64_inversion.py --preset mi355x --device 0
python3 scripts/profile/extract_field_variants.py --preset mi300x --suite wave64
```

The wave64 suite additionally counts `ds_bpermute_b32` wave shuffles. See
[PHASE5_WAVE64_INVERSION.md](PHASE5_WAVE64_INVERSION.md) for the result and
selection decision.

`resources.json` reports compiler metadata for every hot kernel, including
VGPR, SGPR, AGPR, spills, private memory, static/dynamic LDS, wave size, and
maximum workgroup size. `instruction_mix.json` classifies the generated ISA as
VALU, SALU, VMEM, LDS, atomic, or other. The full disassembly remains the source
of truth when an instruction-level optimization is evaluated.

Compiler metadata counts and profiler allocation-granularity counts are both
kept deliberately. They answer different questions and should not be silently
substituted for each other.

## Hardware-counter profiling

The repository includes a preflight wrapper for ROCm Compute Profiler:

```sh
python3 scripts/profile/run_rocprof_compute.py --preset mi355x --print-command
python3 scripts/profile/run_rocprof_compute.py --preset mi355x --kernel KernelA
```

The installed ROCm 7.2 profiler on the current development host is missing its
Python runtime dependencies, beginning with `pandas`; the complete pinned list
is `/opt/rocm/libexec/rocprofiler-compute/requirements.txt`. The wrapper exits
with status 2 and prints the missing packages instead of beginning a partial
profile. Install those dependencies in an isolated environment before
collecting occupancy, active-cycle, cache/HBM, atomic, and stall counters.
Counter collection replays the workload many times, so start with a kernel
filter and keep the 37-second KernelGen cost in mind.

## Before/after comparisons

Compare two baseline, rocprof, or ISA artifact directories with:

```sh
python3 scripts/profile/compare_profiles.py \
  profiles/before profiles/after --output profiles/comparison.md
```

The generated Markdown includes every numeric field common to the two artifact
sets. Interpret direction by metric: higher throughput is better, while lower
kernel time, register use, scratch, and traffic are generally better. Never
compare runs with different workload metadata, launch geometry, clocks, or
power policy.

MI300X (`gfx942`) code objects can be extracted on this host, but runtime
performance and telemetry must be captured on real MI300X hardware before an
MI300X result is accepted.
