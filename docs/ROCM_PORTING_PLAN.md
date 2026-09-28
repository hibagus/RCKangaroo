# RCKangaroo ROCm Porting and Optimization Plan

Status: Phase 5 inversion study complete on `gfx950`; `gfx942` runtime validation is pending
Last updated: 2026-09-28
Targets: AMD Instinct MI300X (`gfx942`, CDNA3) and MI355X (`gfx950`, CDNA4)

## 1. Objective

Port RCKangaroo from CUDA and NVIDIA-specific assembly to HIP/ROCm while preserving correctness and pursuing the highest practical performance on MI300X and MI355X.

The work will proceed in two stages:

1. Establish a correct, maintainable HIP implementation of the existing generic kernels.
2. Develop and measure architecture-specific optimizations for `gfx942` and `gfx950`.

A direct instruction-for-instruction translation of the NVIDIA turbo kernels is not the goal. Those kernels assume 32-lane warps, NVIDIA cache controls, NVIDIA block scheduling, and CUDA cubins. AMD uses 64-lane wavefronts and has materially different LDS and cache characteristics.

## 2. Current-state findings

- The build is CUDA-only and hardcodes CUDA 12.8.
- Device finite-field arithmetic in `RCGpuUtils.h` uses inline PTX carry chains.
- The turbo path consists of NVIDIA SASS sources and committed `sm_89`/`sm_120` cubins.
- The turbo producer/inversion-worker design assumes 32-lane warps and eight warps per 256-thread block.
- `BLOCK_SIZE=256`, `PNT_GROUP_CNT=24`, and `STEP_CNT=1000` are embedded throughout the data layout and kernels.
- The host requests 98 KiB of dynamic shared memory for Kernel A.
- CUDA persisting-L2 controls have no direct ROCm equivalent.
- Kernel parameter structures use implicit ABI details, including `bool` and manually calculated byte offsets.
- There are no automated CPU/GPU correctness tests or isolated field-operation benchmarks.

The development machine currently provides:

- ROCm 7.2.
- Eight MI355X `gfx950` GPUs.
- A compiler capable of producing both `gfx942` and `gfx950` code.

MI355X can therefore be tuned locally. MI300X can be cross-compiled locally, but final MI300X support requires tests and performance measurements on real `gfx942` hardware.

## 3. Architectural constraints

| Property | MI300X | MI355X |
|---|---:|---:|
| Architecture | CDNA3 | CDNA4 |
| LLVM target | `gfx942` | `gfx950` |
| Active compute units | 304 | 256 |
| Wavefront size | 64 | 64 |
| LDS per compute unit | 64 KiB | 160 KiB |

Consequences:

- A 96/98 KiB LDS workgroup cannot run on MI300X.
- MI355X can hold a 96 KiB table in LDS, but doing so will probably restrict the kernel to one workgroup per CU.
- All warp-level algorithms must be redesigned for wave64.
- `gfx942` and `gfx950` must have independently measured launch, LDS, register, and inversion configurations.
- Both architectures provide the integer operations required for an efficient 8x32-bit secp256k1 implementation, including carry-generating addition, add-with-carry, high/low multiplication, and 32x32-to-64 multiply-add.

References:

- [ROCm GPU hardware specifications](https://rocm.docs.amd.com/en/docs-7.2.3/reference/gpu-arch-specs.html)
- [AMD Instinct MI300 CDNA3 ISA](amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct CDNA4 ISA](amd-instinct-cdna4-instruction-set-architecture.pdf)

## 4. Target repository layout

The organization follows the general `apps`, `include`, `src`, `cmake`, profiler, and run-script structure of [GPU_Roofline_Tools](https://github.com/hibagus/GPU_Roofline_Tools), while using target-scoped modern CMake.

```text
RCKangaroo/
├── CMakeLists.txt
├── CMakePresets.json
├── apps/
│   └── rckangaroo.cpp
├── include/rckangaroo/
│   ├── ec.hpp
│   ├── types.hpp
│   ├── config.hpp
│   └── gpu/kangaroo.hpp
├── src/
│   ├── core/
│   │   ├── ec.cpp
│   │   └── utils.cpp
│   └── hip/
│       ├── device.cpp
│       ├── kernels.hip
│       ├── kernel_params.hpp
│       ├── tuning.hpp
│       └── field/
│           ├── portable.hpp
│           ├── amdgcn.hpp
│           └── secp256k1.hpp
├── src/amdgcn/
│   ├── gfx942/
│   └── gfx950/
├── tests/
│   ├── unit/
│   ├── gpu/
│   └── end_to_end/
├── benchmarks/
│   ├── field_ops.hip
│   ├── inversion.hip
│   └── kangaroo.hip
├── scripts/
│   ├── benchmark/
│   └── profile/
├── cmake/
│   ├── CompilerWarnings.cmake
│   ├── ROCmSettings.cmake
│   └── Sanitizers.cmake
├── docs/
│   ├── ROCM_PORTING_PLAN.md
│   ├── PORTING.md
│   ├── PROFILING.md
│   ├── PERFORMANCE.md
│   ├── PHASE5_WAVE64_INVERSION.md
│   └── *.pdf
└── legacy/cuda/
    ├── asm/
    ├── cubin/
    └── msvc/
```

HIP will be the default backend. The existing CUDA implementation will be retained as a legacy correctness and performance reference until the ROCm port is established.

## 5. Implementation phases

### Phase 0: Freeze behavior and build a correctness oracle

Goals:

- Make tests deterministic before modifying GPU arithmetic.
- Establish independent reference results.

Tasks:

- Add `--seed`, `--iterations`, and finite benchmark-duration options.
- Record known end-to-end solutions and CPU ECC outputs.
- Implement independent secp256k1 reference arithmetic with `boost::multiprecision::cpp_int`.
- Test field add, subtract, negate, multiply, square, reduce, and inverse.
- Test point addition, doubling, and scalar multiplication.
- Add known vectors for compressed and uncompressed public keys.
- Add `static_assert` checks for kernel parameter sizes, alignment, and offsets.
- Replace ABI-sensitive `bool` kernel fields with fixed-width integer fields.

Progress as of 2026-09-28:

- [x] Added a CPU-only `rckangaroo_core` build path and CTest target.
- [x] Added deterministic Boost.Multiprecision oracle tests for field arithmetic,
  inversion, square roots, point parsing, point addition/doubling, and scalar
  multiplication.
- [x] Corrected non-canonical CPU results for modular negation of zero and
  modular multiplication results greater than or equal to `p`.
- [x] Added `--seed`, `--iterations`, and `--duration`, including strict parsing,
  deterministic per-case RNG streams, documentation, and unit tests.
- [x] Recorded machine-readable end-to-end vectors for private keys 2 and 3 and
  locked their public points into the CPU oracle.
- [x] Replaced the GPU parameter `bool` with `u32` while preserving the 192-byte
  layout, and asserted every field offset in CPU and CUDA compilation paths.

Exit criteria:

- CPU reference tests are deterministic and pass.
- Known end-to-end cases are recorded.
- Kernel ABI layout is explicit and checked at compile time.

### Phase 1: Reorganize the repository

Perform a mechanical move without changing algorithms:

- `RCKangaroo.cpp` -> `apps/rckangaroo.cpp`
- `Ec.*` and `utils.*` -> `src/core` plus public headers.
- `GpuKang.*` -> the HIP host layer plus public GPU header.
- `RCGpuCore.cu` -> `src/hip/kernels.hip`.
- Split `defs.h` into public types, configuration, and private kernel parameters.
- Move CUDA assembly, cubins, Visual Studio files, and `CallCubin.*` under `legacy/cuda`.
- Add generated results and build products to `.gitignore`.
- Preserve history with move-only commits where practical.

Create these CMake targets:

- `rckangaroo_core`
- `rckangaroo_hip`
- `rckangaroo`
- `rckangaroo_unit_tests`
- `rckangaroo_gpu_tests`
- `rckangaroo_bench`

Create presets for:

- `gfx942-release`
- `gfx950-release`
- `rocm-fat-release` (`gfx942;gfx950`)
- `host-debug-asan`
- `rocm-profile`

Use CMake's native HIP language and target property `HIP_ARCHITECTURES`. Avoid global compiler flags and avoid relocatable device code in the release path unless a real cross-translation-unit device-link requirement appears.

Exit criteria:

- The reorganized legacy CUDA sources remain buildable where CUDA is available.
- The new target structure configures cleanly with ROCm.
- No functional changes are mixed into the repository-layout commit.

### Phase 2: Implement the portable HIP baseline

Port the generic CUDA path before attempting the turbo implementation:

- Convert CUDA runtime calls to HIP.
- Launch `KernelGen`, `KernelA`, `KernelB`, and `KernelC` as directly linked HIP kernels.
- Use HIP constant memory for the secondary jump table.
- Remove the CUDA cubin loader from the default HIP path.
- Remove CUDA persisting-L2 configuration.
- Preserve the existing SoA state layout for the first baseline.
- Replace inline PTX with portable explicit 8x32-bit carry and multiply primitives.
- Set dynamic LDS to the amount each generic kernel actually uses.
- Add HIP error checking around every allocation, copy, launch, and synchronization boundary.
- Use one HIP stream per selected GPU.

Exit criteria:

- `gfx942`, `gfx950`, and fat binaries compile.
- GPU field-operation tests pass on MI355X.
- KernelGen and fixed one-step jump tests match CPU results.
- Fixed end-to-end solver tests match the CPU/reference implementation.
- A first per-GPU MI355X throughput number is recorded.

Completion record (2026-09-28):

- Native HIP builds completed for `gfx942`, `gfx950`, and a combined
  `gfx942;gfx950` binary with ROCm 7.2.
- Portable field tests passed for 256 deterministic add, subtract, multiply,
  square, and inverse vectors on MI355X.
- `KernelGen` and a one-step `KernelA` run matched the CPU implementation for
  6,144 kangaroos.
- Both fixed range-32 end-to-end vectors recovered the expected keys (`2` and
  `3`) on one MI355X.
- The first single-GPU portable `KernelA` baseline measured 113.918 MKeys/s on
  MI355X. See [PERFORMANCE.md](PERFORMANCE.md) for the exact workload and
  limitations.
- Real MI300X execution is still required before release support is declared;
  Phase 2 validates `gfx942` by cross-compilation only.

### Phase 3: Establish repeatable profiling

Add benchmark and profiling scripts that record:

- Git revision and dirty state.
- ROCm and compiler versions.
- GPU model and LLVM target.
- Clock, power, temperature, and memory state.
- Range, DP setting, seed, workgroup size, group count, and iteration count.
- Warm-up duration and sample duration.
- Per-kernel HIP event time and end-to-end MKeys/s.

Use `rocprofv3`, `rocprof-compute`, compiler resource reports, and `llvm-objdump` to collect:

- VGPR, SGPR, LDS, and scratch usage.
- Achieved wave occupancy and active cycles.
- VALU, SALU, VMEM, and LDS instruction mix.
- L1/L2/L3/HBM traffic and hit rates.
- Atomic traffic and stalls.
- Generated ISA for all hot variants.

Benchmark protocol:

- Warm up the GPU before sampling.
- Use multiple fixed-duration samples.
- Report median and dispersion, not only the best sample.
- Separate pure GPU throughput from normal DP-processing throughput.
- Keep power and clock conditions consistent across comparisons.

Exit criteria:

- A reproducible baseline profile exists for every hot kernel.
- Benchmark output contains enough metadata to reproduce the run.
- Before/after profiler comparisons can be generated automatically.

Completion record (2026-09-28):

- Added opt-in HIP-event timing for KernelGen/A/B/C and end-to-end throughput;
  normal runs retain the event-free path.
- Added fixed-duration warm-up/sample automation with JSON records for Git,
  ROCm/compiler, GPU identity, workload, launch geometry, clock, power,
  temperature, memory, median, and MAD.
- Captured an MI355X baseline for all hot kernels with rocprofv3 dispatch,
  memory-copy, and scratch-allocation traces.
- Added code-object extraction, compiler resource parsing, categorized ISA
  counts, full disassembly archives, and automatic before/after comparisons.
- Added a rocprof-compute preflight wrapper. The current system installation is
  missing its pinned Python dependencies, so hardware-counter collection is
  explicitly diagnosed rather than silently skipped or partially recorded.
- Documented the protocol and current results in [PROFILING.md](PROFILING.md)
  and [PERFORMANCE.md](PERFORMANCE.md).
- MI300X remains compile/ISA-extraction only until the workflow is run on real
  `gfx942` hardware.

### Phase 4: Optimize secp256k1 field arithmetic

Build isolated variants before integrating them into the solver:

1. Portable compiler-generated 8x32-bit arithmetic.
2. AMDGCN carry chains using `V_ADD_CO_U32` and `V_ADDC_CO_U32`.
3. Product accumulation using `V_MAD_U64_U32`.
4. Product accumulation using explicit `V_MUL_LO_U32`/`V_MUL_HI_U32` plus carry chains.

Rules:

- Keep a dependent carry chain in one inline-assembly block so LLVM cannot clobber its VCC state.
- Preserve the reduction specialized for `p = 2^256 - 2^32 - 977`.
- Compare generated code and inline assembly rather than assuming assembly is faster.
- Maintain a portable fallback for tests and unsupported targets.
- Reject any hot variant that spills into scratch unless it wins decisively and the end-to-end measurement confirms the benefit.

Inversion candidates:

- Existing divstep/safegcd-style inversion.
- Fixed addition-chain exponentiation.
- Per-lane Montgomery batching.
- Wave64-cooperative Montgomery batching.

Exit criteria:

- Every arithmetic variant passes randomized differential tests.
- Winning multiplication, squaring, and inversion variants are selected independently for `gfx942` and `gfx950`.
- Generated ISA and resource usage are archived with the benchmark result.

Implementation checkpoint (2026-09-28), with measurements in
[PHASE4_FIELD_ARITHMETIC.md](PHASE4_FIELD_ARITHMETIC.md):

- Separated the production API, portable reference arithmetic, and candidate
  implementations under `src/hip/field/`.
- Added compiler-generated Comba, explicit AMDGCN multiply/carry, and single-block
  `V_ADD_CO_U32`/`V_ADDC_CO_U32` carry-chain variants with portable fallbacks.
- Added a fixed `p - 2` addition chain using 255 squarings and 15 multiplies.
- Added 4,096-vector randomized arithmetic tests and 256-vector inversion tests
  against a Boost.Multiprecision oracle. Both known-key end-to-end tests pass.
- On MI355X, selected compiler-generated Comba MAD for multiply/square and the
  same backend with the fixed addition chain for inversion. It was faster than
  explicit assembly and introduced no spills in the integrated hot kernels.
- Archived standalone and integrated `gfx942`/`gfx950` code-object metadata and
  disassembly under ignored `profiles/phase4-*` directories. The two targets
  currently produce matching field-kernel instruction/resource summaries.

The `gfx950` exit criteria are satisfied. The `gfx942` code compiles and is
spill-free, but its winner remains provisional until the same correctness and
benchmark protocol is run on physical MI300X hardware. Per-lane and wave64
batching remain Phase 5 architecture work rather than prerequisites for this
field-arithmetic checkpoint.

### Phase 5: Redesign inversion for wave64

The NVIDIA inversion-worker topology must be reconsidered, not copied. Benchmark:

1. Per-lane batched inversion as the correctness baseline.
2. Wave64-cooperative Montgomery inversion using shuffle or DPP operations.
3. Persistent producer and inversion-worker workgroups using queues and mailboxes.

For the wave-cooperative candidate:

- Combine products across active lanes.
- Perform one or a small number of inversions per wave.
- Recover per-lane results with prefix/suffix products.
- Handle inactive lanes explicitly and test partial waves.

For the persistent-worker candidate:

- Do not assume a workgroup is permanently mapped to a particular CU unless resource limits and launch geometry enforce the required residency.
- Derive mailbox counts from wave64, not from NVIDIA's eight-warps-per-block assumption.
- Use bounded queues and explicit progress/termination rules.
- Include queue contention and cache traffic in the decision.

Exit criteria:

- The selected inversion architecture is faster end-to-end, not only in a microbenchmark.
- Queue/progress tests run without deadlock under long stress tests.
- The design works for both single- and multi-GPU execution.

Implementation checkpoint (2026-09-28), with measurements in
[PHASE5_WAVE64_INVERSION.md](PHASE5_WAVE64_INVERSION.md):

- Added a wave64 Montgomery-batch primitive using prefix/suffix products and
  wave shuffles, with explicit zero and partial-wave handling.
- Added independent GPU correctness coverage for 2,048 values and active
  widths from 1 through 64.
- Added a reproducible per-lane versus wave64 benchmark plus `gfx942`/`gfx950`
  resource and ISA extraction.
- On MI355X, the wave64 candidate reached 0.107435 Ginversion/s versus
  0.275659 Ginversion/s for the per-lane baseline: 0.390x the rate and 2.566x
  the elapsed time.
- The candidate increased demand from 126 VGPR / 46 SGPR with no spills to
  128 VGPR / 76 SGPR with eight VGPR spills and 36 private bytes.
- Retained the uniform per-lane fixed addition chain in production. No
  `KernelA` integration was made because the isolated primitive lost
  decisively.
- Pruned the persistent producer/worker queue before implementation. Moving a
  fixed-chain inversion to one lane does not reduce wave instruction issue,
  while a queue adds synchronization, traffic, and residency constraints.

For the selected design, queue/progress tests are not applicable because no
queue is present. The `gfx950` study is complete with the existing per-lane
architecture selected. `gfx942` targets compile and have matching resource
metadata, but physical MI300X correctness and performance validation remain.

### Phase 6: Tune the hot kernel per architecture

Benchmark matrix:

| Parameter | Candidate values |
|---|---|
| Workgroup size | 64, 128, 256 |
| Points per lane | 8, 12, 16, 24, 32 |
| Iterations per launch | 256, 512, 1000, 2048 |
| Jump-table storage | constant, global read-only, 32/48 KiB LDS, tiled LDS, 96 KiB LDS on `gfx950` |
| Kernel dataflow | split A/B, partially fused, fully fused where registers permit |
| State layout | current group-major, workgroup-major |
| Jump-list stores | cached, non-temporal, eliminated by fusion |

`gfx942` priorities:

- Keep a workgroup within the 64 KiB LDS limit.
- Compare 32 KiB and 48 KiB table arrangements.
- Preserve useful CU residency while avoiding VGPR spills.

`gfx950` priorities:

- Compare the 96 KiB combined jump table against smaller LDS arrangements.
- Determine whether reduced cache traffic offsets one-workgroup-per-CU residency.
- Exploit the larger LDS only when the end-to-end result supports it.

Both targets:

- Template compile-time geometry rather than retaining hidden `256` constants.
- Preserve coalesced vector loads and stores with explicit alignment.
- Evaluate workgroup-major state storage for XCD/cache locality.
- Reassess the Kernel A/B jump-list boundary; fusion may remove global traffic but increase VGPR pressure.
- Keep rare loop-recovery work outside the main hot path.

Exit criteria:

- A measured tuning record exists for both targets.
- Architecture-specific defaults are selected through a small tuning table.
- Command-line overrides remain available for experimentation.

### Phase 7: Optimize host and multi-GPU execution

- Replace synchronous DP transfers with pinned, double-buffered memory and `hipMemcpyAsync`.
- Maintain independent streams and output rings per GPU.
- Shard the CPU distinguished-point database to remove the global insertion lock.
- Separate GPU workers from DP-consumer threads.
- Pin worker and consumer threads according to GPU/NUMA topology.
- Modernize GPU selection to accept lists such as `--gpu 0,3,5`.
- Measure one-, two-, four-, and eight-GPU scaling.
- Keep the normal hot path device-local; peer/XGMI transfers are not expected to be necessary.

Exit criteria:

- The host DP pipeline does not reduce pure GPU throughput at the supported DP settings.
- Scaling efficiency and CPU utilization are documented for 1/2/4/8 MI355X GPUs.
- Shutdown, failure, and buffer-overflow paths are tested.

### Phase 8: Add handwritten AMD ISA only if justified

If tuned HIP and inline AMDGCN remain limited by compiler scheduling:

- Add separate assembly sources under `src/amdgcn/gfx942` and `src/amdgcn/gfx950`.
- Build code objects from source; do not commit generated binaries.
- Load them through a small HIP module abstraction.
- Retain the directly compiled HIP fallback.
- Store disassembly and resource metadata with benchmark artifacts.
- Document and pin the ROCm compiler/toolchain used to build validated code objects.

This phase should target profiler-proven bottlenecks rather than rewriting every kernel in assembly.

Exit criteria:

- Assembly variants have complete correctness coverage.
- They provide a material, repeatable end-to-end improvement over the best HIP variant.
- Both architecture implementations remain source-controlled and reproducible.

### Phase 9: Documentation and continuous validation

- Document build and run instructions in `README.md`.
- Document CUDA-to-HIP differences and removed CUDA features in `docs/PORTING.md`.
- Record benchmark methodology and results in `docs/PERFORMANCE.md`.
- Add CPU-only CI for formatting, warnings, and reference tests.
- Add compile-only ROCm CI for `gfx942` and `gfx950` where GPU runners are unavailable.
- Add self-hosted MI300X and MI355X correctness/performance jobs when hardware is available.
- Treat performance regressions separately from functional CI failures.

## 6. Correctness test plan

### CPU unit tests

- Field boundary values: zero, one, `p-1`, `p`, maximum 256-bit values.
- Random field add/subtract/multiply/square/inverse comparisons.
- Point addition/doubling edge cases.
- Scalar multiplication known vectors.
- Serialization and compressed-key parsing.
- Distinguished-point database insert/find/save/load behavior.

### GPU arithmetic tests

- Large randomized batches for every field primitive.
- Aliased input/output cases supported by the production functions.
- Results around reduction and carry boundaries.
- Inverse verification with `a * inv(a) mod p == 1`.
- Cross-comparison of portable, inline-assembly, and code-object implementations.

### GPU kernel tests

- KernelGen output versus CPU scalar multiplication.
- One jump and multiple jumps versus a CPU simulator.
- Jump inversion flag behavior.
- Distinguished-point detection and serialization.
- Loop detection and Kernel C recovery.
- DP and loop buffer capacity boundaries.
- Multiple block, workgroup, and point-group configurations.

### End-to-end tests

- Known small-range keys with deterministic seeds.
- Tame generation followed by reload and solve.
- Single- and multi-GPU consistency.
- Long-running stress test with error counters and guard regions.

## 7. Performance acceptance criteria

Correctness always takes precedence over throughput. A configuration is eligible for release only when all matching correctness tests pass.

Performance completion requires:

- Release builds for `gfx942`, `gfx950`, and a combined fat binary.
- No accidental scratch spills in the selected hot kernels.
- Reproducible fixed-seed benchmarks with warm-up and multiple samples.
- Separately selected MI300X and MI355X configurations.
- Recorded generated ISA, register use, LDS use, and occupancy.
- Normal DP handling that does not materially throttle the GPU kernels.
- Measured 1/2/4/8-GPU scaling on the available MI355X system.
- A real MI300X correctness and performance run before declaring MI300X support complete.
- A long stress run without incorrect keys, GPU errors, deadlocks, or buffer corruption.

Absolute throughput targets should be set after the portable HIP baseline exists. Optimization decisions will use measured end-to-end MKeys/s rather than instruction-count estimates alone.

## 8. Planned commit sequence

Keep commits independently reviewable:

1. Add deterministic CPU reference tests and known vectors.
2. Reorganize files without functional changes.
3. Introduce native HIP CMake targets and presets.
4. Port the host runtime and generic kernels.
5. Add portable GPU field arithmetic and GPU tests.
6. Establish the first correct MI355X end-to-end baseline.
7. Add profiling and benchmark automation.
8. Optimize field multiplication, squaring, reduction, and inversion.
9. Implement and compare wave64 inversion/dataflow designs.
10. Tune `gfx942` and `gfx950` kernel configurations.
11. Optimize asynchronous DP handling and multi-GPU scaling.
12. Add handwritten AMD ISA only where profiler results justify it.
13. Finalize documentation, MI300X validation, and release configuration.

## 9. Decision log

Record major choices here as implementation progresses.

| Date | Decision | Evidence |
|---|---|---|
| 2026-09-28 | Use a portable HIP baseline before architecture-specific assembly. | Current turbo path is NVIDIA SASS and depends on warp32/CUDA behavior. |
| 2026-09-28 | Treat `gfx942` and `gfx950` as separate tuning targets. | LDS capacity and CU counts differ materially. |
| 2026-09-28 | Retain CUDA sources under `legacy/cuda` until ROCm parity is established. | Provides a correctness and performance reference during the port. |
| 2026-09-28 | Isolate the CPU core and test it before moving source files. | This provides a stable correctness oracle while repository and GPU code change. |
| 2026-09-28 | Require canonical CPU field results in `[0, p)`. | The independent oracle exposed `p` for `-0` and `p + r` after some multiplications. |
| 2026-09-28 | Derive separate deterministic RNG streams for each benchmark case and GPU initialization. | Re-seeding one shared stream inside `SolvePoint` would otherwise repeat benchmark keys. |
| 2026-09-28 | Preserve the legacy `TKparams` size and offsets while replacing `bool` with `u32`. | Existing cubins consume the 192-byte binary layout; `u32` occupies the same padded slot and is explicit for HIP. |
| 2026-09-28 | Use 32 KiB, 48 KiB, and 48 KiB of dynamic LDS for portable Kernels A, B, and C. | These are the tables actually accessed by the generic kernels and all fit the MI300X 64 KiB limit. |
| 2026-09-28 | Use fixed-exponent Fermat inversion in the initial portable field layer. | It is simple, target-independent, and passed independent GPU field tests; inversion optimization remains explicit Phase 4/5 work. |
| 2026-09-28 | Keep long solver vectors opt-in with `RCK_ENABLE_SLOW_GPU_TESTS`. | Each vector allocates about 4.7 GiB and took 25–32 seconds on the current MI355X baseline. |
| 2026-09-28 | Make HIP-event timing opt-in through `RCK_PROFILE`. | It preserves the normal execution path while providing machine-readable per-kernel timing during controlled runs. |
| 2026-09-28 | Use fixed-duration samples and median/MAD as the comparison baseline. | The old cold single-launch result varied with GPU clock state and understated warmed throughput. |
| 2026-09-28 | Retain both compiler metadata and profiler allocation counts. | Static register demand and allocation-granularity counts differ but are independently useful for occupancy work. |
| 2026-09-28 | Select compiler-generated Comba MAD and fixed addition-chain inversion on `gfx950`; keep explicit assembly as a measured fallback. | Comba MAD was 13.1x faster for multiply and its chain inversion was 24.6x faster than the portable baseline; explicit MUL/carry assembly was slower. |
| 2026-09-28 | Retain per-lane inversion and reject wave64 batching and a persistent inversion queue for the current fixed chain. | The queue-free wave64 primitive was correct but 2.566x slower on MI355X, spilled eight VGPRs, and still issued the lane-0 inversion as a wave instruction stream. |

## 10. Progress checklist

- [x] Phase 0: deterministic correctness oracle
- [x] Phase 1: repository reorganization
- [x] Phase 2: portable HIP baseline
- [x] Phase 3: repeatable profiling
- [ ] Phase 4: optimized field arithmetic (`gfx950` complete; `gfx942` runtime pending)
- [ ] Phase 5: wave64 inversion design (`gfx950` complete; `gfx942` runtime pending)
- [ ] Phase 6: per-architecture kernel tuning
- [ ] Phase 7: host and multi-GPU optimization
- [ ] Phase 8: optional handwritten AMD ISA
- [ ] Phase 9: documentation and continuous validation
