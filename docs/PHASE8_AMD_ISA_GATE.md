# Phase 8 AMD ISA assessment

- Date: 2026-09-28
- Base revision: `7db72b9b0690` plus the Phase 8 working tree
- ROCm: 7.2.0
- HIP compiler: AMD clang 22.0.0git, ROCm revision `7b800a194662`
- Runtime-tested GPU: AMD Instinct MI355X (`gfx950`)
- Compile/ISA-tested GPU: AMD Instinct MI300X (`gfx942`)

## Outcome

Do not add handwritten full-kernel AMD code objects at this checkpoint. The
best production path remains directly compiled HIP with compiler-generated
Comba `V_MAD_U64_U32` field arithmetic and the small inline AMDGCN carry block
already selected in Phase 4.

This is a measured rejection of the current explicit-ISA candidate, not a ban
on assembly. The new gate can be rerun after a field-primitive rewrite, a ROCm
compiler change, or on physical MI300X hardware. No `src/amdgcn/` hierarchy,
generated code object, or HIP module loader is added while there is no winning
candidate to maintain.

## Admission policy

Handwritten ISA has a higher correctness and maintenance cost than directly
compiled HIP, so it advances in two stages:

1. The candidate must pass the independent GPU differential suite and beat the
   selected compiler implementation by at least 3% in a 15-sample primitive
   benchmark. Each multiply, square, or inverse candidate is considered
   independently, so a win in one operation is not hidden by losses in others.
2. An admitted candidate must then pass the complete CPU/GPU/known-key suite
   and improve controlled end-to-end throughput by at least 2%, with the
   improvement also larger than three times the run-to-run MAD.

Only a candidate that clears both stages should become separate source-built
`gfx942` and `gfx950` code objects with a directly compiled HIP fallback. The
3% primitive threshold is above the dispersion observed in this study and
prevents a noisy isolated result from triggering a much larger integration.

The dependency-free runner implements the first stage and archives benchmark
metadata, correctness output, field and production code objects, compiler
resource metadata, disassembly, instruction summaries, JSON, and Markdown:

```sh
python3 scripts/benchmark/run_phase8_isa_gate.py \
  --preset mi355x --device 0 --samples 15
```

The official local result is retained under the ignored
`profiles/phase8-mi355x-isa-gate/` directory.

Run the same command with `--preset mi300x` on a physical MI300X. `--skip-build`
is available only when the matching binaries are already current.

## Correctness and MI355X measurement

Before timing, `rckangaroo_gpu_field_variant_tests` passed 4,096 arithmetic
vectors and 256 inversion vectors against the independent
Boost.Multiprecision oracle. The benchmark used 256 workgroups of 256 threads.
Multiply and square executed 256 dependent operations per lane; inversion
executed one fixed addition chain per lane. Values are medians of 15 HIP-event
samples after warm-up.

| Operation | Compiler Gop/s | Explicit ISA Gop/s | Candidate / compiler | Change | Gate |
|---|---:|---:|---:|---:|:---:|
| Multiply | 73.712630 | 60.409887 | 0.820x | -18.05% | fail |
| Square | 78.676517 | 62.127713 | 0.790x | -21.03% | fail |
| Inverse | 0.268940 | 0.223273 | 0.830x | -16.98% | fail |

All three explicit candidates regress by 17-21%, so none qualifies for an
end-to-end integration. The selected launch already uses exactly one
256-thread workgroup per CU and the integrated kernel does not spill, so the
candidate's lower VGPR count cannot unlock another launched workgroup or remove
scratch traffic. There is no measured mechanism to recover its primitive loss.

## Generated ISA evidence

The explicit path reduces VGPR demand, but replaces wide multiply-adds with
more multiply and carry instructions. It therefore executes a larger static
instruction stream and loses despite the lower register count.

| Kernel | VGPR / SGPR | Total inst. | `V_MAD_U64_U32` | `V_MUL_LO/HI` | `V_ADD_CO/ADDC_CO` |
|---|---:|---:|---:|---:|---:|
| Compiler multiply | 56 / 36 | 630 | 77 | 0 / 0 | 0 / 28 |
| Explicit multiply | 42 / 36 | 775 | 13 | 64 / 64 | 64 / 129 |
| Compiler square | 42 / 36 | 603 | 50 | 0 / 0 | 0 / 28 |
| Explicit square | 34 / 36 | 778 | 13 | 64 / 64 | 64 / 129 |
| Compiler inverse | 126 / 46 | 54,885 | 5,455 | 0 / 0 | 0 / 2,824 |
| Explicit inverse | 118 / 42 | 63,094 | 1,222 | 6,016 / 6,016 | 6,016 / 12,126 |

Every listed kernel has a zero-byte private segment and zero reported
SGPR/VGPR spills. The fresh `gfx942` and `gfx950` field code objects have
identical normalized resource and instruction summaries. The production hot
kernels are also spill-free; selected KernelA uses 146 VGPRs and 86 SGPRs on
`gfx942`, and 146 VGPRs and 82 SGPRs on `gfx950`.

The production `gfx950` KernelA disassembly contains 6,037
`V_MAD_U64_U32` instructions in its statically unrolled body. That is the
desired arithmetic instruction family, and the direct measurement provides no
evidence that replacing it with explicit MUL/carry scheduling is beneficial.

## Profiler and architecture limits

ROCm Compute Profiler could not collect hardware counters because the installed
ROCm 7.2 environment is missing its pinned Python dependencies, beginning with
`pandas`. This does not reverse the gate result: the candidate was measured
directly on its target operation and was substantially slower. Hardware-counter
collection remains useful if a new candidate passes the primitive gate.

MI300X is compile/ISA-validated only. A physical `gfx942` run must execute the
same correctness and gate protocol before making an MI300X performance claim.
If it admits an operation that MI355X rejects, an architecture-specific
`gfx942` implementation remains allowed by the plan. The MI355X portion of
Phase 8 is complete; the overall dual-target phase remains provisional until
that MI300X gate can run.

## Reopening Phase 8

Reopen the code-object implementation only when at least one of these changes
provides a concrete candidate:

- a new schedule retains `V_MAD_U64_U32` while shortening dependencies;
- a physical MI300X result exposes an architecture-specific win;
- hardware counters identify a compiler-induced wait or occupancy limit that
  an ISA prototype can address; or
- a ROCm compiler update materially changes the generated instruction mix.

Retain the full artifacts under ignored `profiles/` storage and pin the exact
ROCm/LLVM revision in the report whenever a candidate is accepted.
