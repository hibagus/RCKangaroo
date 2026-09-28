# Phase 4 assessment: is a hand-written kernel worth it?

Phase 4 in the port plan is writing KernelA's inner loop as a single assembly routine with
explicit register allocation - what RC's `main.asm` is. It is a large undertaking, so
before starting it the question was which cost it would actually remove.

**Conclusion: it targets the wrong thing.** KernelA's dominant cost is memory traffic, not
instruction count, and the traffic is not reducible. Realistic remaining headroom is
roughly 10-20%, and Phase 4 addresses only part of that.

## Ablation of KernelA

Each variant disables or alters one thing. All are deliberately incorrect - this measures
cost, not behaviour. Baseline is one MI300X on puzzle #140.

| Variant | MKeys/s | vs baseline |
|---|---|---|
| baseline (correct) | 9,978 | - |
| **state access forced cache-resident** | **14,218** | **+42%** |
| DP check removed | 9,982 | +0.1% |
| jump-list store removed | 9,994 | +0.2% |
| LastPnts tail stores removed | 9,990 | +0.2% |
| constant `jmp_ind` (no divergent LDS gather) | 9,670 | -3% |
| workgroup-contiguous state layout | 6,285 | **-37%** |

The state-resident variant keeps every instruction and only shrinks the working set from
717 MB to ~21 MB. So **memory traffic costs 42%**, and everything else - the DP check, the
jump list, the LastPnts ring, the divergent LDS gather - is free.

## Why the memory traffic cannot be reduced

Four approaches were measured. All failed.

**Cache residency does not help, tested two independent ways.** The state array is 96 bytes
per kangaroo, so its size is set by the kangaroo count:

| Config | kangaroos | state array | fits 256 MB MALL? | MKeys/s |
|---|---|---|---|---|
| 1 block/CU | 2.49 M | 239 MB | **yes** | 8,566 |
| 2 blocks/CU | 4.98 M | 478 MB | no | 9,701 |
| 3 blocks/CU | 7.47 M | 717 MB | no | **9,957** |

Sweeping `PNT_GROUP_CNT` earlier gave the same answer: the 171 MB configuration ran at
8,363 against 9,444 for a 456 MB one. Fitting the Infinity Cache is consistently *slower*,
because the configurations that fit are the ones with worse inverse amortisation and less
per-thread ILP. The MALL is a memory-side cache and does not deliver enough on this access
pattern to pay for that.

**The strided layout is correct, not a weakness.** Groups are 7.47 MB apart, so a
workgroup's 32 group-accesses are 32 scattered 8 KB regions. That looks hostile, and making
them one contiguous 256 KB block - same instruction count, same bytes moved, better DRAM
row locality - is **37% slower**. The stride is what spreads accesses across HBM channels;
concentrating them starves channel parallelism. This layout is load-bearing.

**Software pipelining does not help.** Issuing each group's state loads one iteration ahead,
the overlap RC hand-schedules, cost 2.2%. With `MemUnitStalled` at 0.65% there is no
latency to hide, and the register rotation adds eight moves per group.

**The bytes are close to inherent.** Per point-addition, ~224 bytes:

| | bytes | reducible? |
|---|---|---|
| kangaroo x, y - load and store | 128 | no: each kangaroo's state must be read and written every step, and steps are serial per kangaroo |
| batched-inverse prefix products - store and load | 64 | no: inherent to Montgomery batch inversion. Will not fit LDS (256 KB needed per workgroup, 64 KB available) and cannot use AGPRs without fully unrolling the group loop, which would blow the instruction cache |
| jmp_y from global | 32 | partly - moving it back to LDS costs a wave of occupancy |

Affine coordinates are already the memory-minimal choice; projective would need three
values per kangaroo instead of two, and the batched inverse already makes inversion cheap.

## What is left, and what Phase 4 would get

Measured ceilings on one MI300X:

| | GH/s | |
|---|---|---|
| Current | 10.0 | |
| Memory traffic free (impossible) | 14.2 | ablation |
| Arithmetic alone, no structure (impossible) | 22.9 | `bench/arith_rate.hip` |
| HBM bandwidth wall | ~14-15 | 224 B/point-add against ~4 TB/s achievable |

Phase 4 removes instructions. The instruction gap to RC is 1.39x (1,971 against ~1,420 per
point-addition), and most of the excess is register moves the compiler inserts around
inline-asm boundaries - 347 per group body, of which the asm blocks themselves contain
zero. A hand-written kernel would remove those.

But the memory-free ceiling is 14.2 GH/s and the bandwidth wall sits at about the same
place, so even a perfect kernel cannot exceed ~14 GH/s with this algorithm. That bounds
Phase 4 at roughly **+40% in the best case and realistically much less**, for a rewrite of
~2,500 lines of hand-scheduled assembly with no differential test able to cover it - only
K-value and solve checks.

## Recommendation

Do not start Phase 4 as the next step. The evidence says the arithmetic is done (87% of
VALU peak in isolation), the bookkeeping is free, the memory traffic is irreducible, and
the layout is already right.

The cheap items that remain:

- **CPX compute partitioning is still untested.** It needs root
  (`rocm-smi --setcomputepartition`), which was not available. It is the one structural
  change not yet measured: eight logical devices of 38 CUs, each one XCD with a private
  4 MB L2 and all traffic XCD-local. `MAX_GPU_CNT` has been raised to 64 on the AMD path so
  a node enumerates correctly under it. Expect little given the residency results, but it
  is untested rather than ruled out.
- **jmp_y back in LDS**, trading 32 of 224 bytes per point-addition against a wave of
  occupancy. Small and unmeasured.
- **Host-side usability** carried over from AMDKangaroo: dynamic DP validation, per-GPU
  statistics, progress display.
