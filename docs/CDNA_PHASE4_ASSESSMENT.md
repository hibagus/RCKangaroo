# Phase 4 assessment: is a hand-written kernel worth it?

Phase 4 in the port plan is writing KernelA's inner loop as a single assembly routine with
explicit register allocation - what RC's `main.asm` is. It is a large undertaking, so
before starting it the question was which cost it would actually remove.

**Conclusion (revised): it is worth doing, incrementally.** An earlier version of this
document concluded the opposite - that Phase 4 targeted the wrong cost - on the grounds
that memory traffic dominates. That inference was wrong, and the correction is recorded in
"Marginal value of instruction reduction" below: removing instructions converts to
throughput almost 1:1, because memory and compute overlap rather than adding. Expect
roughly +12-35%.

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

## Marginal value of instruction reduction

The ablation above shows memory traffic costs 42% of runtime. It is tempting to conclude
from that alone that reducing instructions cannot help much. That does not follow, and
measuring it directly gives the opposite answer.

This experiment drops arithmetic while keeping every memory access and the loop structure
identical, so only the instruction count moves. Measured on one CPX partition (38 CUs), so
the absolute figures are per-XCD; the ratios are the point.

| Variant | Instructions | MKeys/s | instruction ratio | speed ratio |
|---|---|---|---|---|
| baseline | 3,023 | 2,266 | - | - |
| -1 multiply | 2,778 | 2,482 | 1.088x | 1.095x |
| -2 multiplies | 2,532 | 2,705 | 1.194x | 1.194x |
| -3 multiplies | 2,274 | 2,900 | 1.329x | 1.280x |

**Instruction reduction converts to throughput at 96-100% efficiency.** Memory and compute
overlap; their costs are not additive. So the 42% memory figure and a near-linear response
to instruction count are both true, and the kernel is instruction-bound at the margin.

Two ceilings, which an earlier version of this document conflated:

| | GH/s |
|---|---|
| Instruction-bound, memory free (the ablation) | 14.2 |
| Memory-bound, instructions free (220 B/point-add at 3.5-4.0 TB/s) | 16-18 |

14.2 is not an absolute wall - it is the wall at the *current* instruction count.

## Recommendation

Phase 4 is worth doing. Cutting the 347 register moves and 363 wait states per group body
is about 1.44x fewer instructions, which at the measured conversion rate is roughly
**11-13.5 GH/s per GPU, 93-112 GH/s for the node**.

But not as a single monolithic hand-scheduled kernel, at least not first. The reason is
testability: every primitive today has defined semantics and is checked against Python
arbitrary-precision ground truth over 10,000,000 vectors. A monolithic kernel has no such
reference - only K values and solve counts, which are statistical, and which would let a
subtle distance-accounting bug hide as "slightly worse K" rather than fail outright.

Staged instead, each step keeping the differential harness applicable:

1. Fuse `SubModP` into `MulModP` as `SubMulModP(res, a, b, c) = ((a - b) mod p) * c mod p`.
   It occurs four times per group, has well-defined semantics so it remains fully testable,
   and removes the intermediate's eight register moves per site.
2. Measure against the predicted saving. If the model holds, widen the fusion - the whole
   back-substitution step, then the elliptic-curve addition tail.
3. Consider a monolithic kernel only if 1 and 2 confirm the response stays linear as the
   blocks grow.

Stop at the first step where the measured return stops matching the model. That is what
happened with software pipelining, the contiguous layout, and cache residency, and it is
cheaper to discover early.

## Still open

- **CPX partitioning: measured, worth +3-5%.** See docs/CDNA_CPX_PARTITIONING.md.
- **jmp_y back in LDS**, trading 32 of 220 bytes per point-addition against a wave of
  occupancy. Small and unmeasured.
- **Host-side usability** carried over from AMDKangaroo: dynamic DP validation, per-GPU
  statistics, progress display.

---

# Phase 4 step 1: attempted, and the result closes Phase 4

Step 1 was to fuse `SubModP` into `MulModP`, on the theory that the 347 register moves per
group body were overhead at inline-asm block boundaries. Two measurements killed it, and a
third invalidated the metric the whole plan rested on.

## The Sub -> Mul boundary is already free

| | issue slots |
|---|---|
| `MulModP` alone | 270 |
| `SubModP` alone | 30 |
| Sub + Mul together | **290** |
| sum if the boundary cost nothing | 300 |
| boundary overhead | **-10** |

Fusing them recovers nothing - the pair together already costs *less* than the two
separately, so the compiler optimises across the boundary. The premise was wrong.

## Where the moves actually are

| Bucket | moves |
|---|---|
| Multiply and square column bookkeeping | 240 |
| `SubModP` internals | 28 |
| Kernel glue (addressing, loop, DP, jump list) | 79 |

**268 of 347 are inside the primitives**, and the multiply's are structurally required:
`v_mad_u64_u32` needs an even-aligned VGPR pair for its 64-bit addend, so a 32-bit-granular
sliding accumulator costs about two moves per column however it is arranged. The compiler
manages 40 per multiply where a naive hand-written version needs ~60. Hand-writing it would
be worse, not better.

## s_nop is free, which invalidates the "issue slots" metric

The fallback for step 1 was to move the reduction's remaining C++ tail into assembly. It
works and passes 10,000,000 differential vectors, and it cut `MulModP` from 240 VALU + 30
wait states to 238 + 17 - a 5.6% reduction by the slot metric used throughout this port.

Back-to-back A/B/A on the full node:

| | MulModP | max MKeys/s |
|---|---|---|
| A: C++ reduction tail | 240 + 30 | 84,341 |
| B: asm reduction tail | 238 + 17 | 84,232 |
| A again | 240 + 30 | 84,531 |

Run-to-run noise 0.23%; measured effect -0.24%. **No gain.**

The reason is that 13 of the 15 removed slots were `s_nop`. CDNA issues scalar and vector
instructions independently, so at 3 waves/SIMD another wave's VALU work fills the cycle a
wait state would otherwise occupy. **`s_nop` costs nothing at this occupancy**, and counting
it alongside VALU instructions - as this port's "issue slots" figure did throughout -
overstates the cost of everything that generates them.

The change was reverted: no measurable benefit, and it adds an asm block and eight
temporaries.

## Revised prize, and the recommendation

Recomputing on VALU count alone, which is what actually costs time:

| | |
|---|---|
| KernelA VALU per point-addition | 1,971 |
| Moves inside the multiply and square (required) | 240 |
| Moves inside `SubModP` (required) | 28 |
| **Kernel glue moves (removable)** | **79 = 4.0%** |

**Phase 4 is worth about 4%**, roughly 84,400 to 87,600 MKeys/s for the node - for
hand-writing and hand-scheduling ~2,500 lines of assembly with no differential test able to
cover it.

**Recommendation: close Phase 4.** The earlier estimate of +12-35% came from counting
`s_nop` as real cost and from assuming the moves were boundary overhead. Both were wrong,
and both were cheap to test. The arithmetic is at 87% of VALU peak in isolation, the
primitives' remaining moves are structurally required by the ISA, and what is left is a
4% sliver behind a large, poorly-testable rewrite.

## A note on measurement method

Two errors in this session came from reading the speed display too early. The solver
averages over a 16-entry ring initialised to zero, so figures taken before roughly 12-14
reports run low - one comparison was reported as a 2% regression that a back-to-back rerun
showed to be 0.24%. Any A/B here needs to be run back to back in one session, taking the
maximum over 13 or more samples, with the baseline re-measured alongside to establish the
noise floor.
