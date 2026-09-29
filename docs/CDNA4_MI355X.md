# CDNA4 / MI355X: measured results and where the remaining performance is

First run of this port on real gfx950 hardware. Everything here was measured on a node of
8x AMD Instinct MI355X (`gfx950:sramecc+:xnack-`, 256 CU, 2400 MHz nominal, 288 GB HBM3E,
1400 W cap), ROCm 7.2.0 / HIP 7.2.26015 / AMD clang 22, SPX compute partitioning and NPS1
memory partitioning. MI300X figures are quoted from the CDNA3 documents in this directory
and were measured on different hardware, so cross-machine comparisons carry the usual
caveat.

Method throughout is the one `CDNA_PHASE4_ASSESSMENT.md` settled on: the solver averages
speed over a 16-entry ring initialised to zero, so every figure is the **maximum over at
least 13 reports**, and every comparison is run back to back in one session on one GPU.
Repeat measurements of an unchanged binary drifted by up to **1.3%**, so anything below
about 1.5% here is noise, not a result.

It also means one GPU at a time and **nothing on any other GPU**: a first pass at the
solver-throughput numbers below had to be thrown away because a crashed `rocprofv3` run had
left an orphaned solver on a neighbouring device. `pgrep` is not enough to catch that, since
it cannot distinguish a harness wrapper from a GPU-resident process. Use

```sh
rocm-smi --showpids     # must list exactly the one process, on the one GPU
```

before every run.

## Headline

| | MI300X (CDNA3) | MI355X (CDNA4) | |
|---|---|---|---|
| Per GPU, SPX | 10.1 GH/s | **15.9 GH/s** | +57% |
| Node of 8, SPX | 81.0 GH/s | **126.2 GH/s** | +56% |
| Node of 8, best measured | 85.5 GH/s (CPX) | 126.2 GH/s (SPX, CPX untested) | +48% |

MI355X figures are with the changes in this document. Before them the port measured
15.5 GH/s per GPU and 122.8 GH/s for the node, so they are worth +2.1% and +2.7%.

Node startup also went from **over four minutes to three seconds** - see the RNG change
below. That is latency, not throughput, but it dominated the wall time of every short run.

Correctness holds. `tests/run_diff.sh` passes 10,000,000 vectors against Python
arbitrary-precision ground truth on gfx950 for all eight primitives (`MulModP`, `SqrModP`,
`SubModP`, `AddModP`, `NegModP`, `InvModP`, `Add192to192`, `Sub192from192`), and benchmark
mode solves real keys with zero reported errors.

Existing tuning carried over unchanged. `RCK_BLOCKS_PER_CU` was re-swept on MI355X and 3 is
still the optimum:

| blocks/CU | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| MKeys/s | 11,143 | 14,333 | **15,676** | 15,036 | 15,562 |

## The gain is not compute

The obvious guess is that CDNA4 executes this integer workload faster per clock. It does
not. `bench/isa_rate.hip`, same harness as the CDNA3 Phase 0 measurements:

| Instruction sequence | MI300X | MI355X | ratio |
|---|---|---|---|
| `v_add_u32`, 8 independent | 33.76 Tops/s | 33.02 Tops/s | 0.98 |
| `v_mad_u64_u32`, 8 independent | 27.59 | 27.96 | 1.01 |
| `v_mad_u64_u32`, 1 serial chain | 28.98 | 30.54 | 1.05 |
| `v_add_co_u32`, 8 independent | 29.11 | 28.44 | 0.98 |
| `v_addc_co_u32`, serial VCC chain | 32.35 | 32.55 | 1.01 |
| `v_accvgpr_write/read` | 32.43 | 28.45 | 0.88 |
| `ds_read_b128`, divergent 32 KB gather | 0.98 | 1.15 | 1.17 |

Per-instruction rates are within a few percent of CDNA3, and MI355X has *fewer* CUs
(256 vs 304). Nominal VALU peak is actually 4% lower: 256 x 64 x 2.4 GHz = 39.3 Tops/s
against MI300X's 40.9.

**gfx950 also adds no new integer arithmetic.** Checked by assembling a candidate list
against both targets with `llvm-mc -mcpu=gfx942` and `-mcpu=gfx950`: every integer
instruction this port uses or could use exists on both. The only gfx950-exclusive
additions found were `v_permlane16_swap_b32`, `v_permlane32_swap_b32` (lane shuffles for
matrix-core layout) and `v_prng_b32`. In particular **there is still no multiply-add with
carry-in**, so the structural 1.57x instruction penalty against NVIDIA's `IMAD.WIDE.U32.X`
described in `CDNA_VS_NVIDIA.md` is unchanged on CDNA4.

So the speedup comes from memory bandwidth and from sustained clock, not from the ALUs.

## MI355X is power-limited, and that changes what optimisation means here

Sampled every 5 s during a steady-state run, one GPU, `-dp 30 -range 139`:

| | clock | power | throughput |
|---|---|---|---|
| KernelA as shipped | **2140 MHz** of 2400 | **1290 W** of 1400 | 15,871 MKeys/s |
| Same kernel, kangaroo state forced cache-resident | **2358 MHz** | **1225 W** | 18,644 MKeys/s |

Junction temperature stayed at 72-76 C throughout, so this is the power cap, not thermal
throttling.

The second row is a deliberately-incorrect ablation (block slices folded 24:1, so the
kangaroo state shrinks from ~600 MB to ~25 MB and stays in cache). It executes **the same
instruction stream** - 2406 VALU against 2403 - and differs only in DRAM traffic. Cutting
that traffic **drops power by 65 W and raises the sustained clock by 218 MHz**, and about
10 of its 17.5 percentage points of speedup are that clock.

This is the CDNA4-specific finding: **DRAM traffic costs power, power costs clock, and
clock costs throughput.** On MI300X, memory reduction was worth only what it saved
directly; here a large enough reduction also buys back clock.

The qualifier matters. This shows up at the ablation's 24x traffic cut; it does *not* show
up at the scale of the real change below. The jmp1-into-LDS change removes 14% of the
bytes, and measured 16 W lower at a clock 33 MHz *lower* than the baseline, not higher -
LDS accesses cost power too, and a 1.5% clock difference is inside what the governor moves
on its own. Its +2.1% is the direct traffic saving, not a clock effect. Treat the clock
feedback as a reason to pursue *large* memory reductions, not as a multiplier on small
ones.

## Cost model

Ablations on one MI355X, each keeping one of instruction count / memory traffic fixed and
moving the other. `m1`-`m3` delete 1, 2 and 3 modular multiplies from the point addition
while leaving every load, store and loop identical.

| Variant | KernelA instructions | MKeys/s | vs baseline |
|---|---|---|---|
| baseline (as shipped) | 3,087 | 15,871 | - |
| state forced cache-resident | 3,091 | 18,644 | +17.5% |
| `m1` (-1 multiply) | 2,799 | 17,488 | +10.2% |
| `m2` (-2 multiplies) | 2,553 | 18,960 | +19.5% |
| `m3` (-3 multiplies)* | 2,309 | 21,596 | +36.1% |

\* `m3` is the one row carried over from the earlier, contaminated pass; every other row
was re-measured on a quiet machine. The rows that were measured both ways agreed to within
0.2%, so `m3` is very likely sound, but it is flagged rather than trusted.

The same ablation on MI300X gave **+42%** for cache residency, against **+17%** here.
MI355X's 8 TB/s of HBM3E has absorbed most of the memory bottleneck that dominated CDNA3.

The multiply-deletion rows fit a simple additive model - a compute term proportional to
instruction count plus a fixed memory term:

```
runtime  =  0.851 * (instructions / 3087)  +  0.149        (of baseline runtime)
```

Both coefficients come from the baseline and the cache-resident row alone. With no further
tuning it then predicts the near rows inside the noise floor, and under-predicts the far
one:

| | predicted | measured | error |
|---|---|---|---|
| `m1` (-9% instructions) | 17,240 | 17,488 | +1.4% |
| `m2` (-17%) | 18,611 | 18,960 | +1.9% |
| `m3` (-25%) | 20,209 | 21,596 | **+6.9%** |

**Consequence: within a few percent of the current operating point, 1% fewer KernelA
instructions buys 0.85% throughput.** That is the number any instruction-level proposal
should be multiplied by, and it is the relevant range for every proposal below.

The `m3` overshoot is the power coupling from the previous section reappearing: a 25% cut
in instructions lowers power enough that the clock rises, so large reductions return
*better* than linearly. That effect is not reachable by shaving a few percent.
(On MI300X the same experiment converted at roughly 1.0 throughout; the near-field
conversion is slightly worse here because the fixed memory term is a larger share of what
is left.)

### Where the time goes

Three throughput measurements are enough to split the runtime, with no instruction counting
involved: the achieved rate, the cache-resident ablation (everything but DRAM), and
`bench/arith_rate.hip` (the per-point-addition primitive sequence run purely in registers,
so no memory and no kernel structure at all).

| | MI300X | MI355X |
|---|---|---|
| Achieved | 10.0 GH/s | 15.9 GH/s |
| Ceiling with DRAM traffic removed | 14.2 | 18.6 |
| Ceiling with arithmetic only | 22.9 | 27.2 |
| → Field arithmetic | 44% | **58%** |
| → DRAM traffic | 30% | **15%** |
| → Kernel structure (addressing, control, DP, jump list, issue loss) | 27% | 27% |

Kernel structure costs the same 27% on both, and it is the memory share that collapsed.

**The field arithmetic has no headroom left.** `arith_rate` reaches 27.22 G point-addition
sequences/s on MI355X at 1,456 VALU per sequence, which is 39.6 Tops/s against a 39.3
Tops/s nominal peak - **101%**. The primitives are at the hardware issue limit; an ASM-level
win has to remove instructions, because it cannot issue the existing ones any faster.

## ASM level: what is actually left

Three facts bound every instruction-level proposal on this machine.

**1. The primitives are at the issue limit.** `arith_rate` runs them at 101% of nominal VALU
peak. There is no scheduling win available - no bubble to fill, no latency to hide. An
ASM-level change has to *remove* instructions; it cannot issue the existing ones faster.

**2. The multiply is at the ISA floor.** An 8x8 32-bit-limb schoolbook product needs 64
partial products, and the port already emits exactly 64 `v_mad_u64_u32`. Each is followed
by one `v_addc_co_u32` to fold the carry the MAD computed and discarded, because CDNA has
no multiply-add with carry-in - and **gfx950 still does not**, which is the single largest
thing CDNA4 could have changed for this workload and did not.

**3. The column bookkeeping is forced.** `v_mad_u64_u32` requires an even-aligned VGPR pair
for its 64-bit addend, so advancing a 32-bit-granular sliding accumulator costs about two
moves per column no matter how it is arranged. `CDNA_PHASE2_DESIGN.md` measured the
compiler at ~40 moves per multiply where a naive hand-written version needs ~60.

Alternative formulations, and why each loses on CDNA:

| Approach | Verdict |
|---|---|
| **Karatsuba** | Trades multiplies for carry-propagating adds. On CDNA the MAD *fuses* the accumulate, so adds are relatively more expensive than on Ada - the same reason `SqrModP` profitably forwards to `MulModP` here. 3 x 128-bit products is ~96 instructions against 128, but the recombination is two 256-bit add/sub chains plus the operand sums, which gives it back. |
| **24-bit limbs to eliminate carry folds** | 11 limbs, 121 MADs, and a column sum of 11 x 2^48 cannot overflow a 64-bit accumulator, so the 64 `v_addc_co_u32` disappear. But the 24-bit column shift is no longer a free word offset (~50 ops), the operands must be split from and rejoined to 64-bit words (~66 ops), and the Solinas reduction assumes 32-bit limbs. ~237 against ~173. |
| **8-bit limbs via `v_dot4_i32_i8`** | **Measured 0.29x.** 8-bit decomposition costs 16x the products while `v_dot4` returns only 4x the rate. |
| **Matrix cores** | Measured 2.85x raw, structurally unusable - see below. |

So the arithmetic is finished. What is left at the instruction level is the *kernel glue*:
`CDNA_PHASE4_ASSESSMENT.md` measured 79 removable moves out of 1,971 VALU per point
addition, or 4%. At this machine's 0.85 conversion that is **+3.4%**, in exchange for
hand-writing ~2,500 lines of assembly that no differential test can cover. That trade did
not pay on MI300X and it does not pay here.

**The ASM-adjacent lever that does look worth it on MI355X is the memory instructions, not
the arithmetic ones** - specifically the cache-policy bits, because of the power coupling
above. That is the non-temporal experiment in the next-steps list.

## Matrix cores do not help, and this is measurable

CDNA4's headline feature is low-precision matrix throughput, so the tempting idea is to
build the 256x256 multiply out of `v_mfma_i32_*_i8`. The unit of comparison is the 32x32 ->
64 multiply-accumulate, because an 8x8 limb schoolbook product needs 64 of them, and one
32x32 product decomposes into 16 8x8 products - so a matrix core has to be more than 16x
faster per MAC just to break even. `bench/mfma_rate.hip`:

| sequence | int8 MAC/s | 32x32 MAC-equivalents/s | vs `v_mad_u64_u32` |
|---|---|---|---|
| `v_mad_u64_u32` (baseline) | - | 27.0 T | 1.00x |
| `v_mfma_i32_16x16x32_i8` | 1060 Tops/s | 66.3 T | 2.46x |
| `v_mfma_i32_32x32x16_i8` | 1228 Tops/s | 76.8 T | **2.85x** |
| `v_dot4_i32_i8` | 123 Tops/s | 7.7 T | 0.29x |

So there is a genuine 2.85x of raw headroom, and `v_dot4_i32_i8` is a clear loss. But the
raw rate is not the binding constraint - **the algorithm's multiplies are rank-1**. MFMA
computes `D[M,N] += A[M,K] * B[K,N]`; a big-integer product is an outer product, so either

- the Toeplitz form uses `M` for output limbs and `K` for the operand limbs, leaving `N=1`
  of 32 columns - unless 32 multiplies share one multiplicand, which KernelA's
  `MulModP(inverse, inverse, tmp)` chain does not; or
- the direct `a_i * b_j` grid form uses `M` and `N` for the limb indices, leaving `K=1` of
  16.

Either way 1/32 or 1/16 of the machine is used, which turns 2.85x into roughly 0.09-0.18x,
before paying for the cross-lane transposes and for carry-propagating 64 int32 columns back
into a 512-bit integer. **Not viable for this algorithm.** The finding is about the
algorithm's shape, not about CDNA4's matrix cores, which are fast.

## Changes made

### 1. jmp1's y-coordinates into LDS (CDNA4 only): +2.0%

CDNA3 has 64 KB of LDS per CU, so at 3 workgroups/CU only jmp1's x half (16 KB) fits and
y is re-read from global on every point addition. CDNA4 has **160 KB**, so 3 x 32 KB fits
and y comes along, removing 32 bytes of DRAM traffic per point addition - about 13% of the
kernel's bytes. Occupancy is unaffected: 147 VGPRs cap KernelA at 3 waves/SIMD long before
96 KB of 160 KB does, and the register count, spill count and occupancy are all unchanged.

Back-to-back on one GPU, with six unrelated variants measured in between so the two A/B
pairs bracket the whole session:

| | MKeys/s |
|---|---|
| A baseline | 15,601 |
| B jmp1_y in LDS | 15,867 |
| ... six other variants ... | |
| A baseline | 15,491 |
| B jmp1_y in LDS | 15,874 |

**+2.1%** on the means, with the two A runs 0.71% apart, the two B runs **0.04%** apart,
and both B runs above both A runs. Selected by
`__gfx950__` in `src/hip/RCGpuCore.hip`; gfx942 codegen is untouched. `JMP1_LDS_STRIDE_CDNA4`
in `defs.h` is shared with the host so `GpuKang.cpp` sizes the dynamic LDS request from the
same constant.

This was listed as "small and unmeasured" in `CDNA_PHASE4_ASSESSMENT.md`, where it cost a
wave of occupancy on CDNA3. On CDNA4 it costs nothing.

### 2. Thread-local RNG: 168x on the node's startup phase

`EcInt::RndBits` took a **global mutex per 64 bits drawn**, and `GenerateRndDistances`
draws one distance per kangaroo - 6.3 million per GPU at the CDNA defaults, so ~50 million
contended acquisitions across an 8-GPU node. The eight worker threads spin at 95% CPU for
minutes before the first jump runs.

Each thread now takes its own `mt19937_64`, seeded from the global seed mixed through a
SplitMix64 finalizer with a claim counter so the streams stay distinct - default-
constructing per thread would instead give every GPU the identical set of kangaroo
distances, which the benchmark below checks for.

Eight threads x 500,000 draws, same code path, no GPU involved:

| | wall | draws/s |
|---|---|---|
| global mutex | 2,613 ms | 1.53 M/s |
| thread-local | **16 ms** | **258 M/s** |

This is a startup-latency fix, not a throughput one - it does not touch the steady-state
jump rate.

## Memory-side experiments that failed

Two follow-ups to the LDS change, both measured back to back on one GPU against the shipped
build, and both rejected.

| Variant | MKeys/s | vs shipped |
|---|---|---|
| shipped (LDS stride 8 u64) | 15,871 | - |
| LDS stride 10 u64 (40 KB/block) | 15,904 | +0.2% (noise) |
| non-temporal kangaroo x/y | **15,615** | **-1.6%** |

**LDS bank padding does nothing.** Packing x and y into one 64-byte entry makes consecutive
jump-table entries start in only 4 of CDNA4's 64 banks, which looks like a 16-way conflict.
Padding the stride to 80 bytes spreads them over 16 banks and changes nothing measurable
(+0.2%, inside the noise floor). The divergent gather is simply not the constraint -
Phase 0 measured ~12x of headroom in it. Stride 8 stays, since it is the smaller footprint.
A 96-byte stride measured -1.1% in the earlier contaminated pass and was not re-run.

**Non-temporal kangaroo state is actively harmful, for an instructive reason.** The idea was
that KernelA's three 201 MB planes - kangaroo x, kangaroo y, prefix products - compete for
the 256 MB MALL, and that the prefix plane is the one with real reuse (written in the
forward pass, read back in the backward pass of the same step). Marking the x/y accesses
`nt` to protect it costs 1.6%.

The premise was wrong: **the x plane is read twice per step too** - once in the forward pass
building the inverse product (`RCGpuCore.hip:231`) and once in the backward pass applying
the jump (`:249`). It has the same forward-to-backward reuse the prefix plane does, and
marking it "miss evict" throws that away. This is also part of why the memory share is only
15%: two of the three planes are already getting cache hits.

The prefix plane still cannot go in LDS either - 256 threads x 32 groups x 32 B is 256 KB
per workgroup against 160 KB available - so CDNA4's larger LDS does not reopen that door.

## Ranked list of what is left

| | expected | confidence | cost |
|---|---|---|---|
| **CPX compute partitioning** | +5.5% | high - measured on MI300X, untested here | needs root: `rocm-smi --setcomputepartition CPX --setmemorypartition NPS4` |
| Hand-written KernelA (Phase 4) | ~+3.4% | low | ~2,500 lines of unschedulable-by-test assembly |
| Larger `JMP_CNT` using spare LDS | ~0 | low | would drop to 2 workgroups/CU, measured worse |

**CPX is the best remaining lever** and it is a configuration change, not code. It was worth
+5.5% on MI300X and scaled linearly to 64 logical devices (`CDNA_CPX_PARTITIONING.md`). The
node here is in SPX/NPS1. Note `-gpu` only parses single digits, so it cannot address the
64 logical devices CPX exposes - that needs fixing first if partitions must be selected
individually.

**Phase 4 stays closed.** `CDNA_PHASE4_ASSESSMENT.md` measured the removable kernel glue at
79 moves out of 1,971 VALU per point addition, or 4%. At this machine's 0.85 conversion
that is +3.4%, for a rewrite with no differential test able to cover it. The reason Phase 4
does not get *more* attractive on MI355X despite memory mattering less is that the
arithmetic is already at 101% of VALU peak and its remaining register moves are required by
`v_mad_u64_u32`'s even-aligned operand pair.

## Reproducing

```sh
make -f Makefile.hip OFFLOAD_ARCH=gfx950 -j16
make -f Makefile.hip OFFLOAD_ARCH=gfx950 test     # 10M-vector differential test
make -f Makefile.hip OFFLOAD_ARCH=gfx950 bench    # instruction rates

./build-hip/rckangaroo-cdna -dp 30 -range 139            # node
./build-hip/rckangaroo-cdna -gpu 0 -dp 30 -range 139     # one GPU

# power and clock under load - the number that matters most on MI355X
rocm-smi --showgpuclocks --showpower

# before every timed run: exactly one process, on the one GPU under test
rocm-smi --showpids
```

Take the maximum over 13 or more speed reports, and re-measure the baseline in the same
session.
