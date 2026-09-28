# Phase 1 results: working HIP port on MI300X

Measured 2026-09-28 on one AMD Instinct MI300X (gfx942, 304 CU, 2.1 GHz), ROCm 10.
Build: `make -f Makefile.hip`.

## It works

```
GPU 0: AMD Instinct MI300X, 191.98 GB, 304 CUs, gfx942:sramecc+:xnack-, PCI 27
GPU 0: CDNA3 (MI300-series) detected
GPU 0: allocated 5592 MB, 1867776 kangaroos.
BENCH: Speed: 6227 MKeys/s, Err: 0, DPs: 883K/6106K
Point solved, K: 1.006 (with DP and GPU overheads)
```

| Metric | Value |
|---|---|
| Speed, single MI300X | **6.1-6.3 GH/s** (stable across runs) |
| K coefficient, 76-bit benchmark | **1.006** |
| Kangaroos | 1,867,776 (256 threads x 24 groups x 304 CU) |
| Device memory | 5,592 MB |
| Errors | 0 |

**K is the correctness signal that matters.** A solver can produce right answers while
silently wasting most of its work: if the SOTA loop handling were broken, kangaroos
would get stuck cycling and K would inflate far above RC's documented 1.15. Landing at
1.006 on a 76-bit solve says the loop detection, the jmp2 escape path, and KernelC's
rewind are all doing their jobs. (1.006 is a single-point draw; the long-run average
converges to ~1.15.)

Also verified: all 8 field-arithmetic primitives pass 10,000,000 differential vectors
(`make -f Makefile.hip test`), and the codegen gate holds (`make -f Makefile.hip isa`).

## Per-kernel resources

| Kernel | VGPRs | AGPRs | Scratch | LDS | Spills |
|---|---|---|---|---|---|
| KernelA | 154 | 0 | 0 | 64 KB (dynamic) | 0 |
| KernelB | 88 | 0 | 0 | 48 KB | 0 |
| KernelC | 174 | 0 | 0 | 48 KB | 0 |
| KernelGen | 185 | 0 | 48 B | 0 | 16 SGPR |

`KernelGen` runs once at startup, so its spills do not matter.

## Where the time goes

From `rocprofv3` hardware counters over 46 KernelA dispatches (1,216 wave64 each,
1.868e9 point-additions per dispatch, 308 ms per dispatch):

| Per point-addition, per lane | Count |
|---|---|
| VALU instructions | **2,967.6** |
| SALU instructions | 148.3 |
| VMEM instructions | 15.0 |
| LDS instructions | **6.0** |

| Throughput | Value |
|---|---|
| Achieved VALU rate | 17.98 Tops/s |
| vs measured achievable peak (33.76 Tops/s) | **53.3%** |
| vs nominal peak (40.86 Tops/s) | 44.0% |

The LDS figure lands exactly on the 6 `ds_read_b128` per point-addition predicted in
Phase 0, which also confirms the jump-table gather is nowhere near being a bottleneck:
Phase 0 measured 0.98 Tops/s for that access pattern, giving ~163 G point-additions/s of
LDS headroom against a ~6 G/s delivered rate.

### Two independent gaps, roughly equal in size

**1. Instruction count.** 2,967 VALU per point-addition is about 2x what RC's
hand-written SASS needs. Working backwards from his published 14.5 GH/s on a 4090
(~20.6 T IMAD/s on Ada) gives ~1,420 slots per point-addition. The static counts for
this port explain most of ours: 5 x MulModP (369) + 1 x SqrModP (307) + 7 x SubModP (49)
+ InvModP/24 (33) = 2,528, with the remaining ~440 being loop overhead, address
arithmetic, the DP-table check, jump-list packing and the LastPnts tail stores.

This is a larger gap than Phase 0 suggested from the multiply alone (~1.5x). The extra
comes from things the compiler will not do: RC fuses SubMod into MulMod as one
interleaved schedule (`fuse.asm`) and leaves no register-shuffling `v_mov` behind.

**2. Utilization.** 53% of achievable VALU throughput, with 15 VMEM + 6 LDS accesses per
point-addition to hide. The cause is structural: **KernelA runs at 1 wave/SIMD, so there
is no second wave to switch to while one waits on memory.** Every VMEM latency is a full
stall. RC accepts the same constraint on NVIDIA and compensates by hand-scheduling the
overlap; we currently rely on the compiler's scheduler.

Raising occupancy to 2 waves/SIMD needs both:
- **VGPRs <= 128** (currently 154). The AccVGPRs are the tool here - Phase 0 measured
  `v_accvgpr_read/write` at full rate, so moving InvModP's live state there costs one
  instruction per access instead of a scratch round trip.
- **LDS <= 32 KB per workgroup** (currently 64 KB, holding both jump tables).
  - On **CDNA3** this means halving `JMP_CNT` to 256, giving 2 x 16 KB. The cost is 2x
    more level-1 size-2 loops (1/512 rather than 1/1024 of jumps), and loop handling is
    cheap - this is the tradeoff flagged in the port plan, now with a concrete reason to
    take it.
  - On **CDNA4** it is free: 160 KB LDS holds two 64 KB workgroups, so `JMP_CNT` stays at
    512. This is the clearest CDNA4 advantage found so far.

### What each fix is worth

Both gaps are multiplicative, and neither has been attempted yet:

| Change | Projected |
|---|---|
| Phase 1 (measured) | 6.25 GH/s |
| Phase 2: instruction count 2,967 -> ~1,700 at unchanged 53% | ~10.9 GH/s |
| Phase 3: occupancy 2 waves/SIMD, ~73% at unchanged instruction count | ~8.6 GH/s |
| Both | ~15 GH/s |

Reference: RTX 4090 = 14.5 GH/s, RTX 5090 = 19.3 GH/s.

## Honest position against the Phase 0 projection

Phase 0 projected 7.0-10.5 GH/s for Phase 1; the actual is 6.25, slightly under the low
end. The efficiency assumption was right (45-65% predicted, 53% measured) - the
instruction count was wrong. Phase 0 estimated ~1,980 VALU per point-addition from
pre-optimisation primitive costs; the real figure is 2,967. The ceiling implied by the
measured count is 33.76e12 / 2,967 = 11.4 G point-additions/s, and 6.25 is 55% of that.

For context, the sibling `/home/bagus/AMDKangaroo` port reports 1.3 GH/s on a 7900 XTX
and its docs target 1.9-2.1 GH/s for MI300X. This port is at 6.25 GH/s with no assembly
written yet.

## Reproducing

```sh
make -f Makefile.hip                    # build for gfx942
make -f Makefile.hip test               # 10M-vector arithmetic differential test
make -f Makefile.hip isa                # static codegen gate
make -f Makefile.hip bench              # Phase 0 instruction-rate benchmark

# speed and K on one GPU
./build-hip/rckangaroo-cdna -dp 16 -range 76 -gpu 0

# profile KernelA. Tames-generation mode is used because it exits cleanly -
# benchmark mode loops forever and rocprofv3 only flushes counters at exit.
rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_LDS SQ_INSTS_SALU SQ_WAVES \
  --kernel-include-regex KernelA -d prof -o prof --output-format csv \
  -- ./build-hip/rckangaroo-cdna -dp 16 -range 70 -tames /tmp/t70.dat -max 1 -gpu 0
```
