# Phase 0: Measured CDNA3 instruction rates

Measured on this machine, 2026-09-28. Hardware: 8x AMD Instinct MI300X
(`gfx942:sramecc+:xnack-`), 304 CU, 4 SIMD/CU, wave64, 2100 MHz, ROCm 10
(HIP 7.15.26333, AMD clang 23.0.0git).

Benchmark: [`bench/isa_rate.hip`](../bench/isa_rate.hip). Every sequence is emitted as
inline asm and the generated ISA is verified by disassembly, so the measured instruction
mix is exactly the intended one. Each test runs `1 workgroup of 256 threads per CU`
(4 wave64 = 1 wave/SIMD), matching how RCKangaroo's `KernelA` is actually dispatched.
Best of 3 runs, ~8 ms each.

Nominal simple-VALU peak: 304 CU x 64 lanes x 2.1 GHz = **40.86 Tops/s**.

## Results

| Instruction sequence | Tops/s | % of peak | vs `v_add_u32` | Rate class |
|---|---|---|---|---|
| `v_add_u32`, 8 independent (reference) | 33.76 | 82.6% | 1.000 | full |
| `v_mad_u64_u32`, 8 independent | 27.59 | 67.5% | 0.817 | full |
| `v_mad_u64_u32`, **1 serial chain** | 28.98 | 70.9% | 0.859 | full |
| `v_add_co_u32`, 8 independent, distinct SGPR carry | 29.11 | 71.3% | 0.862 | full |
| `v_addc_co_u32`, **1 serial VCC chain** | 32.35 | 79.2% | 0.958 | full |
| `v_accvgpr_write_b32` / `v_accvgpr_read_b32` | 32.43 | 79.4% | 0.961 | full |
| `ds_read_b128`, divergent gather over 32 KB | 0.98 | 2.4% | 0.029 | conflict-bound |

Reproduced on GPU 3 within 2% on every row.

## What this establishes

### 1. `v_mad_u64_u32` is a full-rate instruction

This was the single largest unknown going into the port: the CDNA3/CDNA4 ISA manuals
document instruction semantics but publish no rates, and the projected solver throughput
differed by ~2x between the quarter-rate and full-rate cases.

It is full rate. At 0.82-0.86x the throughput of a plain `v_add_u32`, the small deficit is
operand-gather and VGPR pressure (it reads 3 operands and writes a 64-bit pair plus a
carry), not a rate divider. MI300X's full-rate FP64 datapath evidently comes with a
full-width 32-bit integer multiplier.

Consequence: the 64 `v_mad_u64_u32` that an 8x8 32-bit-limb schoolbook multiply needs cost
roughly what 64 adds cost. The multiply core is not a handicap relative to NVIDIA's
`IMAD.WIDE.U32`.

### 2. Serial dependency chains are free - which inverts the NVIDIA design guidance

The serial variants are **faster** than the 8-independent ones:

- `v_mad_u64_u32`: serial 28.98 vs independent 27.59 Tops/s
- `v_addc_co_u32` serial VCC chain reaches 0.958 - the highest multi-operand result measured

A wave64 instruction occupies its SIMD16 for 4 cycles, which fully covers the VALU
dependent-issue latency, so a back-to-back dependent chain issues without a bubble. This
confirms the CDNA3 ISA hazard table (section 4.5): *"VALU writes SGPR/VCC -> VALU reads
SGPR as carry-in: 0 wait states."* There is also no penalty for serializing through VCC
specifically.

This matters a lot for the port. RC's hand-written SASS spends much of its complexity
interleaving multiple independent carry chains through separate predicate registers
(`Pt0..Pt4` in `mod_mul.asm`), because on NVIDIA a serial chain stalls. **On CDNA that
work is unnecessary, and slightly counterproductive** - more parallel chains means more
live registers for no throughput gain. Prefer fewer, longer chains.

Practical effect: the ordinary C++ carry chains measured at 234 VALU ops per 256x256->512
multiply should run at close to their instruction-count-implied speed, so Phase 1 is not
leaving latency on the table, only instruction count.

### 3. AccVGPRs are full-rate scratch

`v_accvgpr_write`/`read` run at 0.961. Since AGPRs are not valid VALU operands on
gfx942/gfx950, their only use is as software-managed scratch - and at 1 full-rate
instruction per access, 256 AGPRs give 1 KB/lane of spill space far cheaper than a scratch
memory round trip. This is the intended home for `InvModP`'s ~78 live u32 of state and
`SqrModP`'s 28-entry cross-product array. NVIDIA has no equivalent resource.

### 4. The divergent LDS gather is slow, but not the bottleneck

A fully divergent wave64 `ds_read_b128` over a 32 KB table (the `jmp1_table` access
pattern) achieves 0.98 Tops/s - 34x slower than VALU, as expected when 64 lanes x 4 dwords
land randomly across 32 banks.

It still has headroom. `KernelA` performs roughly 3 `Copy_int4_x2` per point-addition
= ~6 `ds_read_b128`, giving 0.98e12 / 6 = **163 G point-adds/s** of LDS capability against
a VALU ceiling near 16 G/s: about 8x margin. So the jump-table gather does not gate Phase 1,
and the `JMP_CNT` reduction contemplated in Phase 3 is not needed for this reason.

Worth re-checking if `PNT_GROUP_CNT` grows or the tables move, and note CDNA4 doubles LDS
banks (64 vs 32), which should roughly halve this conflict cost.

## Revised performance model

Instruction budget per elliptic-curve point addition, from the static census of
`RCGpuCore.cu`'s `KernelA` (4.875 `MulModP` + 1 `SqrModP` + ~7 `SubModP` + 1/24 `InvModP`
per point-add) at the Phase 1 codegen quality measured earlier (234 VALU per 512-bit
product, ~294 per full `MulModP` including reduction):

- ~1,980 VALU ops per point-add
- Mix is ~28% MAD (0.82 relative) and ~72% add (1.00 relative) -> effective **31.8 Tops/s**
- VALU-only ceiling: 31.8e12 / 1980 = **16.1 G point-adds/s**

The real kernel adds global loads/stores of kangaroo state, the LDS gather, DP-table
atomics and branches on top of pure VALU work, so expect 45-65% of that ceiling.

| Stage | Per MI300X | 8 GPUs | Basis |
|---|---|---|---|
| Phase 1 (HIP C++) | 7.0-10.5 GH/s | 56-84 GH/s | 1,980 VALU/point-add at measured rates |
| Phase 2 (targeted asm) | 10-14 GH/s | 80-112 GH/s | ~1.35x from 234 -> ~160 VALU per multiply |

Reference points: RTX 4090 = 14.5 GH/s, RTX 5090 = 19.3 GH/s (RC's published figures).
So a single MI300X should reach 4090 class after Phase 2, and this 8-GPU node should
deliver roughly 5-8x a single 4090.

These supersede the projections in `/home/bagus/AMDKangaroo/docs/`, which had no
measurements behind them.

## Reproducing

```sh
hipcc -O3 --offload-arch=gfx942 -o isa_rate bench/isa_rate.hip
./isa_rate 0          # device index

# verify the emitted instruction mix matches intent
hipcc -O3 --offload-arch=gfx942 --offload-device-only -S -o - bench/isa_rate.hip \
  | awk '/^_Z7mad_thr/{f=1} f&&/s_endpgm/{exit} f' | grep -oE 'v_[a-z0-9_]+' | sort | uniq -c
```
