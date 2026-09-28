# Phase 2: where MulModP's instructions actually go

Phase 1 left two gaps of similar size: 2,967 VALU instructions per point-addition
(~2x RC's hand-written SASS) and 53% VALU utilisation. The occupancy work addressed the
second. This is the analysis behind attacking the first.

`MulModP` is called 5 times per point-addition and accounts for 1,845 of the 2,967 VALU
instructions - 62% of the total - so it is the only target worth starting with.

## Breakdown of the 369 instructions

From the generated ISA for gfx942:

| | Count | Theoretical min | Assessment |
|---|---|---|---|
| `v_mad_u64_u32` | 84 | 84 | **already optimal** |
| carry ops (`v_add_co`/`v_addc_co`) | 138 | ~128 | near optimal |
| `v_mov_b32` | **112** | ~0 | pure register shuffling |
| `v_lshl_add_u64` | 25 | - | address arithmetic |
| `v_cndmask_b32` | 5 | - | the third reduction fold |
| `s_nop` | **86** | ~0 | hazard wait states |

The multiply core needs no work: 64 MADs for the 8x8 32-bit-limb grid plus 20 for the
reduction is the floor, and the compiler already hits it. The waste is 112 moves and 86
NOPs, together about 40% of the issue slots.

## The NOPs are a consequence of the moves

They appear wherever a non-carry instruction gets scheduled *inside* a carry chain:

```
v_addc_co_u32 v23, vcc, 0, v24, vcc   <- writes VCC
v_mov_b32     v16, v26                <- interposed
s_nop 0                               <- forced wait
v_addc_co_u32 v30, vcc, 0, v25, vcc   <- reads VCC as carry-in
```

Phase 0 measured a clean serial carry chain at 0.958 of peak with no NOPs, and the CDNA3
ISA hazard table gives 0 wait states for "VALU writes VCC -> VALU reads VCC as carry-in".
So these are not a hardware requirement - they are the cost of the compiler interleaving
register moves into the chains. Removing the moves removes the NOPs.

## Why C++ cannot fix this, measured

The obvious restructuring is to switch from operand scanning (row-wise, what the reference
uses) to column scanning (Comba), where the accumulator is the MAD's own addend so a
product costs one MAD plus one carry-accumulate rather than a MAD plus two adds. Both
written in C++, for the 256x256->512 multiply:

| Form | VALU | mad | carry | v_mov | s_nop | issue slots |
|---|---|---|---|---|---|---|
| operand scanning (current) | 264 | 64 | 96 | 82 | 63 | **327** |
| Comba | 255 | 64 | 149 | **2** | 104 | 359 |

Comba does eliminate the moves - 82 down to 2 - but pays for it in carry ops, 96 up to
149, and comes out slightly worse overall.

The reason is exact and is the whole argument for assembly here. The hardware computes
`{carry, acc64} = a.u32 * b.u32 + acc64` in one instruction with the carry delivered free
into an SGPR. **C++ has no way to consume a multiply-accumulate's carry-out**, so the
compiler has to recover it with `(s < acc) ? 1 : 0`, which costs more than the carry the
hardware already produced and threw away.

## Validated design

Two instructions per product, with the carry consumed directly:

```
v_mad_u64_u32 v[60:61], vcc, a_i, b_j, v[60:61]   // acc += a[i]*b[j]
v_addc_co_u32 v62,      vcc, 0,   v62,  vcc       // fold carry into the third word
```

Compiled as inline asm this emits exactly those two instructions and **zero NOPs**,
confirming the hazard waits were scheduling artifacts.

Three feasibility questions, all resolved by measurement:

- **Operand count.** 32-bit limbs need 16 inputs and 16 outputs. Inline asm accepts at
  least 40 operands, so this is not a constraint.
- **Register pairing.** `v_mad_u64_u32` needs an aligned consecutive VGPR pair for its
  64-bit addend, which separate `u32` operands cannot guarantee. Solved with an explicit
  VGPR window inside the asm block, declared in the clobber list; verified that the
  allocator keeps clear of it (`vgpr_count` rose to exactly the reserved bound) and that
  the 32-bit halves become individually addressable, which C++ operands do not allow.
- **Column shifts become free.** In hand-written asm, advancing the accumulator to the
  next column is a change of *register name*, not a data movement. This is the specific
  thing the compiler cannot do and where the 112 moves go.

Projected: 64 products x 2 instructions + ~16 result extractions + reduction ~40 = ~150
instructions against the current 327 issue slots, i.e. ~2.2x on the multiply.

Per point-addition that takes 2,967 VALU toward roughly 1,900, which at the current
utilisation would put a single MI300X near 12-13 GH/s.

## Correctness plan

The differential harness already exists and is the reason this is a tractable change: 8
primitives against Python arbitrary-precision ground truth over 10,000,000 vectors,
leading with the full edge-case cross product (0, 1, p-1, p, 2^256-1, values near 2^256
that stress the Solinas fold). Every asm primitive gets validated against the C++ version
it replaces before it goes anywhere near the hot loop, and `tests/isa_quality.sh` guards
the MAD count so a codegen regression cannot masquerade as an algorithmic one.

---

# Phase 2 results

Measured on MI300X, ROCm 10, using puzzle #140 (139-bit range) as the benchmark
configuration: K = 1.15 with 0% DP overhead, which is the regime a real solve runs in.

## Primitive costs

Issue slots, i.e. VALU instructions plus the `s_nop` wait states they force:

| | before | after | |
|---|---|---|---|
| `mul_256_to_512` | 264 VALU + 63 nop = 327 | 171 + 8 = **179** | 1.83x |
| `MulModP` | 369 + 86 = 455 | 271 + 31 = **302** | 1.51x |
| `SqrModP` | 307 + 32 = 339 | 269 + 32 = **301** | 1.13x |
| `SubModP` | 49 + 9 = 58 | 30 + 0 = **30** | 1.93x |

The multiply's 64 MADs and 64 carry ops are both exactly optimal - one
carry-accumulate per product, nothing spare.

## End to end

| | VALU / point-add | VALU utilisation | per GPU | 8 GPUs |
|---|---|---|---|---|
| Phase 1 | 2,967.6 | 53.3% | 6.25 GH/s | 49.2 GH/s |
| + occupancy | 2,967.6 | - | 8.1 GH/s | 64.5 GH/s |
| + asm multiply & subtract | **2,290.9** | **62.7%** | **9.24 GH/s** | **73.9 GH/s** |

Instruction count fell 1.30x and utilisation rose from 53% to 63%. Throughput gained
1.15x from the assembly on top of 1.30x from the occupancy work.

Reference: RTX 4090 = 14.5 GH/s, RTX 5090 = 19.3 GH/s. The node is at 5.1x a single 4090
and 3.8x a single 5090.

## What did not pay off, and why

**A dedicated squaring is not worth writing.** The symmetry saving relies on doubling each
off-diagonal product, and `v_mad_u64_u32` cannot double - adding the product twice costs
two MADs, exactly what computing both halves of the general multiply costs. The two-pass
alternative (accumulate 28 off-diagonal products, double the whole 512-bit result, then add
8 diagonal squares) works out to roughly 156 instructions against 171, about 9%, for real
added complexity and carry-handling risk. Skipped.

Note `SqrModP` gained only 1.13x, versus 1.51x for `MulModP`. Routing it through the asm
multiply cost it the CSE the compiler had been doing on duplicate partial products - it was
15% cheaper than `MulModP` in C++ and is now the same. The asm win still dominates.

**`SubModP` gained less end to end than its slot count implied.** It is 7 calls per
point-addition and its 28-slot saving is 8.7% of the primitive budget, which at the
observed instruction-to-throughput ratio should have been worth ~4%. Measured: +0.5%. No
register regression caused it (KernelA is 152 VGPRs, no spills), so the arithmetic is no
longer the constraint at the margin.

## Where the remaining overhead is

KernelA's static body - roughly one group iteration - still contains 475 `v_mov_b32` and
408 `s_nop` against 621 `v_mad_u64_u32`. That overhead is no longer *inside* the
primitives; it is the glue between them. Each inline-asm block is an opaque scheduling
barrier, so the compiler must materialise 256-bit operands into and out of registers
around every call instead of keeping them in place.

This is the structural limit of per-primitive assembly, and it is exactly what RC's
`fuse.asm` addresses on the NVIDIA side: `CalcToInv_FusedA` interleaves SubMod, MulMod and
the copies into a single schedule rather than calling them in sequence. Fusing the
batched-inversion inner sequence into composite asm blocks is the next step, and the
remaining gap - 2,291 instructions against RC's ~1,420 - is about the right size for it.
