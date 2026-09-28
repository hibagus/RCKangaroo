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
