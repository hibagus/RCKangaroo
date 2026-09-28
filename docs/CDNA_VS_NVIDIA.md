# Why RTX 4090/5090 are faster per GPU than MI300X on this workload

RetiredCoder's published figures are 14.5 GH/s on an RTX 4090 and 19.3 GH/s on an RTX
5090. This port currently reaches 10.1 GH/s on an MI300X, or 81.0 GH/s across a node of
eight.

The per-GPU gap has one dominant cause, and it is not the one most people expect. It is
not ALU count and it is not cache size - MI300X has more of both. **It is a single
instruction-set feature: NVIDIA's wide integer multiply-add accepts a carry-in, and AMD's
does not.**

Every figure below is labelled as measured on this machine, published by RC, or derived.
The measurement harnesses are `bench/isa_rate.hip` and `rocprofv3`; see
`docs/CDNA_PHASE0_MEASUREMENTS.md` and `docs/CDNA_PHASE1_RESULTS.md`.

## It is not ALU throughput

This workload is integer-multiply bound - roughly 500 32x32 multiplies per elliptic-curve
point addition. So the relevant number is integer multiply-accumulate throughput, not
FP32 cores.

| | Integer MAD throughput | Source |
|---|---|---|
| RTX 4090 | 128 SM x 64 INT32/clk x 2.52 GHz = **20.6 Tops/s** | derived from spec |
| MI300X | **27.5 Tops/s** (`v_mad_u64_u32`) | **measured**, `bench/isa_rate.hip` |

MI300X has about **33% more** integer multiply throughput than a 4090, and it is full rate
relative to a plain add (0.82-0.86x), which was itself a measured finding rather than an
assumption - the CDNA ISA manuals publish no instruction rates.

For context on why FP32 core counts mislead here: the 4090's 16,384 "CUDA cores" are FP32
units, of which only 64 per SM can issue INT32. MI300X's 304 CUs x 64 lanes give 40.9
Tops/s of simple VALU work at 2.1 GHz.

## It is not SRAM

| | RTX 4090 | MI300X |
|---|---|---|
| L1 / vector cache | 128 KB per SM | 32 KB per CU |
| L2 | 72 MB | 4 MB per XCD, 32 MB across 8 XCDs |
| Last-level | - | **256 MB** Infinity Cache (MALL) |
| Scratchpad | shared memory, up to 100 KB/SM | 64 KB LDS per CU (CDNA3), 160 KB (CDNA4) |

MI300X has more total on-die memory. Note `rocminfo` reports the per-XCD 4 MB as "L2" and
the 256 MB MALL as "L3", which is a common source of confusion - the claim of "256 MB L2"
in some AMD ports of this solver conflates the two.

More importantly, **cache residency was measured not to matter for this workload.** RC
sizes his kangaroo count so the state array fits the 4090's 72 MB L2 - 786,432 kangaroos
x 96 B = 75 MB - which is exactly what his `cudaStreamSetAttribute` persistence window
does. The natural inference is that fitting MI300X's 256 MB MALL should win. It does not:

| PNT_GROUP_CNT | state array | fits 256 MB MALL? | MKeys/s (measured) |
|---|---|---|---|
| 12 | 171 MB | yes | 8363 |
| 24 | 342 MB | no | 9325 |
| 32 | 456 MB | no | **9444** |

The configurations that fit are the slow ones. Inverse amortisation and per-thread
instruction-level parallelism dominate cache residency here.

## It is the missing carry-in

NVIDIA's `IMAD.WIDE.U32.X` performs, in one instruction:

```
{carry_out, dst64} = a.u32 * b.u32 + src64 + carry_in
```

AMD's `v_mad_u64_u32` performs:

```
{carry_out, dst64} = a.u32 * b.u32 + src64            // no carry_in operand
```

So a partial product inside a carry chain costs AMD **two** instructions where NVIDIA
needs one:

```
v_mad_u64_u32 acc, vcc, a_i, b_j, acc     // product and accumulate
v_addc_co_u32 ext, vcc, 0, ext, vcc       // fold the carry - NVIDIA gets this free
```

Across the 8x8 32-bit limb grid of a 256-bit multiply that is 128 instructions instead of
64. This is not inference; it is visible in RC's own assembly, `mod_mul.asm`:

```
IMAD.WIDE.U32.X Ro4, Pt4, RFirst1.reuse, RSecond3, Ro4, Pt4
```

`Pt4` appears twice - once as the carry-in operand and once as the carry-out destination -
in a single instruction.

### The consequence, quantified

| | Issue slots per modular multiply | Source |
|---|---|---|
| RC on RTX 4090 | **~172** | derived from his `mod_mul.asm` note "time for 4090: about 120G/sec" against 20.6 Tops/s |
| This port on MI300X | **270** (240 VALU + 30 wait states) | **measured**, `tests/isa_quality.sh` |

About **57% more work for the same arithmetic**. Note the 64 `v_mad_u64_u32` in our
multiply are already the theoretical minimum for an 8x8 limb grid, and the 64
carry-accumulates are one per product - there is nothing spare left to remove. The gap is
structural.

### Decomposition of the per-GPU difference

| Factor | Direction | Magnitude |
|---|---|---|
| Integer MAD throughput | favours MI300X | 1.33x |
| Instructions per modular multiply (carry-in) | favours NVIDIA | 1.57x |
| Expected net | favours NVIDIA | 0.85x -> ~12.3 GH/s |
| Achieved VALU utilisation (59%, measured) | favours NVIDIA | 0.82x |
| **Actual** | | **10.1 GH/s, 0.70x of a 4090** |

The last row is where hand-scheduled assembly still pays for RC: he schedules the whole
kernel by hand, while this port uses assembly for the arithmetic primitives and leaves the
surrounding kernel to the compiler.

## Two secondary factors

**Clock.** 2.52 GHz boost on the 4090 against 2.1 GHz on MI300X, worth 1.2x per lane.

**Operand reuse.** The `.reuse` flag in that same SASS line is an Ada operand-reuse cache
that avoids re-reading a register-file port when consecutive instructions share an
operand. CDNA has no equivalent, and a schoolbook multiply reuses operands heavily.

## One thing that favours CDNA, measured

Serial dependency chains are free on CDNA and are not on Ada. Measured on MI300X, a serial
`v_mad_u64_u32` chain runs at 28.98 Tops/s against 27.59 for eight independent chains - the
serial version is *faster*, and a serial VCC carry chain reaches 0.958 of peak, the best
multi-operand result measured. A wave64 instruction occupies its SIMD16 for four cycles,
which covers the dependent-issue latency, and the CDNA3 ISA hazard table confirms zero
wait states for carry-in.

This inverts NVIDIA's design guidance. RC's SASS spends much of its complexity
interleaving carry chains across separate predicate registers `Pt0..Pt4` because a serial
chain stalls on Ada. On CDNA that work buys nothing and costs live registers, so this port
keeps chains long and simple.

## Where the ceilings are

Two independent limits converge at roughly the same place:

- **Instruction count.** Closing the remaining 1.39x gap (1,971 VALU per point-addition
  measured, against ~1,420 for RC) would give roughly 14-15 GH/s per GPU.
- **Memory bandwidth.** Measured 2.20 TB/s now (1.23 read + 0.97 write) against MI300X's
  5.3 TB/s peak, matching an analytical ~258 bytes per point-addition. At 15 GH/s that
  traffic reaches ~3.9 TB/s, which is the practical HBM limit.

So ~15 GH/s per MI300X is the ceiling for this algorithm structure, and the missing
carry-in accounts for approximately the entire remaining gap to it. Going further would
require reducing bytes moved per point-addition, not instructions.

## Honest scorecard

| | RTX 4090 | RTX 5090 | MI300X (this port) | 8x MI300X |
|---|---|---|---|---|
| Throughput | 14.5 GH/s | 19.3 GH/s | 10.1 GH/s | **81.0 GH/s** |
| Relative to 4090 | 1.0x | 1.33x | 0.70x | **5.6x** |
| Board power (approx) | 450 W | 575 W | 750 W | 6000 W |
| Efficiency | ~32 MH/s/W | ~34 MH/s/W | ~13 MH/s/W | ~13 MH/s/W |

**Per watt, the consumer NVIDIA parts win this workload outright**, by roughly 2.5x. That
is a direct consequence of the carry-in feature plus the clock advantage, and no amount of
tuning on our side closes it.

Where MI300X wins is aggregate throughput per host: eight accelerators in one node deliver
5.6x a single 4090 and 4.2x a single 5090, in one chassis, with 1.5 TB of HBM. Whether
that trade is worth it depends entirely on whether the constraint is power, rack space, or
capital already spent on the hardware.

## Practical takeaways

- If choosing hardware for ECDLP specifically and power is the binding constraint, consumer
  NVIDIA is the better buy. This is not a close call.
- If the MI300X hardware already exists for other work, 10.1 GH/s per GPU is a reasonable
  return, and 81 GH/s per node is competitive in absolute terms.
- Anyone porting big-integer arithmetic to CDNA should expect the missing carry-in to cost
  roughly 1.5x on multiply-heavy kernels, and should not expect cache-residency tuning to
  recover it.
- CDNA4 (gfx950) is built and arch-tuned here but has never been run - there is no MI355X
  on this machine. Its 160 KB LDS per CU (against 64 KB) should allow better occupancy, but
  that is untested.
