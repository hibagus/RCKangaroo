// 256-bit secp256k1 field arithmetic for AMD CDNA3 (gfx942) / CDNA4 (gfx950).
//
// This is the AMD counterpart of RCGpuUtils.h. The algorithms are RetiredCoder's
// (lazy Solinas reduction, Bernstein-Yang divsteps inversion); only the way the
// carry chains are expressed has changed.
//
// (c) 2024-2026 RetiredCoder (RC) for the original algorithms.
// CDNA port. License: GPLv3, see "LICENSE.TXT".
//
// ---------------------------------------------------------------------------
// Why this is plain C++ and not inline assembly
// ---------------------------------------------------------------------------
// RC's CUDA original expresses every carry chain through PTX's implicit sticky
// carry flag (add.cc/addc.cc). CDNA has no equivalent global flag - carries live
// in VCC or in an arbitrary SGPR pair - so a literal transliteration is not
// possible and the macros have to be rebuilt from scratch.
//
// Measurements (docs/CDNA_PHASE0_MEASUREMENTS.md) decided how:
//
//   * Written as ordinary __uint128_t accumulation, ROCm's backend lowers a
//     256x256->512 multiply to exactly 64 v_mad_u64_u32 - the theoretical
//     minimum for an 8x8 32-bit-limb schoolbook. It decomposes the 64-bit limbs
//     into 32-bit MADs itself, so we get RC's 32-bit-limb ISA from 64-bit-limb
//     source. 234 VALU ops total, 37 VGPRs, no spills.
//
//   * v_mad_u64_u32 is full rate (0.82-0.86x a plain v_add_u32), so those 64
//     MADs cost about what 64 adds cost.
//
//   * The carry chains use clang's __builtin_addcll/__builtin_subcll, which map
//     1:1 onto v_add_co_u32 / v_addc_co_u32. The __uint128_t form makes the
//     backend model a full 128-bit operation instead, costing ~35% more ops
//     (SubModP: 49 VALU with the builtins vs 67 with u128). Note this is the
//     opposite of the multiply, where __uint128_t wins and the builtins lose
//     (234 vs 306 VALU) - so the two are deliberately written differently.
//
//   * Serial dependency chains are FREE on CDNA - a wave64 instruction occupies
//     its SIMD16 for 4 cycles, which covers the dependent-issue latency. A
//     serial carry chain measured 0.958 of peak, the best multi-operand result.
//     This is the opposite of NVIDIA, where RC must interleave several
//     independent chains across Pt0..Pt4 to avoid stalls. We therefore keep the
//     chains long and simple: it is both faster and uses fewer registers.
//
// Two things are deliberately *not* done the obvious way:
//
//   * The multiply works on 64-bit limbs (clean source, optimal codegen) but the
//     reduction works on a 32-bit view, because reduction needs a multiply by
//     2^32 and on 32-bit limbs that is a free word-offset add. On 64-bit limbs it
//     would cost 8 v_alignbit_b32 plus wider adds.
//
//   * SqrModP forwards to MulModP. RC's dedicated squaring saves 28 of 74
//     multiplies but spends 19 more adds and needs a 28-entry cross-product
//     array (~56 live VGPRs). On CDNA, v_mad_u64_u32 fuses the accumulate, which
//     makes standalone adds relatively *more* expensive than on Ada, so trading
//     multiplies for adds is a losing move here. RCGpuCore.cu itself already
//     calls MulModP(x,x) in two of three places.
//
// Type punning between u64* and u32* is used throughout, as in the original.
// Build with -fno-strict-aliasing.

#pragma once

#include <hip/hip_runtime.h>

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef unsigned __int128 u128;

// secp256k1 field prime: p = 2^256 - 2^32 - 977
//   so 2^256 == 2^32 + 977 == 0x1000003D1  (mod p), which is what makes the
//   Solinas reduction below cheap: one 33-bit multiplier, no division.
#define P_0         0xFFFFFFFEFFFFFC2Full
#define P_123       0xFFFFFFFFFFFFFFFFull
#define P_INV32     0x000003D1              // the 977 part; the 2^32 part is a word offset
#define P0_INV_FULL 0x00000001000003D1ull   // 2^32 + 977

// ---------------------------------------------------------------------------
// Carry primitives.
//
// Each returns the sum/difference limb and updates the caller's carry/borrow.
// Expressed over __uint128_t so the backend can see a real carry dependency and
// emit v_add_co_u32 / v_addc_co_u32 chains, rather than the
// v_cmp_lt_u64 + v_cndmask_b32 pairs that a hand-rolled software carry produces.
// ---------------------------------------------------------------------------

__device__ __forceinline__ u64 adc64(u64 a, u64 b, u64* carry)
{
    u64 cout;
    u64 r = __builtin_addcll(a, b, *carry, &cout);
    *carry = cout;
    return r;
}

__device__ __forceinline__ u64 sbb64(u64 a, u64 b, u64* borrow)
{
    u64 bout;
    u64 r = __builtin_subcll(a, b, *borrow, &bout);
    *borrow = bout;
    return r;
}

__device__ __forceinline__ u32 adc32(u32 a, u32 b, u32* carry)
{
    u32 cout;
    u32 r = __builtin_addc(a, b, *carry, &cout);
    *carry = cout;
    return r;
}

__device__ __forceinline__ u32 sbb32(u32 a, u32 b, u32* borrow)
{
    u32 bout;
    u32 r = __builtin_subc(a, b, *borrow, &bout);
    *borrow = bout;
    return r;
}

// ---------------------------------------------------------------------------
// Copies. Copy_int4_x2 moves 256 bits as two 128-bit vector accesses; sources
// are usually LDS jump-table entries, so this becomes ds_read_b128 and REQUIRES
// 16-byte alignment on both operands (all call sites use __align__(16)).
// ---------------------------------------------------------------------------

#define Copy_int4_x2(dst, src) { \
    ((int4*)(dst))[0] = ((int4*)(src))[0]; \
    ((int4*)(dst))[1] = ((int4*)(src))[1]; }

#define Copy_u64_x4(dst, src) { \
    (dst)[0] = (src)[0]; (dst)[1] = (src)[1]; \
    (dst)[2] = (src)[2]; (dst)[3] = (src)[3]; }

// ---------------------------------------------------------------------------
// 192-bit kangaroo distance accumulation. Plain integers, never reduced mod n.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void Add192to192(u64* res, const u64* val)
{
    u64 c = 0;
    res[0] = adc64(res[0], val[0], &c);
    res[1] = adc64(res[1], val[1], &c);
    res[2] = adc64(res[2], val[2], &c);
}

__device__ __forceinline__ void Sub192from192(u64* res, const u64* val)
{
    u64 b = 0;
    res[0] = sbb64(res[0], val[0], &b);
    res[1] = sbb64(res[1], val[1], &b);
    res[2] = sbb64(res[2], val[2], &b);
}

// ---------------------------------------------------------------------------
// Modular add / sub / negate.
//
// Both SubModP and AddModP are branchless. RC's reference uses a data-dependent
// `if (carry)` to add p back, which on a 64-wide wave costs a divergence every
// call - and SubModP is called ~7 times per point addition. mod_sub.asm shows RC
// does it predicated (@!Pt0) in the shipped SASS; masking is the portable
// equivalent.
//
// INPUT CONTRACT: operands must be in [0, p), not merely in [0, 2^256).
//
// All three apply a *single* conditional correction, so one step of add-p or
// sub-p has to be enough. It is not, if an operand is >= p: AddModP(p+1, 2^256-1)
// returns 0 where the answer is 0x1000003D1, and SubModP(0, p+1) is off by the
// same amount. tests/check_results.py therefore skips these three whenever an
// operand is >= p, and reports how many it skipped.
//
// This is safe in the solver, and is RC's design rather than an accident. The
// only values that are not already reduced are MulModP/SqrModP outputs, which
// land in [p, 2^256) - a window of just 2^32+977 out of 2^256, i.e. with
// probability 2^-224. Making the correction unconditional would cost extra
// instructions in the hottest routine in the program to defend against an event
// that will not occur in the lifetime of the hardware.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void NegModP(u64* res)
{
    // res = p - res. Correct without a final correction because inputs are
    // always in [0, p). Note NegModP(0) == p, which the algorithm never hits.
    u64 b = 0;
    res[0] = sbb64(P_0,   res[0], &b);
    res[1] = sbb64(P_123, res[1], &b);
    res[2] = sbb64(P_123, res[2], &b);
    res[3] = sbb64(P_123, res[3], &b);
}

__device__ __forceinline__ void SubModP(u64* res, const u64* val1, const u64* val2)
{
    u64 b = 0;
    u64 r0 = sbb64(val1[0], val2[0], &b);
    u64 r1 = sbb64(val1[1], val2[1], &b);
    u64 r2 = sbb64(val1[2], val2[2], &b);
    u64 r3 = sbb64(val1[3], val2[3], &b);

    // Add p back iff it went negative. mask is all-ones on borrow; three of p's
    // four limbs are all-ones, so they need nothing beyond the mask itself.
    const u64 mask = 0ull - b;
    u64 c = 0;
    res[0] = adc64(r0, P_0 & mask, &c);
    res[1] = adc64(r1, mask, &c);
    res[2] = adc64(r2, mask, &c);
    res[3] = adc64(r3, mask, &c);
    // final carry is the 2^256 that cancels the borrow - correctly discarded
}

__device__ __forceinline__ void AddModP(u64* res, const u64* val1, const u64* val2)
{
    u64 c = 0;
    u64 t0 = adc64(val1[0], val2[0], &c);
    u64 t1 = adc64(val1[1], val2[1], &c);
    u64 t2 = adc64(val1[2], val2[2], &c);
    u64 t3 = adc64(val1[3], val2[3], &c);
    const u64 bit256 = c;          // the 257th bit of the sum

    // Speculatively subtract p, then select.
    u64 b = 0;
    u64 s0 = sbb64(t0, P_0,   &b);
    u64 s1 = sbb64(t1, P_123, &b);
    u64 s2 = sbb64(t2, P_123, &b);
    u64 s3 = sbb64(t3, P_123, &b);

    // Keep the subtracted form if the sum was >= p: either it overflowed 256
    // bits, or the speculative subtract did not borrow.
    const u64 m = 0ull - (bit256 | (b ^ 1ull));
    res[0] = (s0 & m) | (t0 & ~m);
    res[1] = (s1 & m) | (t1 & ~m);
    res[2] = (s2 & m) | (t2 & ~m);
    res[3] = (s3 & m) | (t3 & ~m);
}

// ---------------------------------------------------------------------------
// 256 x 256 -> 512 multiply.
//
// Operand-scanning schoolbook on 64-bit limbs. This exact shape is what the
// backend lowers to the minimum 64 v_mad_u64_u32: every
// (u128)a*b + acc + carry maps 1:1 onto the hardware's
// { carry, D.u64 } = S0.u32 * S1.u32 + S2.u64.
//
// `r[i+4] = c` rather than `+=` is safe: in this traversal order r[i+4] is still
// zero when row i finishes, and row i+1 then reads that carry back in.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mul_256_to_512(u64* r, const u64* a, const u64* b)
{
    #pragma unroll
    for (int i = 0; i < 8; i++) r[i] = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        u64 c = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            u128 t = (u128)a[i] * (u128)b[j] + (u128)r[i + j] + (u128)c;
            r[i + j] = (u64)t;
            c = (u64)(t >> 64);
        }
        r[i + 4] = c;
    }
}

// ---------------------------------------------------------------------------
// Reduce 512 bits mod p, using 2^256 == 2^32 + 977 == 0x1000003D1.
//
//   res = lo256 + hi256 * 0x1000003D1
//
// in two folds, because hi256 * 0x1000003D1 is 289 bits wide.
//
// This runs on 64-bit limbs. The 32-bit-limb version - which is what RC uses,
// because on 32-bit limbs the 2^32 term is a free word-offset add rather than a
// shift - measured 42% *worse* here (195 vs 113 VALU). Expressing the whole
// thing as one 4-limb-by-scalar multiply by 0x1000003D1 lets the backend apply
// the same optimal mul-accumulate lowering it uses for the main multiply, and
// that is worth more than avoiding the shift.
//
// The result is in [0, 2^256), NOT reduced to [0, p). That lazy convention is
// RC's and every consumer here tolerates it; forcing a full reduction would add
// a conditional subtract to the hottest routine in the solver.
//
// Unlike RC's reference we do fold the final carry back rather than discarding
// it. Discarding is wrong by 2^256 whenever the second fold overflows - which is
// astronomically rare, but folding costs ~4 instructions out of ~370, so there
// is no reason to keep the hazard.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void reduce_512_to_256(u64* res, const u64* buff)
{
    const u64* lo = buff;
    const u64* hi = buff + 4;

    // First fold: t = hi * 0x1000003D1, five limbs.
    u64 t[5];
    {
        u64 c = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            u128 pr = (u128)hi[i] * (u128)P0_INV_FULL + (u128)c;
            t[i] = (u64)pr;
            c = (u64)(pr >> 64);
        }
        t[4] = c;
    }

    // res = lo + t[0..3]; everything above 2^256 collects into ov (<= ~34 bits).
    u64 r[4], ov;
    {
        u64 c = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) r[i] = adc64(lo[i], t[i], &c);
        ov = c + t[4];
    }

    // Second fold: ov * 0x1000003D1 < 2^67, so two limbs.
    {
        u128 f = (u128)ov * (u128)P0_INV_FULL;
        u64 c = 0;
        r[0] = adc64(r[0], (u64)f, &c);
        r[1] = adc64(r[1], (u64)(f >> 64), &c);
        r[2] = adc64(r[2], 0, &c);
        r[3] = adc64(r[3], 0, &c);

        // Third fold: at most one bit escaped, worth 0x1000003D1. Uniformly
        // not taken across a wave in practice, so the branch costs ~3 ops and
        // still beats the branchless mask form (113 vs 116 VALU).
        if (c)
        {
            u64 c2 = 0;
            r[0] = adc64(r[0], P0_INV_FULL, &c2);
            r[1] = adc64(r[1], 0, &c2);
            r[2] = adc64(r[2], 0, &c2);
            r[3] = adc64(r[3], 0, &c2);
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) res[i] = r[i];
}

__device__ __forceinline__ void MulModP(u64* res, const u64* val1, const u64* val2)
{
    u64 buff[8];
    mul_256_to_512(buff, val1, val2);
    reduce_512_to_256(res, buff);
}

// See the header comment: a dedicated squaring is a net loss on CDNA because
// v_mad_u64_u32 fuses the accumulate, so trading multiplies for adds does not pay.
// Kept as a separate symbol so call sites match the CUDA reference and so a
// specialised version can be dropped in later without touching the kernels.
__device__ __forceinline__ void SqrModP(u64* res, const u64* val)
{
    u64 buff[8];
    mul_256_to_512(buff, val, val);
    reduce_512_to_256(res, buff);
}

// ---------------------------------------------------------------------------
// 288-bit signed helpers for the inversion. 9 x u32, two's complement, word 8
// carries the sign.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void add_288(u32* res, const u32* val1, const u32* val2)
{
    u32 c = 0;
    #pragma unroll
    for (int i = 0; i < 9; i++) res[i] = adc32(val1[i], val2[i], &c);
}

__device__ __forceinline__ void neg_288(u32* res)
{
    u32 b = 0;
    #pragma unroll
    for (int i = 0; i < 9; i++) res[i] = sbb32(0u, res[i], &b);
}

__device__ __forceinline__ void mul_288_by_i32(u32* res, const u32* val288, int ival32)
{
    // |ival32| times a two's-complement 288-bit value, truncated to 288 bits,
    // then negated if the scalar was negative. Word 8 keeps only the low half of
    // its product, which is what makes the truncation correct.
    const u32 val32 = (u32)abs(ival32);
    u64 c = 0;
    #pragma unroll
    for (int i = 0; i < 9; i++)
    {
        u64 pr = (u64)val288[i] * (u64)val32 + c;
        res[i] = (u32)pr;
        c = pr >> 32;
    }
    if (ival32 < 0) neg_288(res);
}

__device__ __forceinline__ void set_288_i32(u32* res, int val)
{
    res[0] = (u32)val;
    const u32 sx = (val < 0) ? 0xFFFFFFFFu : 0u;
    #pragma unroll
    for (int i = 1; i < 9; i++) res[i] = sx;
}

__device__ __forceinline__ void mul_P_by_32(u32* res, u32 val)
{
    // p * val in 288 bits with a single multiply, because
    // p * val == val * 2^256 - val * 0x1000003D1.
    u64 m = (u64)val * (u64)P_INV32;
    u32 tmp0 = (u32)m;
    u32 c = 0;
    u32 tmp1 = adc32((u32)(m >> 32), val, &c);
    u32 tmp2 = c;

    u32 b = 0;
    res[0] = sbb32(0u, tmp0, &b);
    res[1] = sbb32(0u, tmp1, &b);
    res[2] = sbb32(0u, tmp2, &b);
    #pragma unroll
    for (int i = 3; i < 8; i++) res[i] = sbb32(0u, 0u, &b);
    res[8] = sbb32(val, 0u, &b);
}

__device__ __forceinline__ void shiftR_288_by_30(u32* res)
{
    // __funnelshift_r lowers to a single v_alignbit_b32 on CDNA - an exact
    // equivalent of PTX shf.r.wrap.b32, so this one is free.
    #pragma unroll
    for (int i = 0; i < 8; i++) res[i] = __funnelshift_r(res[i], res[i + 1], 30);
    res[8] = (u32)(((int)res[8]) >> 30);   // arithmetic: the value is signed
}

__device__ __forceinline__ void add_288_P(u32* res)
{
    static const u32 Pw[9] = { 0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
                               0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0 };
    u32 c = 0;
    #pragma unroll
    for (int i = 0; i < 9; i++) res[i] = adc32(res[i], Pw[i], &c);
}

__device__ __forceinline__ void sub_288_P(u32* res)
{
    static const u32 Pw[9] = { 0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
                               0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0 };
    u32 b = 0;
    #pragma unroll
    for (int i = 0; i < 9; i++) res[i] = sbb32(res[i], Pw[i], &b);
}

// ---------------------------------------------------------------------------
// Modular inverse: Bernstein-Yang "safegcd" divsteps, 30 steps per batch.
// https://tches.iacr.org/index.php/TCHES/article/download/8298/7648/4494
//
// This is a direct port of RC's version, itself a port of libsecp256k1's
// secp256k1_modinv32_divsteps_30_var. 0xD2253531 & 0x3FFFFFFF == 0x12253531 is
// libsecp256k1's modulus_inv30 (-p^-1 mod 2^30).
//
// Operates in place on 9 x u32 (288 bits); res[0..7] in, res[0..8] out, and the
// output is in [0, 2^256) like MulModP's.
//
// Variable-time and data-dependent. On wave64 all 64 lanes run until the slowest
// lane's divsteps converge - typically ~12-13 batches against a bound of 20 -
// which is a worse penalty than on NVIDIA's warp32. KernelA amortises it over
// PNT_GROUP_CNT kangaroos via the batched-inversion trick, so it is ~1/24 of a
// point addition; measure the real divergence cost before restructuring it.
// ---------------------------------------------------------------------------

#define APPLY_DIV_SHIFT() { matrix[0] <<= index; matrix[1] <<= index; kbnt -= index; _val >>= index; }
#define DO_INV_STEP() { kbnt = -kbnt; int tw = -_modp; _modp = _val; _val = tw; \
                        tw = -matrix[0]; matrix[0] = matrix[2]; matrix[2] = tw; \
                        tw = -matrix[1]; matrix[1] = matrix[3]; matrix[3] = tw; }

// One 30-step divstep batch: refines the 2x2 transition matrix from the low words.
// Fully serial and latency-bound; ~16-19 inner iterations, ~1.6-2 bits each.
#define DIVSTEP_BATCH_30() { \
    index = __ffs(_val | 0x40000000) - 1; \
    APPLY_DIV_SHIFT(); \
    cnt = 30 - index; \
    while (cnt > 0) { \
        if (kbnt < 0) DO_INV_STEP(); \
        mx = (kbnt + 1 < cnt) ? 31 - kbnt : 32 - cnt; \
        i32 mul = (-_modp * _val) & 7; \
        mul &= 0xFFFFFFFF >> mx; \
        _val += _modp * mul; \
        matrix[2] += matrix[0] * mul; \
        matrix[3] += matrix[1] * mul; \
        index = __ffs(_val | (1 << cnt)) - 1; \
        APPLY_DIV_SHIFT(); \
        cnt -= index; \
    } }

__device__ __forceinline__ void InvModP(u32* res)
{
    int matrix[4], _val, _modp, index, cnt, mx, kbnt;
    __align__(8) u32 modp[9];
    __align__(8) u32 val[9];
    __align__(8) u32 a[9];
    __align__(8) u32 tmp[4][9 + 1];   // +1 keeps tmp[>0] 64-bit aligned

    ((u64*)modp)[0] = P_0;
    ((u64*)modp)[1] = P_123;
    ((u64*)modp)[2] = P_123;
    ((u64*)modp)[3] = P_123;
    modp[8] = 0;

    res[8] = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) val[i] = res[i];
    val[8] = 0;

    matrix[0] = matrix[3] = 1;
    matrix[1] = matrix[2] = 0;
    kbnt = -1;
    _val  = (int)res[0];
    _modp = (int)P_0;
    DIVSTEP_BATCH_30();

    // First batch application. res/a start as the identity pair (1, 0) implicitly,
    // so the first update uses set_288_i32 of the matrix entries rather than a
    // full 288-bit multiply.
    mul_288_by_i32(tmp[0], modp, matrix[0]);
    mul_288_by_i32(tmp[1], val,  matrix[1]);
    mul_288_by_i32(tmp[2], modp, matrix[2]);
    mul_288_by_i32(tmp[3], val,  matrix[3]);
    add_288(modp, tmp[0], tmp[1]); shiftR_288_by_30(modp);
    add_288(val,  tmp[2], tmp[3]); shiftR_288_by_30(val);

    set_288_i32(tmp[1], matrix[1]);
    set_288_i32(tmp[3], matrix[3]);
    // Add the multiple of p that zeroes the low 30 bits, so the >>30 is exact.
    mul_P_by_32(res, (tmp[1][0] * 0xD2253531u) & 0x3FFFFFFFu);
    add_288(res, res, tmp[1]); shiftR_288_by_30(res);
    mul_P_by_32(a,   (tmp[3][0] * 0xD2253531u) & 0x3FFFFFFFu);
    add_288(a, a, tmp[3]); shiftR_288_by_30(a);

    while (1)
    {
        matrix[0] = matrix[3] = 1;
        matrix[1] = matrix[2] = 0;
        _val  = (int)val[0];
        _modp = (int)modp[0];
        DIVSTEP_BATCH_30();

        mul_288_by_i32(tmp[0], modp, matrix[0]);
        mul_288_by_i32(tmp[1], val,  matrix[1]);
        mul_288_by_i32(tmp[2], modp, matrix[2]);
        mul_288_by_i32(tmp[3], val,  matrix[3]);
        add_288(modp, tmp[0], tmp[1]); shiftR_288_by_30(modp);
        add_288(val,  tmp[2], tmp[3]); shiftR_288_by_30(val);

        mul_288_by_i32(tmp[0], res, matrix[0]);
        mul_288_by_i32(tmp[1], a,   matrix[1]);

        // gcd reached: val == 0. tmp[0]/tmp[1] hold the final res update, applied
        // after the loop - which is why they are computed before this test.
        if ((val[0] | val[1] | val[2] | val[3] |
             val[4] | val[5] | val[6] | val[7]) == 0) break;

        mul_288_by_i32(tmp[2], res, matrix[2]);
        mul_288_by_i32(tmp[3], a,   matrix[3]);
        mul_P_by_32(res, ((tmp[0][0] + tmp[1][0]) * 0xD2253531u) & 0x3FFFFFFFu);
        add_288(res, res, tmp[0]); add_288(res, res, tmp[1]); shiftR_288_by_30(res);
        mul_P_by_32(a,   ((tmp[2][0] + tmp[3][0]) * 0xD2253531u) & 0x3FFFFFFFu);
        add_288(a, a, tmp[2]); add_288(a, a, tmp[3]); shiftR_288_by_30(a);
    }

    mul_P_by_32(res, ((tmp[0][0] + tmp[1][0]) * 0xD2253531u) & 0x3FFFFFFFu);
    add_288(res, res, tmp[0]); add_288(res, res, tmp[1]); shiftR_288_by_30(res);

    if ((int)modp[8] < 0) neg_288(res);          // gcd came out as -1
    while ((int)res[8] < 0) add_288_P(res);      // normalise word 8 to zero
    while ((int)res[8] > 0) sub_288_P(res);
}

// ---------------------------------------------------------------------------
// Non-temporal 16-bit store, the CDNA equivalent of PTX st.global.cs.b16.
//
// The jump list is written once by KernelA and read once by KernelB, so it must
// not evict the kangaroo state. Per the CDNA3 ISA load/store control tables,
// nt=1 gives "Miss Evict" in the CU cache and "Hit Stream" in L2 - exactly the
// streaming behaviour .cs asks for on NVIDIA.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void st_cs_b16(u16* addr, u32 val)
{
    __builtin_nontemporal_store((u16)val, addr);
}

__device__ __forceinline__ void st_cs_b32(u32* addr, u32 val)
{
    __builtin_nontemporal_store(val, addr);
}
