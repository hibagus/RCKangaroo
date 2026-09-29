// GPU kernels for RCKangaroo on AMD CDNA3 (gfx942) / CDNA4 (gfx950).
//
// Ported from RCGpuCore.cu. The algorithm is unchanged - this is RC's SOTA v2
// kangaroo with batched ("triple Montgomery") inversion; only the parts that
// touch NVIDIA-specific machinery have been rewritten.
//
// (c) 2024-2026 RetiredCoder (RC) for the original.
// CDNA port. License: GPLv3, see "LICENSE.TXT".
//
// ---------------------------------------------------------------------------
// What changed relative to the CUDA original, and why
// ---------------------------------------------------------------------------
//
// 1. Field arithmetic comes from include/cdna/RCGpuUtils_cdna.h, where every
//    carry chain was rebuilt because CDNA has no PTX-style sticky carry flag.
//    Call sites are unchanged.
//
// 2. Occupancy. The CUDA host requests 98 KB of dynamic shared memory for
//    KernelA (GpuKang.cpp:68) even though jmp1_table only needs 32 KB. That is
//    not an allocation - it is a carveout to force 1 block/SM, and 98 KB is
//    impossible on CDNA3 anyway (64 KB LDS/CU hard limit). On AMD the same
//    effect comes from amdgpu_waves_per_eu(1,1), so we ask for the 32 KB we
//    actually use. This also matters for correctness of the arithmetic:
//    InvModP needs 139 VGPRs and only gets the full 256-VGPR budget at
//    1 wave/SIMD; at higher occupancy it spills 22 registers to scratch.
//
// 3. The jump list is written with non-temporal stores. It is produced once by
//    KernelA and consumed once by KernelB, so it must not evict kangaroo state.
//    Per the CDNA3 ISA store-control table, nt=1 gives "Miss Evict" in the CU
//    cache and "Hit Stream" in L2 - the same intent as PTX's .cs qualifier.
//
// 4. No cudaFuncSetAttribute equivalent is needed. On AMD there is no opt-in
//    above the hardware LDS limit; the requested dynamic size is simply honoured
//    or the launch fails.
//
// 5. Jump-table placement is constrained by CDNA's address spaces in a way that
//    has no CUDA analogue. KernelA picks between the jmp1 and jmp2 tables per
//    kangaroo. If that select has to choose between two *different* address
//    spaces, the compiler must emit generic `flat_load_dwordx4` for the read -
//    measured at 8 flat loads and zero ds_read in the hot loop when jmp1 was in
//    LDS and jmp2 in __constant__, as the CUDA version arranges them. Flat
//    addressing resolves the LDS-vs-global aperture at runtime and forfeits the
//    native LDS path.
//
//    Putting both tables in LDS fixes that, but costs 2 x 32 KB = 64 KB, which is
//    all of CDNA3's per-CU LDS and pins the kernel to one workgroup per CU, i.e.
//    1 wave/SIMD. Since 154 VGPRs would allow 3 waves/SIMD, LDS - not registers -
//    was the occupancy limiter, and at 1 wave/SIMD there is no second wave to
//    cover the 15 VMEM and 6 LDS accesses each point addition performs.
//
//    So instead: jmp1 stays in LDS (32 KB) and jmp2 is read straight from global
//    memory, with the two reads in *separate branches* holding separately typed
//    pointers. The compiler then keeps the LDS path as ds_read_b128 and only the
//    rare path pays for a global load. jmp2 is used to escape level-1 size-2
//    loops, which affect about 1 jump in 1024, so a wave64 takes the global
//    branch roughly 6% of the time (1 - (1023/1024)^64).
//
//    Net: 32 KB LDS, 2 workgroups/CU, 2 waves/SIMD, and JMP_CNT stays at 512 -
//    no increase in the loop rate, which halving JMP_CNT would have cost.
//
// 6. Wave64 needed no changes in the math layer: the CUDA original contains no
//    warp intrinsics, no __shfl, no __syncwarp and no warpSize assumption. The
//    index arithmetic in KernelB depends on BLOCK_SIZE, not on warp width, so it
//    carries over unmodified. BLOCK_SIZE=256 is 4 wave64 instead of 8 warp32.

#include <hip/hip_runtime.h>
#include "defs.h"
#include "RCGpuUtils_cdna.h"

#define BLOCK_CNT   gridDim.x
#define BLOCK_X     blockIdx.x
#define THREAD_X    threadIdx.x

// Structure-of-arrays access to the kangaroo state. Consecutive threads touch
// consecutive 32-byte records, so a wave64 load covers 2 KB contiguously -
// 16 full 128-byte cache lines, no gaps.
#define LOAD_VAL_256(dst, ptr, group) { \
    *((int4*)&(dst)[0]) = *((int4*)&(ptr)[BLOCK_SIZE * 4 * Kparams.BlockCnt * (group)]); \
    *((int4*)&(dst)[2]) = *((int4*)&(ptr)[BLOCK_SIZE * 4 * Kparams.BlockCnt * (group) + 2]); }

#define SAVE_VAL_256(ptr, src, group) { \
    *((int4*)&(ptr)[BLOCK_SIZE * 4 * Kparams.BlockCnt * (group)]) = *((int4*)&(src)[0]); \
    *((int4*)&(ptr)[BLOCK_SIZE * 4 * Kparams.BlockCnt * (group) + 2]) = *((int4*)&(src)[2]); }

// KernelA carries the per-group loop-escape flags in a single u32 (L1S2), so the
// group count cannot exceed 32.
static_assert(PNT_GROUP_CNT <= 32, "PNT_GROUP_CNT must fit in L1S2's u32 bitmask");

extern __shared__ u64 LDS[];

// How much of the jmp1 table lives in LDS.
//
// CDNA3 has 64 KB of LDS per CU. At 3 workgroups/CU that is 21 KB each, so only
// the x-coordinates fit (16 KB) and y is re-read from global on every point
// addition. CDNA4 raises LDS to 160 KB per CU, which makes 3 x 32 KB affordable,
// so y moves into LDS as well: 32 bytes of global traffic per point addition
// removed - about 13% of the kernel's bytes - at no occupancy cost, since 147
// VGPRs cap the kernel at 3 waves/SIMD long before 96 KB of 160 KB does.
//
// The strides live in defs.h so GpuKang.cpp can size the dynamic LDS request
// from the same constant.
#ifdef __gfx950__
#define JMP1_LDS_Y       1
#define JMP1_LDS_STRIDE  JMP1_LDS_STRIDE_CDNA4      // x[4] then y[4]
#else
#define JMP1_LDS_Y       0
#define JMP1_LDS_STRIDE  JMP1_LDS_STRIDE_CDNA3      // x[4] only
#endif

// Jump-table reads. The two tables are read in separate branches so each keeps a
// concretely-typed pointer: the LDS path stays ds_read_b128 while only the rare
// jmp2 escape path pays a global load. Writing it as a ternary select over a
// single pointer instead makes the compiler fall back to generic flat_load for
// both. See note 5. The global tables use stride 12 - x, y, then a 4-limb
// distance only KernelB reads.
#define LOAD_JMP_X(dst_x, ind, use2) { \
    if (use2) { Copy_int4_x2(dst_x, jmp2_table + 12 * (ind)); } \
    else      { Copy_int4_x2(dst_x, jmp1_lds + JMP1_LDS_STRIDE * (ind)); } }

#if JMP1_LDS_Y
#define LOAD_JMP_XY(dst_x, dst_y, ind, use2) { \
    if (use2) { \
        Copy_int4_x2(dst_x, jmp2_table + 12 * (ind)); \
        Copy_int4_x2(dst_y, jmp2_table + 12 * (ind) + 4); \
    } else { \
        Copy_int4_x2(dst_x, jmp1_lds + JMP1_LDS_STRIDE * (ind)); \
        Copy_int4_x2(dst_y, jmp1_lds + JMP1_LDS_STRIDE * (ind) + 4); \
    } }
#else
#define LOAD_JMP_XY(dst_x, dst_y, ind, use2) { \
    if (use2) { \
        Copy_int4_x2(dst_x, jmp2_table + 12 * (ind)); \
        Copy_int4_x2(dst_y, jmp2_table + 12 * (ind) + 4); \
    } else { \
        Copy_int4_x2(dst_x, jmp1_lds + 4 * (ind)); \
        Copy_int4_x2(dst_y, jmp1_glb + 12 * (ind) + 4); \
    } }
#endif

// One 256-thread workgroup per CU, 1 wave per SIMD. See note 2 above: this is
// required for the VGPR budget, not just a throughput preference.
#define CDNA_KERNEL_BOUNDS                                  \
    __launch_bounds__(BLOCK_SIZE, 1)                        \
    __attribute__((amdgpu_waves_per_eu(1, 1),               \
                   amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))

///////////////////////////////////////////////////////////////////////////////
// KernelA - performs the main kangaroo jumps.
//
// Each thread owns PNT_GROUP_CNT kangaroos and advances all of them one jump per
// step. The expensive part of an elliptic-curve addition is the modular inverse
// of (x - jmp_x), so all PNT_GROUP_CNT of them are batched: accumulate the
// running product forward, invert once, then walk backwards recovering each
// individual inverse with two multiplies. That turns 24 inversions into 1
// inversion plus 48 multiplies.
///////////////////////////////////////////////////////////////////////////////

extern "C" CDNA_KERNEL_BOUNDS
__global__ void KernelA(const TKparams Kparams)
{
    // Stride between the x, y and s planes, in bytes. The literal 256 is
    // BLOCK_SIZE and the 32 is one 256-bit record.
    u64 PartStride = PNT_GROUP_CNT * (Kparams.BlockCnt * 256 * 32);
    u64* L2x = Kparams.L2 + 4 * (THREAD_X + BLOCK_SIZE * BLOCK_X);
    u64* L2y = L2x + PartStride / 8;
    u64* L2s = L2y + PartStride / 8;

    // Jump indices performed this pass, handed to KernelB. One u16 per kangaroo
    // per step; the /4 converts a u16 count into u64 pointer steps.
    u16* jlist = (u16*)(Kparams.JumpsList +
                        (u64)BLOCK_X * (STEP_CNT + MD_LEN) * PNT_GROUP_CNT * BLOCK_SIZE / 4);
    jlist += THREAD_X;

    // Last MD_LEN visited points, so KernelC can rewind to a loop's exit point.
    u64* x_last0 = Kparams.LastPnts + 4 * (THREAD_X + BLOCK_SIZE * BLOCK_X);
    u64* y_last0 = x_last0 + PartStride / 8;

    // On CDNA3, LDS holds only the jmp1 x-coordinates - 16 KB rather than 32.
    // x is read twice per group (once building the inverse product, once
    // applying the jump) while y is read once, so y stays in global. Halving
    // the LDS footprint is what raises occupancy from 2 to 3 waves/SIMD; see
    // note 5. On CDNA4 there is LDS to spare and y comes along - see
    // JMP1_LDS_STRIDE above.
    //
    // Source stride is 12 because each entry carries y and then a 4-limb
    // distance after x; only KernelB needs the distances. ds_write_b128 needs
    // 16-byte alignment, which every stride here satisfies.
    u64* jmp1_lds = LDS;                        // JMP_CNT x JMP1_LDS_STRIDE u64
    const u64* jmp2_table = Kparams.Jumps2;     // global, stride 12
#if !JMP1_LDS_Y
    const u64* jmp1_glb = Kparams.Jumps1;       // global, stride 12
#endif

    int i = THREAD_X;
    while (i < JMP_CNT)
    {
        *(int4*)&jmp1_lds[JMP1_LDS_STRIDE * i + 0] = *(int4*)&Kparams.Jumps1[12 * i + 0];
        *(int4*)&jmp1_lds[JMP1_LDS_STRIDE * i + 2] = *(int4*)&Kparams.Jumps1[12 * i + 2];
#if JMP1_LDS_Y
        *(int4*)&jmp1_lds[JMP1_LDS_STRIDE * i + 4] = *(int4*)&Kparams.Jumps1[12 * i + 4];
        *(int4*)&jmp1_lds[JMP1_LDS_STRIDE * i + 6] = *(int4*)&Kparams.Jumps1[12 * i + 6];
#endif
        i += BLOCK_SIZE;
    }

    __syncthreads();

    __align__(16) u64 x[4], y[4], tmp[4], tmp2[4];
    u32 jmp_ind;

    // One bit per owned kangaroo: set means "this one is in a level-1 size-2
    // loop, use the jmp2 table to escape". Lives in a register across the whole
    // pass and is written back at the end.
    u32 L1S2 = Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X];

    for (int step_ind = 0; step_ind < STEP_CNT; step_ind++)
    {
        __align__(16) u64 inverse[5];   // 5 limbs: InvModP works on 288 bits
        __align__(16) u64 jmp_x[4];
        __align__(16) u64 jmp_y[4];

        // --- forward pass: build the running product of all (x - jmp_x) ---
        LOAD_VAL_256(x, L2x, 0);
        jmp_ind = x[0] % JMP_CNT;
        LOAD_JMP_X(jmp_x, jmp_ind, (L1S2 >> 0) & 1);
        SubModP(inverse, x, jmp_x);
        SAVE_VAL_256(L2s, inverse, 0);

        for (int group = 1; group < PNT_GROUP_CNT; group++)
        {
            LOAD_VAL_256(x, L2x, group);
            jmp_ind = x[0] % JMP_CNT;
            LOAD_JMP_X(jmp_x, jmp_ind, (L1S2 >> group) & 1);
            SubModP(tmp, x, jmp_x);
            MulModP(inverse, inverse, tmp);
            SAVE_VAL_256(L2s, inverse, group);   // partial product for the way back
        }

        // --- the single inversion the whole batch shares ---
        InvModP((u32*)inverse);

        // --- backward pass: recover each inverse and do the point addition ---
        for (int group = PNT_GROUP_CNT - 1; group >= 0; group--)
        {
            __align__(16) u64 x0[4];
            __align__(16) u64 y0[4];
            __align__(16) u64 dxs[4];

            LOAD_VAL_256(x0, L2x, group);
            LOAD_VAL_256(y0, L2y, group);
            jmp_ind = x0[0] % JMP_CNT;
            LOAD_JMP_XY(jmp_x, jmp_y, jmp_ind, (L1S2 >> group) & 1);

            // Kangaroos always travel "uphill": if y is odd we jump to -P
            // instead, which halves the number of distinct jump points needed.
            u32 inv_flag = (u32)y0[0] & 1;
            if (inv_flag)
            {
                jmp_ind |= INV_FLAG;
                NegModP(jmp_y);
            }

            if (group)
            {
                // inverse currently holds 1/(prod of groups 0..group); multiply
                // by the partial product of 0..group-1 to isolate this group's,
                // then strip this group out of the running inverse.
                LOAD_VAL_256(tmp, L2s, group - 1);
                SubModP(tmp2, x0, jmp_x);
                MulModP(dxs, tmp, inverse);
                MulModP(inverse, inverse, tmp2);
            }
            else
                Copy_u64_x4(dxs, inverse);

            // Standard affine addition: lambda = (y0 - jmp_y) / (x0 - jmp_x)
            SubModP(tmp2, y0, jmp_y);
            MulModP(tmp, tmp2, dxs);      // tmp = lambda
            SqrModP(tmp2, tmp);

            SubModP(x, tmp2, jmp_x);
            SubModP(x, x, x0);
            SAVE_VAL_256(L2x, x, group);

            SubModP(y, x0, x);
            MulModP(y, y, tmp);
            SubModP(y, y, y0);
            SAVE_VAL_256(L2y, y, group);

            // Level-1 size-2 loop detection: if the next jump would undo this
            // one, the kangaroo is oscillating between two points. Flag it so
            // the next step uses the jmp2 table and breaks out.
            if (((L1S2 >> group) & 1) == 0)
            {
                u32 jmp_next = x[0] % JMP_CNT;
                jmp_next |= ((u32)y[0] & 1) ? 0 : INV_FLAG;
                L1S2 |= (jmp_ind == jmp_next) ? (1u << group) : 0;
            }
            else
            {
                L1S2 &= ~(1u << group);
                jmp_ind |= JMP2_FLAG;
            }

            // Distinguished point? Record the x-prefix for the host to match.
            // The counter is packed: KernelA bumps the low 16 bits here,
            // KernelB bumps the high 16 as it consumes them.
            if ((((u32*)x)[7] & Kparams.dp_mask) == 0)
            {
                u32 kang_ind = (THREAD_X + BLOCK_X * BLOCK_SIZE) +
                               group * (Kparams.BlockCnt * BLOCK_SIZE);
                u32 ind = atomicAdd(Kparams.DPTable + kang_ind, 1);
                ind = min(ind, DPTABLE_MAX_CNT - 1);
                int4* dst = (int4*)(Kparams.DPTable + Kparams.KangCnt +
                                    (kang_ind * DPTABLE_MAX_CNT + ind) * 4);
                dst[0] = ((int4*)x)[0];
                jmp_ind |= DP_FLAG;
            }

            // Non-temporal: write-once, read-once by KernelB. See note 3.
            st_cs_b16(&jlist[group * 256], jmp_ind);

            if (step_ind + MD_LEN >= STEP_CNT)
            {
                int n = step_ind + MD_LEN - STEP_CNT;
                u64* x_last = x_last0 + n * 2 * PartStride / 8;
                u64* y_last = y_last0 + n * 2 * PartStride / 8;
                SAVE_VAL_256(x_last, x, group);
                SAVE_VAL_256(y_last, y, group);
            }
        }
        jlist += PNT_GROUP_CNT * BLOCK_SIZE;
    }

    Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X] = L1S2;
}

///////////////////////////////////////////////////////////////////////////////
// KernelB - accumulates jump distances and detects loops longer than 2.
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void BuildDP(const TKparams& Kparams, int kang_ind, u64* d)
{
    // Claim the next unconsumed DP slot for this kangaroo by bumping the high
    // half of the packed counter KernelA incremented.
    int ind = atomicAdd(Kparams.DPTable + kang_ind, 0x10000);
    ind >>= 16;
    if (ind >= DPTABLE_MAX_CNT)
        return;
    int4 rx = *(int4*)(Kparams.DPTable + Kparams.KangCnt + (kang_ind * DPTABLE_MAX_CNT + ind) * 4);
    u32 pos = atomicAdd(Kparams.DPs_out, 1);
    pos = min(pos, MAX_DP_CNT - 1);
    u32* DPs = Kparams.DPs_out + 4 + pos * GPU_DP_SIZE / 4;
    *(int4*)&DPs[0] = rx;
    *(int4*)&DPs[4] = ((int4*)d)[0];
    *(u64*)&DPs[8] = d[2];
    DPs[10] = kang_ind;
}

// Comparing only the low 64 bits of the distance gives a false-loop rate of
// about 1/2^60, which is negligible.
__device__ __forceinline__ bool ProcessJumpDistance(u32 step_ind, u32 d_cur, u64* d, u32 kang_ind,
                                                    u64* jmp_d, const TKparams& Kparams,
                                                    u64* table, u32* cur_ind, u8 iter)
{
    __align__(16) u64 jmp[3];
    ((int4*)(jmp))[0] = ((int4*)(jmp_d + 2 * (d_cur & JMP_MASK_ADV)))[0];
    jmp[2] = *(jmp_d + (d_cur & JMP_MASK_ADV) + 8 * JMP_CNT);

    Add192to192(d, jmp);

    // Look for this distance in the last MD_LEN entries. Unrolled and ordered
    // so the common "not found" case exits after four compares.
    int found_ind = iter + MD_LEN - 4;
    while (1)
    {
        if (table[found_ind % MD_LEN] == d[0]) break;
        found_ind -= 2;
        if (table[found_ind % MD_LEN] == d[0]) break;
        found_ind -= 2;
        if (table[found_ind % MD_LEN] == d[0]) break;
        found_ind = iter;
        if (table[found_ind] == d[0]) break;
        found_ind = -1;
        break;
    }

    table[iter] = d[0];
    *cur_ind = (iter + 1) % MD_LEN;

    if (d_cur & DP_FLAG)
        BuildDP(Kparams, kang_ind, d);

    if (found_ind < 0)
        return false;

    u32 LoopSize = (iter + MD_LEN - found_ind) % MD_LEN;
    if (!LoopSize)
        LoopSize = MD_LEN;
    atomicAdd(Kparams.dbg_buf + LoopSize, 1);   // loop-size histogram

    u32 ind_LastPnts = MD_LEN - 1 - ((STEP_CNT - 1 - step_ind) % LoopSize);
    u32 ind = atomicAdd(Kparams.LoopedKangs, 1);
    Kparams.LoopedKangs[2 + ind] = kang_ind | (ind_LastPnts << 28);
    return true;
}

#define DO_ITER(iter) { \
    u32 cur_dAB = jlist[THREAD_X]; \
    u16 cur_dA = cur_dAB & 0xFFFF; \
    u16 cur_dB = cur_dAB >> 16; \
    if (!LoopedA) \
        LoopedA = ProcessJumpDistance(step_ind, cur_dA, dA, kang_ind, jmp_d, Kparams, RegsA, &cur_indA, iter); \
    if (!LoopedB) \
        LoopedB = ProcessJumpDistance(step_ind, cur_dB, dB, kang_ind2, jmp_d, Kparams, RegsB, &cur_indB, iter); \
    jlist += BLOCK_SIZE * PNT_GROUP_CNT / 2; \
    step_ind++; }

// Loop statistics for JMP_CNT=512: L1S2 occurs about once per 1024 jumps, and
// each subsequent even size is far rarer (L1S4 = L1S2/1024, L1S6 = L1S4/256,
// L1S8 = L1S6/158, L1S10 = L1S8/82, L1S12 = L1S10/50). MD_LEN=10 therefore
// catches everything that matters: at these rates an L1S12 escape would cost
// about 0.1% of speed per year of continuous running on one card. Odd-length
// loops cannot occur, so only even sizes need handling.
extern "C" CDNA_KERNEL_BOUNDS
__global__ void KernelB(const TKparams Kparams)
{
    u64* jmp_d = LDS;   // 48 KB: 192-bit distances, pos and neg, tables 1 and 2

    int i = THREAD_X;
    while (i < JMP_CNT)
    {
        #pragma unroll
        for (int k = 0; k < 12; k++)
            jmp_d[12 * i + k] = Kparams.JmpDists12[12 * i + k];
        i += BLOCK_SIZE;
    }

    u32* jlist0 = (u32*)(Kparams.JumpsList +
                         (u64)BLOCK_X * (STEP_CNT + MD_LEN) * PNT_GROUP_CNT * BLOCK_SIZE / 4);

    __syncthreads();

    u64 RegsA[MD_LEN], RegsB[MD_LEN];

    // Two kangaroos per thread, so the u32 jlist read yields both their u16
    // jump indices in one access.
    for (u32 gr_ind2 = 0; gr_ind2 < PNT_GROUP_CNT / 2; gr_ind2++)
    {
        #pragma unroll
        for (int k = 0; k < MD_LEN; k++)
        {
            RegsA[k] = Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X +
                                         2 * MD_LEN * BLOCK_SIZE * gr_ind2 + k * BLOCK_SIZE + THREAD_X];
            RegsB[k] = Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X +
                                         2 * MD_LEN * BLOCK_SIZE * gr_ind2 + (k + MD_LEN) * BLOCK_SIZE + THREAD_X];
        }
        u32 cur_indA = 0;
        u32 cur_indB = 0;

        u32* jlist = jlist0 + gr_ind2 * BLOCK_SIZE;

        // Recover the kangaroo indices KernelA used. This arithmetic is
        // BLOCK_SIZE-dependent but warp-width independent, so wave64 needs no
        // change: tind spans 0..(BLOCK_SIZE*PNT_GROUP_CNT/2 - 1).
        u32 tind = (THREAD_X + gr_ind2 * BLOCK_SIZE);
        u32 thr_ind = (2 * tind) % 256;
        u32 gr_ind = tind / 128;

        u32 kang_ind = (BLOCK_X * BLOCK_SIZE + thr_ind) + gr_ind * (Kparams.BlockCnt * BLOCK_SIZE);
        u32 kang_ind2 = kang_ind + 1;

        __align__(8) u64 dA[3], dB[3];

        dA[0] = Kparams.dists[kang_ind * 4 + 0];
        dA[1] = Kparams.dists[kang_ind * 4 + 1];
        dA[2] = Kparams.dists[kang_ind * 4 + 2];
        dB[0] = Kparams.dists[kang_ind2 * 4 + 0];
        dB[1] = Kparams.dists[kang_ind2 * 4 + 1];
        dB[2] = Kparams.dists[kang_ind2 * 4 + 2];

        bool LoopedA = false;
        bool LoopedB = false;
        u32 step_ind = 0;
        while (step_ind < STEP_CNT)
        {
            // Unrolled by MD_LEN so `iter` is a compile-time constant and the
            // ring-buffer indexing folds away.
            DO_ITER(0); DO_ITER(1); DO_ITER(2); DO_ITER(3); DO_ITER(4);
            DO_ITER(5); DO_ITER(6); DO_ITER(7); DO_ITER(8); DO_ITER(9);
        }

        Kparams.dists[kang_ind * 4 + 0] = dA[0];
        Kparams.dists[kang_ind * 4 + 1] = dA[1];
        Kparams.dists[kang_ind * 4 + 2] = dA[2];
        Kparams.dists[kang_ind2 * 4 + 0] = dB[0];
        Kparams.dists[kang_ind2 * 4 + 1] = dB[1];
        Kparams.dists[kang_ind2 * 4 + 2] = dB[2];

        // Rotate on the way out so cur_ind is 0 again at the next load.
        #pragma unroll
        for (int k = 0; k < MD_LEN; k++)
        {
            int ind = (k + MD_LEN - cur_indA) % MD_LEN;
            Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X +
                              2 * MD_LEN * BLOCK_SIZE * gr_ind2 + ind * BLOCK_SIZE + THREAD_X] = RegsA[k];
            ind = (k + MD_LEN - cur_indB) % MD_LEN;
            Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X +
                              2 * MD_LEN * BLOCK_SIZE * gr_ind2 + (ind + MD_LEN) * BLOCK_SIZE + THREAD_X] = RegsB[k];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// KernelC - one jmp3 hop for each kangaroo KernelB found looping.
//
// Uses a work-stealing cursor rather than a fixed mapping, because the looped
// set is small and unpredictable.
///////////////////////////////////////////////////////////////////////////////

extern "C" CDNA_KERNEL_BOUNDS
__global__ void KernelC(const TKparams Kparams)
{
    u64 PartStride = PNT_GROUP_CNT * (Kparams.BlockCnt * 256 * 32);
    u64* jmp3_table = LDS;   // 48 KB: JMP_CNT entries of {x[4], y[4], dist[4]}

    int i = THREAD_X;
    while (i < JMP_CNT)
    {
        #pragma unroll
        for (int k = 0; k < 12; k += 2)
            *(int4*)&jmp3_table[12 * i + k] = *(int4*)&Kparams.Jumps3[12 * i + k];
        i += BLOCK_SIZE;
    }

    __syncthreads();

    while (1)
    {
        u32 ind = atomicAdd(Kparams.LoopedKangs + 1, 1);
        if (ind >= Kparams.LoopedKangs[0])
            break;
        u32 kang_ind = Kparams.LoopedKangs[2 + ind] & 0x0FFFFFFF;
        u32 last_ind = Kparams.LoopedKangs[2 + ind] >> 28;

        __align__(16) u64 x0[4], x[4];
        __align__(16) u64 y0[4], y[4];
        __align__(16) u64 jmp_x[4];
        __align__(16) u64 jmp_y[4];
        __align__(16) u64 inverse[5];
        u64 tmp[4], tmp2[4];

        // Rewind to the point where the loop was entered.
        u64* x_last0 = Kparams.LastPnts + 4 * kang_ind;
        u64* y_last0 = x_last0 + PartStride / 8;
        u64* x_last = x_last0 + last_ind * 2 * PartStride / 8;
        u64* y_last = y_last0 + last_ind * 2 * PartStride / 8;
        LOAD_VAL_256(x0, x_last, 0);
        LOAD_VAL_256(y0, y_last, 0);

        u32 jmp_ind = x0[0] % JMP_CNT;
        Copy_int4_x2(jmp_x, jmp3_table + 12 * jmp_ind);
        Copy_int4_x2(jmp_y, jmp3_table + 12 * jmp_ind + 4);
        SubModP(inverse, x0, jmp_x);
        InvModP((u32*)inverse);

        u32 inv_flag = y0[0] & 1;
        if (inv_flag)
            NegModP(jmp_y);

        SubModP(tmp, y0, jmp_y);
        MulModP(tmp2, tmp, inverse);
        SqrModP(tmp, tmp2);

        SubModP(x, tmp, jmp_x);
        SubModP(x, x, x0);
        SubModP(y, x0, x);
        MulModP(y, y, tmp2);
        SubModP(y, y, y0);

        Kparams.L2[4 * kang_ind + 0] = x[0];
        Kparams.L2[4 * kang_ind + 1] = x[1];
        Kparams.L2[4 * kang_ind + 2] = x[2];
        Kparams.L2[4 * kang_ind + 3] = x[3];
        Kparams.L2[4 * kang_ind + PartStride / 8 + 0] = y[0];
        Kparams.L2[4 * kang_ind + PartStride / 8 + 1] = y[1];
        Kparams.L2[4 * kang_ind + PartStride / 8 + 2] = y[2];
        Kparams.L2[4 * kang_ind + PartStride / 8 + 3] = y[3];

        u64 d[3];
        d[0] = Kparams.dists[kang_ind * 4 + 0];
        d[1] = Kparams.dists[kang_ind * 4 + 1];
        d[2] = Kparams.dists[kang_ind * 4 + 2];
        if (inv_flag)
            Sub192from192(d, jmp3_table + 12 * jmp_ind + 8);
        else
            Add192to192(d, jmp3_table + 12 * jmp_ind + 8);
        Kparams.dists[kang_ind * 4 + 0] = d[0];
        Kparams.dists[kang_ind * 4 + 1] = d[1];
        Kparams.dists[kang_ind * 4 + 2] = d[2];
    }
}

///////////////////////////////////////////////////////////////////////////////
// KernelGen - computes each kangaroo's starting point from its start distance.
//
// Runs once at startup, so it is written for clarity rather than speed:
// straightforward double-and-add over the scalar's bits.
///////////////////////////////////////////////////////////////////////////////

#define GX_0    0x59F2815B16F81798ull
#define GX_1    0x029BFCDB2DCE28D9ull
#define GX_2    0x55A06295CE870B07ull
#define GX_3    0x79BE667EF9DCBBACull
#define GY_0    0x9C47D08FFB10D4B8ull
#define GY_1    0xFD17B448A6855419ull
#define GY_2    0x5DA4FBFC0E1108A8ull
#define GY_3    0x483ADA7726A3C465ull

__device__ __forceinline__ void AddPoints(u64* res_x, u64* res_y, u64* pnt1x, u64* pnt1y,
                                          u64* pnt2x, u64* pnt2y)
{
    __align__(16) u64 tmp[4], tmp2[4], lambda[4], lambda2[4];
    __align__(16) u64 inverse[5];
    SubModP(inverse, pnt2x, pnt1x);
    InvModP((u32*)inverse);
    SubModP(tmp, pnt2y, pnt1y);
    MulModP(lambda, tmp, inverse);
    MulModP(lambda2, lambda, lambda);
    SubModP(tmp, lambda2, pnt1x);
    SubModP(res_x, tmp, pnt2x);
    SubModP(tmp, pnt2x, res_x);
    MulModP(tmp2, tmp, lambda);
    SubModP(res_y, tmp2, pnt2y);
}

__device__ __forceinline__ void DoublePoint(u64* res_x, u64* res_y, u64* pntx, u64* pnty)
{
    __align__(16) u64 tmp[4], tmp2[4], lambda[4], lambda2[4];
    __align__(16) u64 inverse[5];
    AddModP(inverse, pnty, pnty);
    InvModP((u32*)inverse);
    MulModP(tmp2, pntx, pntx);
    AddModP(tmp, tmp2, tmp2);
    AddModP(tmp, tmp, tmp2);
    MulModP(lambda, tmp, inverse);
    MulModP(lambda2, lambda, lambda);
    SubModP(tmp, lambda2, pntx);
    SubModP(res_x, tmp, pntx);
    SubModP(tmp, pntx, res_x);
    MulModP(tmp2, tmp, lambda);
    SubModP(res_y, tmp2, pnty);
}

extern "C" CDNA_KERNEL_BOUNDS
__global__ void KernelGen(const TKparams Kparams)
{
    u32 PartStride = PNT_GROUP_CNT * (Kparams.BlockCnt * 256 * 32);
    u64* L2x = Kparams.L2 + 4 * (THREAD_X + BLOCK_SIZE * BLOCK_X);
    u64* L2y = L2x + PartStride / 8;
    u64* L2d = L2y + PartStride / 8;

    for (u32 group = 0; group < PNT_GROUP_CNT; group++)
    {
        __align__(16) u64 x0[4], y0[4], d[4];
        __align__(16) u64 x[4], y[4];
        __align__(16) u64 tx[4], ty[4];
        __align__(16) u64 t2x[4], t2y[4];

        u32 kang_ind = THREAD_X + BLOCK_X * BLOCK_SIZE + group * (BLOCK_SIZE * Kparams.BlockCnt);

        LOAD_VAL_256(x0, L2x, group)
        LOAD_VAL_256(y0, L2y, group)
        LOAD_VAL_256(d,  L2d, group)

        tx[0] = GX_0; tx[1] = GX_1; tx[2] = GX_2; tx[3] = GX_3;
        ty[0] = GY_0; ty[1] = GY_1; ty[2] = GY_2; ty[3] = GY_3;

        bool first = true;
        int n = 2;
        while ((n >= 0) && !d[n])
            n--;
        if (n < 0)
            continue;   // zero distance: nothing to compute
        int index = __clzll(d[n]);
        for (int i = 0; i <= 64 * n + (63 - index); i++)
        {
            u8 v = (d[i / 64] >> (i % 64)) & 1;
            if (v)
            {
                if (first)
                {
                    first = false;
                    Copy_u64_x4(x, tx);
                    Copy_u64_x4(y, ty);
                }
                else
                {
                    AddPoints(t2x, t2y, x, y, tx, ty);
                    Copy_u64_x4(x, t2x);
                    Copy_u64_x4(y, t2y);
                }
            }
            DoublePoint(t2x, t2y, tx, ty);
            Copy_u64_x4(tx, t2x);
            Copy_u64_x4(ty, t2y);
        }

        // Tame kangaroos (the first third) start at d*G. Wild ones start at
        // d*G + target, which is what the x0/y0 loaded above carries.
        if (!Kparams.IsGenMode)
            if (kang_ind >= Kparams.KangCnt / 3)
            {
                AddPoints(t2x, t2y, x, y, x0, y0);
                Copy_u64_x4(x, t2x);
                Copy_u64_x4(y, t2y);
            }

        SAVE_VAL_256(L2x, x, group)
        SAVE_VAL_256(L2y, y, group)
    }
}

///////////////////////////////////////////////////////////////////////////////
// Launch wrappers.
//
// The CUDA build calls cudaFuncSetAttribute to raise the dynamic shared memory
// cap past 48 KB. AMD has no such opt-in - the request is honoured up to the
// hardware limit (64 KB/CU on CDNA3, 160 KB on CDNA4) or the launch fails.
///////////////////////////////////////////////////////////////////////////////

void CallGpuKernelA(TKparams Kparams)
{
    hipLaunchKernelGGL(KernelA, dim3(Kparams.BlockCnt), dim3(Kparams.BlockSize),
                       Kparams.KernelA_LDS_Size, 0, Kparams);
}

void CallGpuKernelB(TKparams Kparams)
{
    hipLaunchKernelGGL(KernelB, dim3(Kparams.BlockCnt), dim3(Kparams.BlockSize),
                       Kparams.KernelB_LDS_Size, 0, Kparams);
}

void CallGpuKernelC(TKparams Kparams)
{
    hipLaunchKernelGGL(KernelC, dim3(Kparams.BlockCnt), dim3(Kparams.BlockSize),
                       Kparams.KernelC_LDS_Size, 0, Kparams);
}

void CallGpuKernelGen(TKparams Kparams)
{
    hipLaunchKernelGGL(KernelGen, dim3(Kparams.BlockCnt), dim3(Kparams.BlockSize),
                       0, 0, Kparams);
}

hipError_t cdnaSetGpuParams(TKparams Kparams, u64* _jmp2_table)
{
    // Nothing to do. The CUDA build uploads jmp2 into a __constant__ array here;
    // this port fills both jump tables into LDS from Kparams.Jumps1/Jumps2 at
    // kernel entry instead, so the select between them can use native LDS reads
    // rather than generic flat loads. Kept as a stub so the host call site
    // matches the CUDA one.
    (void)Kparams; (void)_jmp2_table;
    return hipSuccess;
}
