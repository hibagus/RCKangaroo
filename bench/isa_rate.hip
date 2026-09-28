// Phase 0 microbenchmark for the RCKangaroo CDNA port.
//
// Purpose: measure the *achieved issue rate* of the handful of CDNA instructions
// that dominate 256-bit modular arithmetic. The CDNA3/CDNA4 ISA manuals document
// semantics but publish no instruction rates, and the single most important unknown
// for this port is whether V_MAD_U64_U32 issues at full, half or quarter rate.
// That one number moves the projected solver throughput by ~2x.
//
// Method: each test runs a long unrolled sequence of one instruction, emitted via
// inline asm so the measured instruction mix is exactly what we intend (verified
// separately by disassembling the binary). Every test comes in two shapes:
//   LAT  - one dependency chain, so the result is bound by dependent-issue latency
//   THR  - 8 independent chains, so the result is bound by issue throughput
// The gap between them tells us how much ILP the kernel needs to saturate the VALU.
//
// Build:  hipcc -O3 --offload-arch=gfx942 -o isa_rate bench/isa_rate.hip
//
// (c) 2026 - part of the RCKangaroo CDNA3/CDNA4 port. License: GPLv3.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>

typedef unsigned long long u64;
typedef unsigned int u32;

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    return 1; } } while (0)

// Inner loop trip count. Chosen so each kernel runs for a few hundred ms, which
// makes launch overhead and clock ramp-up negligible.
#define ITER 400000

// Instructions emitted per unrolled block, per test. Kept identical across tests
// so the ops/s numbers are directly comparable.
#define UNROLL 8

// ---------------------------------------------------------------------------
// V_MAD_U64_U32 : {carry, D.u64} = S0.u32 * S1.u32 + S2.u64
// The workhorse of the 8x8 32-bit-limb schoolbook multiply. 64 of these per
// 256x256->512 product, which is the theoretical minimum.
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256) void mad_lat(u64* out, u32 seed)
{
    u64 a = out[0] + threadIdx.x + seed;   // opaque to the compiler
    u32 x = (u32)a | 1u, y = (u32)(a >> 32) | 1u;
    for (int i = 0; i < ITER; i++) {
        // 8 mads, each consuming the previous result: pure latency chain.
        asm volatile(
            "v_mad_u64_u32 %0, vcc, %1, %2, %0\n\t"
            "v_mad_u64_u32 %0, vcc, %1, %2, %0\n\t"
            "v_mad_u64_u32 %0, vcc, %1, %2, %0\n\t"
            "v_mad_u64_u32 %0, vcc, %1, %2, %0\n\t"
            "v_mad_u64_u32 %0, vcc, %1, %2, %0\n\t"
            "v_mad_u64_u32 %0, vcc, %1, %2, %0\n\t"
            "v_mad_u64_u32 %0, vcc, %1, %2, %0\n\t"
            "v_mad_u64_u32 %0, vcc, %1, %2, %0\n\t"
            : "+v"(a) : "v"(x), "v"(y) : "vcc");
    }
    if (a == 0x123456789ull) out[threadIdx.x] = a;  // never true; keeps 'a' live
}

__global__ __launch_bounds__(256) void mad_thr(u64* out, u32 seed)
{
    u64 base = out[0] + threadIdx.x + seed;
    u32 x = (u32)base | 1u, y = (u32)(base >> 32) | 1u;
    u64 a0=base, a1=base+1, a2=base+2, a3=base+3;
    u64 a4=base+4, a5=base+5, a6=base+6, a7=base+7;
    for (int i = 0; i < ITER; i++) {
        // 8 independent accumulators: nothing to wait on, pure throughput.
        asm volatile(
            "v_mad_u64_u32 %0, vcc, %8, %9, %0\n\t"
            "v_mad_u64_u32 %1, vcc, %8, %9, %1\n\t"
            "v_mad_u64_u32 %2, vcc, %8, %9, %2\n\t"
            "v_mad_u64_u32 %3, vcc, %8, %9, %3\n\t"
            "v_mad_u64_u32 %4, vcc, %8, %9, %4\n\t"
            "v_mad_u64_u32 %5, vcc, %8, %9, %5\n\t"
            "v_mad_u64_u32 %6, vcc, %8, %9, %6\n\t"
            "v_mad_u64_u32 %7, vcc, %8, %9, %7\n\t"
            : "+v"(a0),"+v"(a1),"+v"(a2),"+v"(a3),
              "+v"(a4),"+v"(a5),"+v"(a6),"+v"(a7)
            : "v"(x), "v"(y) : "vcc");
    }
    u64 s = a0^a1^a2^a3^a4^a5^a6^a7;
    if (s == 0x123456789ull) out[threadIdx.x] = s;
}

// ---------------------------------------------------------------------------
// V_ADD_CO_U32 / V_ADDC_CO_U32 : the carry chain.
// 132 of these per 256-bit multiply in the compiler's output, so their rate
// matters as much as the mad rate. Note the CDNA3 ISA hazard table states
// 0 required wait states for "VALU writes VCC -> VALU reads VCC as carry-in",
// so a serial carry chain should NOT stall. LAT vs THR here tests that claim.
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256) void addc_lat(u64* out, u32 seed)
{
    u32 a = (u32)out[0] + threadIdx.x + seed;
    u32 b = a | 1u;
    for (int i = 0; i < ITER; i++) {
        // One 8-long carry chain. Each addc depends on both the previous
        // sum (VGPR) and the previous carry (VCC).
        asm volatile(
            "v_add_co_u32  %0, vcc, %1, %0\n\t"
            "v_addc_co_u32 %0, vcc, %1, %0, vcc\n\t"
            "v_addc_co_u32 %0, vcc, %1, %0, vcc\n\t"
            "v_addc_co_u32 %0, vcc, %1, %0, vcc\n\t"
            "v_addc_co_u32 %0, vcc, %1, %0, vcc\n\t"
            "v_addc_co_u32 %0, vcc, %1, %0, vcc\n\t"
            "v_addc_co_u32 %0, vcc, %1, %0, vcc\n\t"
            "v_addc_co_u32 %0, vcc, %1, %0, vcc\n\t"
            : "+v"(a) : "v"(b) : "vcc");
    }
    if (a == 0x12345678u) out[threadIdx.x] = a;
}

__global__ __launch_bounds__(256) void addc_thr(u64* out, u32 seed)
{
    u32 base = (u32)out[0] + threadIdx.x + seed;
    u32 b = base | 1u;
    u32 a0=base, a1=base+1, a2=base+2, a3=base+3;
    u32 a4=base+4, a5=base+5, a6=base+6, a7=base+7;
    for (int i = 0; i < ITER; i++) {
        // 8 independent carry chains, each using a distinct SGPR pair for carry
        // so they do not serialize through VCC. This is the pattern the ported
        // Comba multiply will use (RC does the same with Pt0..Pt4 on NVIDIA).
        asm volatile(
            "v_add_co_u32 %0, s[20:21], %8, %0\n\t"
            "v_add_co_u32 %1, s[22:23], %8, %1\n\t"
            "v_add_co_u32 %2, s[24:25], %8, %2\n\t"
            "v_add_co_u32 %3, s[26:27], %8, %3\n\t"
            "v_add_co_u32 %4, s[28:29], %8, %4\n\t"
            "v_add_co_u32 %5, s[30:31], %8, %5\n\t"
            "v_add_co_u32 %6, s[34:35], %8, %6\n\t"
            "v_add_co_u32 %7, s[36:37], %8, %7\n\t"
            : "+v"(a0),"+v"(a1),"+v"(a2),"+v"(a3),
              "+v"(a4),"+v"(a5),"+v"(a6),"+v"(a7)
            : "v"(b)
            : "s20","s21","s22","s23","s24","s25","s26","s27",
              "s28","s29","s30","s31","s34","s35","s36","s37");
    }
    u32 s = a0^a1^a2^a3^a4^a5^a6^a7;
    if (s == 0x12345678u) out[threadIdx.x] = s;
}

// ---------------------------------------------------------------------------
// V_ADD_U32 : plain full-rate integer add. This is the calibration reference -
// everything else is reported relative to this, which cancels out clock
// uncertainty and tells us the rate ratio directly.
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256) void add_thr(u64* out, u32 seed)
{
    u32 base = (u32)out[0] + threadIdx.x + seed;
    u32 b = base | 1u;
    u32 a0=base, a1=base+1, a2=base+2, a3=base+3;
    u32 a4=base+4, a5=base+5, a6=base+6, a7=base+7;
    for (int i = 0; i < ITER; i++) {
        asm volatile(
            "v_add_u32 %0, %8, %0\n\t"
            "v_add_u32 %1, %8, %1\n\t"
            "v_add_u32 %2, %8, %2\n\t"
            "v_add_u32 %3, %8, %3\n\t"
            "v_add_u32 %4, %8, %4\n\t"
            "v_add_u32 %5, %8, %5\n\t"
            "v_add_u32 %6, %8, %6\n\t"
            "v_add_u32 %7, %8, %7\n\t"
            : "+v"(a0),"+v"(a1),"+v"(a2),"+v"(a3),
              "+v"(a4),"+v"(a5),"+v"(a6),"+v"(a7)
            : "v"(b));
    }
    u32 s = a0^a1^a2^a3^a4^a5^a6^a7;
    if (s == 0x12345678u) out[threadIdx.x] = s;
}

// ---------------------------------------------------------------------------
// V_ACCVGPR_WRITE / V_ACCVGPR_READ : the AGPR round trip.
// On gfx942/gfx950 AGPRs are NOT valid VALU operands, so the 256 AccVGPRs are
// only usable as software-managed scratch via these moves. If they are full
// rate, they are a far cheaper spill target than scratch memory - relevant for
// InvModP's ~78 live u32 and SqrModP's 28-entry cross-product array.
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256) void agpr_thr(u64* out, u32 seed)
{
    u32 a0 = (u32)out[0] + threadIdx.x + seed;
    u32 a1=a0+1, a2=a0+2, a3=a0+3;
    for (int i = 0; i < ITER; i++) {
        // 4 write+read pairs = 8 instructions, matching UNROLL.
        asm volatile(
            "v_accvgpr_write_b32 a0, %0\n\t"
            "v_accvgpr_write_b32 a1, %1\n\t"
            "v_accvgpr_write_b32 a2, %2\n\t"
            "v_accvgpr_write_b32 a3, %3\n\t"
            "v_accvgpr_read_b32  %0, a0\n\t"
            "v_accvgpr_read_b32  %1, a1\n\t"
            "v_accvgpr_read_b32  %2, a2\n\t"
            "v_accvgpr_read_b32  %3, a3\n\t"
            : "+v"(a0),"+v"(a1),"+v"(a2),"+v"(a3)
            : : "a0","a1","a2","a3");
    }
    u32 s = a0^a1^a2^a3;
    if (s == 0x12345678u) out[threadIdx.x] = s;
}

// ---------------------------------------------------------------------------
// DS_READ_B128 with a fully divergent per-lane index.
// This is the jump-table gather in KernelA: every lane picks an independent
// random entry out of a 32 KB table, so it is the worst case for LDS bank
// conflicts (32 banks on CDNA3, 64 on CDNA4). If this is the real bottleneck
// rather than the VALU, shrinking JMP_CNT becomes the highest-value tuning knob.
// ---------------------------------------------------------------------------

#define LDS_TABLE_U64 4096          // 4096 * 8 B = 32 KB, same as jmp1_table

__global__ __launch_bounds__(256) void lds_gather(u64* out, u32 seed)
{
    __shared__ u64 tbl[LDS_TABLE_U64];
    for (u32 i = threadIdx.x; i < LDS_TABLE_U64; i += 256) tbl[i] = i * 0x9E3779B97F4A7C15ull;
    __syncthreads();

    // The address operand of ds_* is a 32-bit byte offset inside the LDS aperture,
    // which is exactly what casting a __shared__ pointer to u32 yields on AMDGPU.
    const u32 base = (u32)(size_t)(void*)tbl;

    u32 rng = threadIdx.x * 2654435761u + seed + 1u;
    u32 acc = 0;
    for (int i = 0; i < ITER / 8; i++) {
        // 8 independent 128-bit LDS reads at fully divergent per-lane addresses.
        // Emitted as inline asm because the compiler will happily narrow a
        // C-level int4 read down to ds_read2_b32 if it thinks it can.
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            // 64 B-aligned entry, same granularity as one jmp1_table record
            u32 off = base + ((rng & (LDS_TABLE_U64 / 8 - 1)) << 6);
            u32 lo;
            asm volatile(
                "ds_read_b128 v[8:11], %1\n\t"
                "s_waitcnt lgkmcnt(0)\n\t"
                "v_mov_b32 %0, v8\n\t"
                : "=v"(lo) : "v"(off)
                : "v8","v9","v10","v11");
            acc ^= lo;
        }
    }
    if (acc == 0x1234567u) out[threadIdx.x] = acc;
}

// ---------------------------------------------------------------------------

struct Result { const char* name; double ops_per_s; u64 ops; double ms; };

template <typename K>
static double time_kernel(K kern, int blocks, int threads, u64* d_out, u64 total_ops, double* ms_out)
{
    // Warm-up: also lets the clocks ramp so we measure sustained, not boost-transient, rate.
    hipLaunchKernelGGL(kern, dim3(blocks), dim3(threads), 0, 0, d_out, 1u);
    (void)hipDeviceSynchronize();

    double ms = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(kern, dim3(blocks), dim3(threads), 0, 0, d_out, 1u);
        (void)hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (t < ms) ms = t;
    }
    *ms_out = ms;
    return (double)total_ops / (ms / 1000.0);
}

int main(int argc, char** argv)
{
    int dev = (argc > 1) ? atoi(argv[1]) : 0;
    HIP_CHECK(hipSetDevice(dev));

    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, dev));

    // One workgroup of 256 threads (4 wave64) per CU, matching how RCKangaroo's
    // KernelA is actually dispatched: 1 persistent block per CU, 1 wave per SIMD.
    const int threads = 256;
    const int blocks  = p.multiProcessorCount;
    const u64 lanes   = (u64)blocks * threads;

    printf("Device %d: %s (%s)\n", dev, p.name, p.gcnArchName);
    printf("  CUs=%d  wave=%d  clock=%.2f GHz  LDS/blk=%zu KB\n",
           p.multiProcessorCount, p.warpSize, p.clockRate / 1e6,
           (size_t)(p.sharedMemPerBlock / 1024));
    printf("  grid: %d blocks x %d threads = %llu lanes (1 workgroup/CU)\n\n",
           blocks, threads, (unsigned long long)lanes);

    // Nominal simple-VALU peak: lanes issue one op per clock per SIMD lane.
    double peak = (double)p.multiProcessorCount * 64.0 * (p.clockRate * 1000.0);
    printf("  nominal simple-VALU peak = %.2f Tops/s\n\n", peak / 1e12);

    u64* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, 4096 * sizeof(u64)));
    HIP_CHECK(hipMemset(d_out, 0, 4096 * sizeof(u64)));

    const u64 ops_valu = lanes * (u64)ITER * UNROLL;
    const u64 ops_lds  = lanes * (u64)(ITER / 8) * 8ull;  // one ds_read_b128 per k-step

    std::vector<Result> rs;
    double ms;

    rs.push_back({"v_add_u32          (THR, 8 chains)", time_kernel(add_thr,   blocks, threads, d_out, ops_valu, &ms), ops_valu, ms});
    rs.push_back({"v_mad_u64_u32      (THR, 8 chains)", time_kernel(mad_thr,   blocks, threads, d_out, ops_valu, &ms), ops_valu, ms});
    rs.push_back({"v_mad_u64_u32      (LAT, 1 chain) ", time_kernel(mad_lat,   blocks, threads, d_out, ops_valu, &ms), ops_valu, ms});
    rs.push_back({"v_add_co_u32 x8 indep SGPR carry  ", time_kernel(addc_thr,  blocks, threads, d_out, ops_valu, &ms), ops_valu, ms});
    rs.push_back({"v_addc_co_u32 serial VCC chain    ", time_kernel(addc_lat,  blocks, threads, d_out, ops_valu, &ms), ops_valu, ms});
    rs.push_back({"v_accvgpr_wr/rd    (THR)          ", time_kernel(agpr_thr,  blocks, threads, d_out, ops_valu, &ms), ops_valu, ms});
    rs.push_back({"ds_read (divergent 32KB gather)  ", time_kernel(lds_gather, blocks, threads, d_out, ops_lds,  &ms), ops_lds,  ms});

    // The v_add_u32 result is the full-rate reference; ratios against it are what
    // we actually care about, since they are immune to clock/measurement error.
    double ref = rs[0].ops_per_s;

    printf("%-36s %10s %8s %9s %s\n", "instruction", "Gops/s", "ms", "vs peak", "vs v_add_u32");
    printf("%-36s %10s %8s %9s %s\n", "-----------", "------", "--", "-------", "------------");
    for (const auto& r : rs) {
        double ratio = r.ops_per_s / ref;
        const char* rate = "";
        if (ratio > 0.75)       rate = "  full rate";
        else if (ratio > 0.35)  rate = "  ~1/2 rate";
        else if (ratio > 0.18)  rate = "  ~1/4 rate";
        else                    rate = "  memory-bound / heavily conflicted";
        printf("%-36s %10.1f %8.1f %8.1f%% %11.3f%s\n",
               r.name, r.ops_per_s / 1e9, r.ms,
               100.0 * r.ops_per_s / peak, ratio, rate);
    }

    HIP_CHECK(hipFree(d_out));
    return 0;
}
