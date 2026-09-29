// Can CDNA4's matrix cores beat the VALU at 256-bit integer multiplication?
//
// The unit of comparison is the 32x32->64 multiply-accumulate, because that is
// what an 8x8 limb schoolbook 256x256->512 product is built from (64 of them).
// A matrix core works in int8, and one 32x32 product decomposes into 16 8x8
// products, so MFMA has to be >16x faster per MAC just to break even.
//
// The answer is no, and not because the matrix cores are slow - see
// docs/CDNA4_MI355X.md. They measure 2.85x the VALU in 32x32 MAC-equivalents,
// but the algorithm's multiplies are rank-1, so MFMA's K or N dimension sits
// almost entirely idle and the effective rate lands far below the VALU.
//
// Note on measuring this: the MFMA intrinsics are pure, so if the accumulator
// is dead the whole loop is eliminated and the kernel "runs" in 0.0 ms. The
// guards below are on a kernel argument for that reason.
//
// Build: hipcc -O3 --offload-arch=gfx950 -o mfma_rate bench/mfma_rate.hip
//
// (c) 2026 - part of the RCKangaroo CDNA3/CDNA4 port. License: GPLv3.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>

typedef unsigned long long u64;
typedef unsigned int u32;
typedef int   v4i  __attribute__((ext_vector_type(4)));
typedef int   v16i __attribute__((ext_vector_type(16)));

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP error %s at %d\n",hipGetErrorString(e),__LINE__); return 1;} } while(0)

#define ITER   200000
#define UNROLL 8

// --- VALU reference: v_mad_u64_u32, 8 independent chains -------------------
__global__ __launch_bounds__(256) void mad_thr(u64* out, u32 seed)
{
    u64 a = out[0] + threadIdx.x + seed;
    u32 x = (u32)a | 1u, y = (u32)(a >> 32) | 1u;
    u64 c0=a,c1=a+1,c2=a+2,c3=a+3,c4=a+4,c5=a+5,c6=a+6,c7=a+7;
    for (int i = 0; i < ITER; i++) {
        asm volatile(
            "v_mad_u64_u32 %0, vcc, %8, %9, %0\n\t" "v_mad_u64_u32 %1, vcc, %8, %9, %1\n\t"
            "v_mad_u64_u32 %2, vcc, %8, %9, %2\n\t" "v_mad_u64_u32 %3, vcc, %8, %9, %3\n\t"
            "v_mad_u64_u32 %4, vcc, %8, %9, %4\n\t" "v_mad_u64_u32 %5, vcc, %8, %9, %5\n\t"
            "v_mad_u64_u32 %6, vcc, %8, %9, %6\n\t" "v_mad_u64_u32 %7, vcc, %8, %9, %7\n\t"
            : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7)
            : "v"(x),"v"(y) : "vcc");
    }
    if (seed == 0xFFFFFFFFu) out[0] = c0+c1+c2+c3+c4+c5+c6+c7;
}

// --- MFMA 16x16x32 int8: 16*16*32 = 8192 int8 MACs per instruction ---------
__global__ __launch_bounds__(256) void mfma_16x16x32(u64* out, u32 seed)
{
    u64 a = out[0] + threadIdx.x + seed;
    long A = (long)a, B = (long)(a ^ 0x5555555555555555ull);
    v4i c0={0,0,0,0},c1=c0,c2=c0,c3=c0,c4=c0,c5=c0,c6=c0,c7=c0;
    for (int i = 0; i < ITER; i++) {
        c0 = __builtin_amdgcn_mfma_i32_16x16x32_i8(A,B,c0,0,0,0);
        c1 = __builtin_amdgcn_mfma_i32_16x16x32_i8(A,B,c1,0,0,0);
        c2 = __builtin_amdgcn_mfma_i32_16x16x32_i8(A,B,c2,0,0,0);
        c3 = __builtin_amdgcn_mfma_i32_16x16x32_i8(A,B,c3,0,0,0);
        c4 = __builtin_amdgcn_mfma_i32_16x16x32_i8(A,B,c4,0,0,0);
        c5 = __builtin_amdgcn_mfma_i32_16x16x32_i8(A,B,c5,0,0,0);
        c6 = __builtin_amdgcn_mfma_i32_16x16x32_i8(A,B,c6,0,0,0);
        c7 = __builtin_amdgcn_mfma_i32_16x16x32_i8(A,B,c7,0,0,0);
    }
    if (seed == 0xFFFFFFFFu) out[0] = c0[0]+c1[0]+c2[0]+c3[0]+c4[0]+c5[0]+c6[0]+c7[0];
}

// --- MFMA 32x32x16 int8: 32*32*16 = 16384 int8 MACs per instruction --------
__global__ __launch_bounds__(256) void mfma_32x32x16(u64* out, u32 seed)
{
    u64 a = out[0] + threadIdx.x + seed;
    long A = (long)a, B = (long)(a ^ 0x5555555555555555ull);
    v16i c0={0}, c1={0}, c2={0}, c3={0};
    for (int i = 0; i < ITER; i++) {
        c0 = __builtin_amdgcn_mfma_i32_32x32x16_i8(A,B,c0,0,0,0);
        c1 = __builtin_amdgcn_mfma_i32_32x32x16_i8(A,B,c1,0,0,0);
        c2 = __builtin_amdgcn_mfma_i32_32x32x16_i8(A,B,c2,0,0,0);
        c3 = __builtin_amdgcn_mfma_i32_32x32x16_i8(A,B,c3,0,0,0);
    }
    if (seed == 0xFFFFFFFFu) out[0] = c0[0]+c1[0]+c2[0]+c3[0];
}

// --- v_dot4_i32_i8: 4 int8 MACs per lane, no cross-lane layout needed ------
__global__ __launch_bounds__(256) void dot4_thr(u64* out, u32 seed)
{
    u64 a = out[0] + threadIdx.x + seed;
    u32 x = (u32)a | 1u, y = (u32)(a >> 32) | 1u;
    u32 c0=x,c1=x+1,c2=x+2,c3=x+3,c4=x+4,c5=x+5,c6=x+6,c7=x+7;
    for (int i = 0; i < ITER; i++) {
        asm volatile(
            "v_dot4_i32_i8 %0, %8, %9, %0\n\t" "v_dot4_i32_i8 %1, %8, %9, %1\n\t"
            "v_dot4_i32_i8 %2, %8, %9, %2\n\t" "v_dot4_i32_i8 %3, %8, %9, %3\n\t"
            "v_dot4_i32_i8 %4, %8, %9, %4\n\t" "v_dot4_i32_i8 %5, %8, %9, %5\n\t"
            "v_dot4_i32_i8 %6, %8, %9, %6\n\t" "v_dot4_i32_i8 %7, %8, %9, %7\n\t"
            : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7)
            : "v"(x),"v"(y));
    }
    if (seed == 0xFFFFFFFFu) out[0] = c0+c1+c2+c3+c4+c5+c6+c7;
}

template <class K>
double timeit(K k, u64* d, int blocks, int thr, double& ms)
{
    k<<<blocks,thr>>>(d,1); HIP_CHECK(hipDeviceSynchronize());   // warm up
    auto t0 = std::chrono::high_resolution_clock::now();
    k<<<blocks,thr>>>(d,1); HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    return ms;
}

int main(int argc, char** argv)
{
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t p; HIP_CHECK(hipGetDeviceProperties(&p, dev));
    printf("Device %d: %s (%s)  CUs=%d  clock=%.2f GHz\n\n",
           dev, p.name, p.gcnArchName, p.multiProcessorCount, p.clockRate/1e6);

    u64* d; HIP_CHECK(hipMalloc(&d, 8)); HIP_CHECK(hipMemset(d, 0, 8));
    int blocks = p.multiProcessorCount, thr = 256;
    double waves = (double)blocks * (thr/64);        // wave64 count
    double ms;

    printf("%-26s %14s %10s %16s\n", "sequence", "int8 MAC/s", "ms", "32x32 MAC-equiv/s");
    printf("%-26s %14s %10s %16s\n", "--------", "----------", "--", "-----------------");

    // v_mad_u64_u32: one 32x32 MAC per lane per instruction
    timeit(mad_thr, d, blocks, thr, ms);
    double mad_ops = (double)ITER*UNROLL*waves*64 / (ms/1e3);
    printf("%-26s %14s %10.1f %16.1f G   <- baseline\n", "v_mad_u64_u32 (THR)", "-", ms, mad_ops/1e9);

    // MFMA: per-wave instruction, 8192 int8 MACs; /16 to convert to 32x32 equivalents
    timeit(mfma_16x16x32, d, blocks, thr, ms);
    double m1 = (double)ITER*UNROLL*waves*8192 / (ms/1e3);
    printf("%-26s %14.1f T %8.1f %16.1f G   (%.2fx baseline)\n",
           "v_mfma_i32_16x16x32_i8", m1/1e12, ms, m1/16/1e9, (m1/16)/mad_ops);

    timeit(mfma_32x32x16, d, blocks, thr, ms);
    double m2 = (double)ITER*4*waves*16384 / (ms/1e3);
    printf("%-26s %14.1f T %8.1f %16.1f G   (%.2fx baseline)\n",
           "v_mfma_i32_32x32x16_i8", m2/1e12, ms, m2/16/1e9, (m2/16)/mad_ops);

    timeit(dot4_thr, d, blocks, thr, ms);
    double d4 = (double)ITER*UNROLL*waves*64*4 / (ms/1e3);
    printf("%-26s %14.1f T %8.1f %16.1f G   (%.2fx baseline)\n",
           "v_dot4_i32_i8 (THR)", d4/1e12, ms, d4/16/1e9, (d4/16)/mad_ops);

    printf("\nA 256x256->512 schoolbook product needs 64 of the 32x32 MAC-equivalents.\n");
    printf("Break-even for a matrix-core multiply is 1.00x, before any cost for\n");
    printf("transposing operands into matrix layout and carry-propagating the result.\n");
    return 0;
}
