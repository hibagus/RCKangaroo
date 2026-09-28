// Isolate the field arithmetic: run the per-point-addition primitive sequence in
// registers with no memory traffic, and see what fraction of VALU peak it reaches.
// If this hits peak, KernelA's shortfall is structural (memory, glue). If it does
// not, the arithmetic's own dependency chains are the limit.
#include <hip/hip_runtime.h>
typedef unsigned long long u64; typedef unsigned int u32; typedef unsigned short u16;
typedef unsigned __int128 u128;
#include "cdna/RCGpuUtils_cdna.h"
#include <cstdio>
#include <chrono>

#define ITER 4000

__global__ __launch_bounds__(256,1) void arith(u64* out, u32 seed)
{
    __align__(16) u64 a[4], b[4], c[4], d[4];
    u64 t = threadIdx.x + blockIdx.x * 256 + seed;
    for (int i = 0; i < 4; i++) { a[i] = t * 0x9E3779B97F4A7C15ull + i; b[i] = t * 0xC2B2AE3D27D4EB4Full + i; }
    for (int k = 0; k < ITER; k++) {
        // the real per-point-addition mix: 5 MulModP, 1 SqrModP, 7 SubModP
        SubModP(c, a, b);
        MulModP(d, a, c);
        SubModP(c, d, b);
        MulModP(a, d, c);
        SubModP(c, a, b);
        MulModP(d, a, c);
        SqrModP(a, d);
        SubModP(c, a, b);
        MulModP(d, a, c);
        SubModP(c, d, b);
        MulModP(a, d, c);
        SubModP(c, a, b);
        SubModP(b, c, a);
    }
    if (a[0] == 0x123456789ull) out[threadIdx.x] = a[0];
}

int main(int argc, char** argv)
{
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    (void)hipSetDevice(dev);
    hipDeviceProp_t p; (void)hipGetDeviceProperties(&p, dev);
    int blocks = p.multiProcessorCount * 3, threads = 256;
    u64* o; (void)hipMalloc(&o, 4096*8); (void)hipMemset(o, 0, 4096*8);
    hipLaunchKernelGGL(arith, dim3(blocks), dim3(threads), 0, 0, o, 1u);
    (void)hipDeviceSynchronize();
    double best = 1e30;
    for (int r = 0; r < 3; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(arith, dim3(blocks), dim3(threads), 0, 0, o, 1u);
        (void)hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        if (ms < best) best = ms;
    }
    double lanes = (double)blocks * threads;
    double point_adds = lanes * ITER;           // one full sequence per iteration
    double secs = best / 1000.0;
    printf("blocks=%d threads=%d iters=%d  time=%.1f ms\n", blocks, threads, ITER, best);
    printf("point-addition sequences/s : %.2f G\n", point_adds/secs/1e9);
    printf("  vs KernelA's achieved     : 10.1 G (per GPU, in the real kernel)\n");
    return 0;
}
