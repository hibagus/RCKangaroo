// Differential test for the CDNA field arithmetic in include/hip/RCGpuUtils_cdna.h.
//
// Ground truth comes from Python's arbitrary-precision integers, not from a second
// C++ bignum implementation, so the reference cannot share a bug with the code
// under test. tests/gen_vectors.py writes the operands, this program computes on
// the GPU, and tests/check_results.py verifies.
//
// Usage (see tests/run_diff.sh):
//   gen_vectors.py vec.bin N
//   modarith_diff vec.bin out.bin
//   check_results.py vec.bin out.bin
//
// License: GPLv3.

#include "../include/cdna/RCGpuUtils_cdna.h"
#include <cstdio>
#include <cstdlib>


#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// Per test vector: two 256-bit operands a, b (4 x u64 each, little-endian limbs).
#define OPERAND_U64 4
#define IN_U64  (2 * OPERAND_U64)

// Results, in the order check_results.py expects.
enum { R_MUL, R_SQR, R_SUB, R_ADD, R_NEG, R_INV, R_ADD192, R_SUB192, R_COUNT };
#define OUT_U64 (R_COUNT * OPERAND_U64)

__global__ void run_ops(const u64* __restrict__ in, u64* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    __align__(16) u64 a[4], b[4], r[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) { a[k] = in[i * IN_U64 + k]; b[k] = in[i * IN_U64 + 4 + k]; }

    u64* o = out + (size_t)i * OUT_U64;

    MulModP(r, a, b);                    { for (int k=0;k<4;k++) o[R_MUL*4+k] = r[k]; }
    SqrModP(r, a);                       { for (int k=0;k<4;k++) o[R_SQR*4+k] = r[k]; }
    SubModP(r, a, b);                    { for (int k=0;k<4;k++) o[R_SUB*4+k] = r[k]; }
    AddModP(r, a, b);                    { for (int k=0;k<4;k++) o[R_ADD*4+k] = r[k]; }

    // NegModP is in-place
    { u64 t[4]; Copy_u64_x4(t, a); NegModP(t); for (int k=0;k<4;k++) o[R_NEG*4+k] = t[k]; }

    // InvModP is in-place over 288 bits (9 x u32); needs 5 u64 of storage.
    { __align__(16) u64 t[5]; Copy_u64_x4(t, a); t[4] = 0;
      InvModP((u32*)t);
      for (int k=0;k<4;k++) o[R_INV*4+k] = t[k]; }

    // 192-bit distance accumulation (3 limbs used, 4th written as 0)
    { u64 t[3] = { a[0], a[1], a[2] }; const u64 v[3] = { b[0], b[1], b[2] };
      Add192to192(t, v);
      o[R_ADD192*4+0]=t[0]; o[R_ADD192*4+1]=t[1]; o[R_ADD192*4+2]=t[2]; o[R_ADD192*4+3]=0; }
    { u64 t[3] = { a[0], a[1], a[2] }; const u64 v[3] = { b[0], b[1], b[2] };
      Sub192from192(t, v);
      o[R_SUB192*4+0]=t[0]; o[R_SUB192*4+1]=t[1]; o[R_SUB192*4+2]=t[2]; o[R_SUB192*4+3]=0; }
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <vectors.bin> <results.bin> [device]\n", argv[0]); return 2; }
    int dev = (argc > 3) ? atoi(argv[3]) : 0;
    HIP_CHECK(hipSetDevice(dev));

    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(bytes / (IN_U64 * sizeof(u64)));
    if (n <= 0) { fprintf(stderr, "no vectors in %s\n", argv[1]); return 2; }

    u64* h_in = (u64*)malloc((size_t)n * IN_U64 * sizeof(u64));
    if (fread(h_in, IN_U64 * sizeof(u64), n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);

    u64 *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  (size_t)n * IN_U64  * sizeof(u64)));
    HIP_CHECK(hipMalloc(&d_out, (size_t)n * OUT_U64 * sizeof(u64)));
    HIP_CHECK(hipMemcpy(d_in, h_in, (size_t)n * IN_U64 * sizeof(u64), hipMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(run_ops, dim3(blocks), dim3(threads), 0, 0, d_in, d_out, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    u64* h_out = (u64*)malloc((size_t)n * OUT_U64 * sizeof(u64));
    HIP_CHECK(hipMemcpy(h_out, d_out, (size_t)n * OUT_U64 * sizeof(u64), hipMemcpyDeviceToHost));

    FILE* g = fopen(argv[2], "wb");
    if (!g) { fprintf(stderr, "cannot write %s\n", argv[2]); return 2; }
    fwrite(h_out, OUT_U64 * sizeof(u64), n, g);
    fclose(g);

    printf("computed %d vectors x %d ops -> %s\n", n, R_COUNT, argv[2]);
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    free(h_in); free(h_out);
    return 0;
}
