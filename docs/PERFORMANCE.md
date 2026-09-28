# Performance results

This file records reproducible performance checkpoints. Phase 3 will expand
the metadata collection, profiling artifacts, and multi-GPU methodology.

## Portable HIP baseline: MI355X

- Date: 2026-09-28
- Revision: `319cd94`
- ROCm: 7.2.0
- GPU: AMD Instinct MI355X, `gfx950`, 256 CUs
- Backend: portable HIP 8x32-bit field arithmetic

Build and run:

```sh
cmake --preset mi355x
cmake --build --preset mi355x --target rckangaroo_hip_benchmark -j
build/mi355x/benchmarks/rckangaroo_hip_benchmark 0
```

Result:

```text
GPU 0: AMD Instinct MI355X (gfx950:sramecc+:xnack-), 256 CUs
KernelA portable baseline: 13.807 ms, 113.918 MKeys/s (1572864 kangaroos, one step)
```

The value is the median of five HIP-event samples after one warm-up. Each
sample launches one workgroup per CU, 256 threads per workgroup, and 24
kangaroos per thread. Every kangaroo starts at `G` and uses a table containing
`2G`; host-to-device resets are outside the timed region.

This is a focused `KernelA` baseline, not an end-to-end solver throughput
claim. It excludes `KernelGen`, distance/loop processing in Kernels B/C, host
distinguished-point handling, and transfers. Those measurements belong to
Phase 3. No MI300X performance number has been measured yet.
