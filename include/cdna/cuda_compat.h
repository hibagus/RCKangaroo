// CUDA-to-HIP compatibility shim for the CDNA port of RCKangaroo.
//
// The host code is shared between the NVIDIA and AMD builds. Roughly 90 of the
// ~100 CUDA call sites are mechanical renames, so rather than fork the host
// sources we map the names here and leave #ifdef seams only where the two
// platforms genuinely behave differently.
//
// The seams that could NOT be handled by renaming, and why:
//
//   * L2 persistence (cudaDeviceSetLimit + cudaStreamSetAttribute with an
//     accessPolicyWindow). hipLimit_t has no cudaLimitPersistingL2CacheSize
//     equivalent, and although hipStreamSetAttribute compiles, it is a no-op on
//     CDNA - MI300/MI350 have no L2 persistence window. Both
//     deviceProp.l2CacheSize and deviceProp.persistingL2CacheMaxSize also read
//     0 or garbage on AMD. The whole block is compiled out; locality comes from
//     the 256 MB Infinity Cache and from non-temporal hints on streaming
//     traffic instead.
//
//   * Architecture dispatch. The CUDA code selects the turbo kernels on
//     compute capability 8.9 / 12.0. deviceProp.major/minor are gfx-derived on
//     ROCm and are not a reliable discriminator, so the AMD path matches on
//     deviceProp.gcnArchName instead.
//
//   * The prebuilt SASS cubins and the CUDA driver API used to load them
//     (cuModuleLoad / cuLaunchKernel / cuFuncSetAttribute). Not portable; the
//     AMD build always uses the compiled HIP kernels.
//
//   * Anything assuming 32-wide warps in host-side sizing. See the Inv_DataSize
//     and StopThr notes in GpuKang.cpp.
//
// License: GPLv3, see "LICENSE.TXT".

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// --- error handling ---------------------------------------------------------
#define cudaError_t                     hipError_t
#define cudaSuccess                     hipSuccess
#define cudaGetErrorString              hipGetErrorString
#define cudaGetLastError                hipGetLastError
#define cudaDeviceSynchronize           hipDeviceSynchronize

// --- device management ------------------------------------------------------
#define cudaGetDeviceCount              hipGetDeviceCount
#define cudaSetDevice                   hipSetDevice
#define cudaGetDeviceProperties         hipGetDeviceProperties
#define cudaDeviceProp                  hipDeviceProp_t
#define cudaRuntimeGetVersion           hipRuntimeGetVersion
#define cudaDriverGetVersion            hipDriverGetVersion
#define cudaSetDeviceFlags              hipSetDeviceFlags
// On ROCm this is documented as a synonym for hipDeviceScheduleYield rather
// than a true blocking sync, so the AMD build does not depend on it.
#define cudaDeviceScheduleBlockingSync  hipDeviceScheduleYield

// --- memory -----------------------------------------------------------------
#define cudaMalloc                      hipMalloc
#define cudaFree                        hipFree
#define cudaMemcpy                      hipMemcpy
#define cudaMemset                      hipMemset
#define cudaMemcpyToSymbol              hipMemcpyToSymbol
#define cudaMemcpyHostToDevice          hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost          hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice        hipMemcpyDeviceToDevice

// --- streams ----------------------------------------------------------------
#define cudaStream_t                    hipStream_t

// The kernels are declared with cdna-prefixed names in src/hip/RCGpuCore.hip so
// the two device implementations can coexist in one tree.
#define cuSetGpuParams                  cdnaSetGpuParams
