#pragma once

#include <hip/hip_runtime.h>

#include "rckangaroo/gpu/kernel_params.hpp"

hipError_t SetGpuParams(const TKparams& params, const u64* jump_table_2);
hipError_t LaunchKernelGen(const TKparams& params, hipStream_t stream);
hipError_t LaunchKernelA(const TKparams& params, hipStream_t stream);
hipError_t LaunchKernelB(const TKparams& params, hipStream_t stream);
hipError_t LaunchKernelC(const TKparams& params, hipStream_t stream);
