#pragma once

#include "field/variants.hpp"

namespace rckangaroo::hip_field
{
constexpr unsigned kWave64Width = 64;

namespace detail
{
__device__ __forceinline__ void SetOne(u32* value)
{
	value[0] = 1;
#pragma unroll
	for (int limb = 1; limb < kLimbs; ++limb)
		value[limb] = 0;
}

__device__ __forceinline__ void SetZero(u32* value)
{
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		value[limb] = 0;
}

__device__ __forceinline__ void ShuffleUp(
	u32* result, const u32* value, unsigned delta)
{
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		result[limb] = __shfl_up(value[limb], delta, kWave64Width);
}

__device__ __forceinline__ void ShuffleDownInPlace(u32* value, unsigned delta)
{
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		value[limb] = __shfl_down(value[limb], delta, kWave64Width);
}

__device__ __forceinline__ void Broadcast(
	u32* result, const u32* value, unsigned source_lane)
{
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		result[limb] = __shfl(value[limb], source_lane, kWave64Width);
}
} // namespace detail

__device__ __forceinline__ void InvertWave64(u32* value, unsigned active_count = kWave64Width)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
	if (warpSize != kWave64Width)
	{
		InvertAdditionChain<MultiplyVariant::CombaMad>(value);
		return;
	}
	if (active_count == 0)
		return;
	active_count = active_count > kWave64Width ? kWave64Width : active_count;

	const unsigned lane = threadIdx.x & (kWave64Width - 1);
	const bool active = lane < active_count;
	const bool zero = active && IsZero(value);
	u32 prefix[kLimbs];
	u32 suffix[kLimbs];
	if (active && !zero)
	{
		detail::Copy(prefix, value);
		detail::Copy(suffix, value);
	}
	else
	{
		detail::SetOne(prefix);
		detail::SetOne(suffix);
	}

	// Inclusive prefix and suffix scans over non-zero active values. Every
	// lane executes every shuffle; inactive and zero lanes contribute one.
#pragma unroll
	for (unsigned offset = 1; offset < kWave64Width; offset <<= 1)
	{
		u32 neighbor[kLimbs];
		detail::ShuffleUp(neighbor, prefix, offset);
		if (active && lane >= offset)
			MultiplyCombaMad(prefix, neighbor, prefix);
	}
#pragma unroll
	for (unsigned offset = 1; offset < kWave64Width; offset <<= 1)
	{
		u32 neighbor[kLimbs];
#pragma unroll
		for (int limb = 0; limb < kLimbs; ++limb)
			neighbor[limb] = __shfl_down(suffix[limb], offset, kWave64Width);
		if (active && lane + offset < active_count)
			MultiplyCombaMad(suffix, suffix, neighbor);
	}

	u32 inverse_total[kLimbs];
	detail::Broadcast(inverse_total, prefix, active_count - 1);
	if (lane == 0)
		InvertAdditionChain<MultiplyVariant::CombaMad>(inverse_total);
	detail::Broadcast(inverse_total, inverse_total, 0);

	u32 prefix_before[kLimbs];
	detail::ShuffleUp(prefix_before, prefix, 1);
	if (lane == 0)
		detail::SetOne(prefix_before);
	detail::ShuffleDownInPlace(suffix, 1);
	if (lane + 1 >= active_count)
		detail::SetOne(suffix);

	MultiplyCombaMad(prefix, prefix_before, suffix);
	MultiplyCombaMad(prefix, prefix, inverse_total);
	if (active)
	{
		if (zero)
			detail::SetZero(value);
		else
			detail::Copy(value, prefix);
	}
#else
	(void)active_count;
	InvertAdditionChain<MultiplyVariant::CombaMad>(value);
#endif
}
} // namespace rckangaroo::hip_field
