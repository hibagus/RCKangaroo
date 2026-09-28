#pragma once

#include "field/portable.hpp"

namespace rckangaroo::hip_field
{
enum class MultiplyVariant
{
	Portable,
	CombaMad,
	CombaExplicit,
};

namespace detail
{
__device__ __forceinline__ void Copy(u32* destination, const u32* source)
{
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		destination[limb] = source[limb];
}

__device__ __forceinline__ void AccumulateMad(
	u64& accumulator, u32& overflow, u32 lhs, u32 rhs)
{
	// Clang maps this multiply-add idiom to V_MAD_U64_U32 on CDNA3/CDNA4.
	// overflow records the bits above the 64-bit accumulator.
	const u64 next = accumulator + static_cast<u64>(lhs) * rhs;
	overflow += next < accumulator;
	accumulator = next;
}

__device__ __forceinline__ void AccumulateExplicit(
	u64& accumulator, u32& overflow, u32 lhs, u32 rhs)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
	u32 low;
	u32 high;
	u32 carry;
	const u32 accumulator_low = static_cast<u32>(accumulator);
	const u32 accumulator_high = static_cast<u32>(accumulator >> 32);
	asm volatile(
		"v_mul_lo_u32 %0, %5, %6\n\t"
		"v_mul_hi_u32 %1, %5, %6\n\t"
		"v_add_co_u32 %0, vcc, %0, %3\n\t"
		"v_addc_co_u32 %1, vcc, %1, %4, vcc\n\t"
		"v_addc_co_u32 %2, vcc, 0, 0, vcc"
		: "=&v"(low), "=&v"(high), "=&v"(carry)
		: "v"(accumulator_low), "v"(accumulator_high), "v"(lhs), "v"(rhs)
		: "vcc");
	accumulator = static_cast<u64>(low) | (static_cast<u64>(high) << 32);
	overflow += carry;
#else
	AccumulateMad(accumulator, overflow, lhs, rhs);
#endif
}

template<MultiplyVariant Variant>
__device__ __forceinline__ void Accumulate(
	u64& accumulator, u32& overflow, u32 lhs, u32 rhs)
{
	if constexpr (Variant == MultiplyVariant::CombaExplicit)
		AccumulateExplicit(accumulator, overflow, lhs, rhs);
	else
		AccumulateMad(accumulator, overflow, lhs, rhs);
}

template<MultiplyVariant Variant>
__device__ __forceinline__ void MultiplyComba(u32* result, const u32* lhs, const u32* rhs)
{
	u32 left[kLimbs];
	u32 right[kLimbs];
	Copy(left, lhs);
	Copy(right, rhs);

	u32 product[2 * kLimbs]{};
	u64 carry = 0;
#pragma unroll
	for (int column = 0; column < 2 * kLimbs - 1; ++column)
	{
		// At most eight 32x32 products enter a column, so the exact sum is
		// below 2^67. overflow counts 64-bit wraps and fits comfortably in u32.
		u64 accumulator = carry;
		u32 overflow = 0;
#pragma unroll
		for (int lhs_limb = 0; lhs_limb < kLimbs; ++lhs_limb)
		{
			const int rhs_limb = column - lhs_limb;
			if (rhs_limb >= 0 && rhs_limb < kLimbs)
				Accumulate<Variant>(accumulator, overflow,
					left[lhs_limb], right[rhs_limb]);
		}
		product[column] = static_cast<u32>(accumulator);
		carry = (static_cast<u64>(overflow) << 32) | (accumulator >> 32);
	}
	product[2 * kLimbs - 1] = static_cast<u32>(carry);
	ReduceProduct(result, product);
}

template<MultiplyVariant Variant, int Count>
__device__ __forceinline__ void SquareN(u32* value)
{
#pragma unroll
	for (int iteration = 0; iteration < Count; ++iteration)
	{
		u32 squared[kLimbs];
		if constexpr (Variant == MultiplyVariant::Portable)
			Multiply(squared, value, value);
		else
			MultiplyComba<Variant>(squared, value, value);
		Copy(value, squared);
	}
}

template<MultiplyVariant Variant>
__device__ __forceinline__ void MultiplySelected(
	u32* result, const u32* lhs, const u32* rhs)
{
	if constexpr (Variant == MultiplyVariant::Portable)
		Multiply(result, lhs, rhs);
	else
		MultiplyComba<Variant>(result, lhs, rhs);
}

template<MultiplyVariant Variant, int SquareCount>
__device__ __forceinline__ void RaiseAndMultiply(
	u32* result, const u32* value, const u32* multiplier)
{
	Copy(result, value);
	SquareN<Variant, SquareCount>(result);
	MultiplySelected<Variant>(result, result, multiplier);
}
} // namespace detail

__device__ __forceinline__ void AddCarryChain(u32* result, const u32* lhs, const u32* rhs)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
	u32 out[kLimbs];
	u32 carry;
	asm volatile(
		"v_add_co_u32 %0, vcc, %9, %17\n\t"
		"v_addc_co_u32 %1, vcc, %10, %18, vcc\n\t"
		"v_addc_co_u32 %2, vcc, %11, %19, vcc\n\t"
		"v_addc_co_u32 %3, vcc, %12, %20, vcc\n\t"
		"v_addc_co_u32 %4, vcc, %13, %21, vcc\n\t"
		"v_addc_co_u32 %5, vcc, %14, %22, vcc\n\t"
		"v_addc_co_u32 %6, vcc, %15, %23, vcc\n\t"
		"v_addc_co_u32 %7, vcc, %16, %24, vcc\n\t"
		"v_addc_co_u32 %8, vcc, 0, 0, vcc"
		: "=&v"(out[0]), "=&v"(out[1]), "=&v"(out[2]), "=&v"(out[3]),
		  "=&v"(out[4]), "=&v"(out[5]), "=&v"(out[6]), "=&v"(out[7]),
		  "=&v"(carry)
		: "v"(lhs[0]), "v"(lhs[1]), "v"(lhs[2]), "v"(lhs[3]),
		  "v"(lhs[4]), "v"(lhs[5]), "v"(lhs[6]), "v"(lhs[7]),
		  "v"(rhs[0]), "v"(rhs[1]), "v"(rhs[2]), "v"(rhs[3]),
		  "v"(rhs[4]), "v"(rhs[5]), "v"(rhs[6]), "v"(rhs[7])
		: "vcc");
	if (carry != 0 || GreaterOrEqualPrime(out))
		SubtractPrime(out);
	detail::Copy(result, out);
#else
	Add(result, lhs, rhs);
#endif
}

__device__ __forceinline__ void MultiplyCombaMad(
	u32* result, const u32* lhs, const u32* rhs)
{
	detail::MultiplyComba<MultiplyVariant::CombaMad>(result, lhs, rhs);
}

__device__ __forceinline__ void MultiplyCombaExplicit(
	u32* result, const u32* lhs, const u32* rhs)
{
	detail::MultiplyComba<MultiplyVariant::CombaExplicit>(result, lhs, rhs);
}

template<MultiplyVariant Variant>
__device__ __forceinline__ void InvertAdditionChain(u32* value)
{
	// Fixed chain for p - 2. It uses 255 squarings and 15 multiplies instead
	// of a multiply for almost every set exponent bit.
	u32 x[kLimbs];
	u32 x2[kLimbs];
	u32 x3[kLimbs];
	u32 x6[kLimbs];
	u32 x9[kLimbs];
	u32 x11[kLimbs];
	u32 x22[kLimbs];
	u32 x44[kLimbs];
	u32 x88[kLimbs];
	u32 x176[kLimbs];
	u32 x220[kLimbs];
	u32 x223[kLimbs];
	u32 tail[kLimbs];
	detail::Copy(x, value);
	detail::MultiplySelected<Variant>(x2, x, x);
	detail::MultiplySelected<Variant>(x2, x2, x);
	detail::MultiplySelected<Variant>(x3, x2, x2);
	detail::MultiplySelected<Variant>(x3, x3, x);
	detail::RaiseAndMultiply<Variant, 3>(x6, x3, x3);
	detail::RaiseAndMultiply<Variant, 3>(x9, x6, x3);
	detail::RaiseAndMultiply<Variant, 2>(x11, x9, x2);
	detail::RaiseAndMultiply<Variant, 11>(x22, x11, x11);
	detail::RaiseAndMultiply<Variant, 22>(x44, x22, x22);
	detail::RaiseAndMultiply<Variant, 44>(x88, x44, x44);
	detail::RaiseAndMultiply<Variant, 88>(x176, x88, x88);
	detail::RaiseAndMultiply<Variant, 44>(x220, x176, x44);
	detail::RaiseAndMultiply<Variant, 3>(x223, x220, x3);
	detail::RaiseAndMultiply<Variant, 23>(tail, x223, x22);
	detail::RaiseAndMultiply<Variant, 5>(tail, tail, x);
	detail::RaiseAndMultiply<Variant, 3>(tail, tail, x2);
	detail::RaiseAndMultiply<Variant, 2>(tail, tail, x);
	detail::Copy(value, tail);
}
} // namespace rckangaroo::hip_field
