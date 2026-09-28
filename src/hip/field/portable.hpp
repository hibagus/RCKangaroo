#pragma once

#include <hip/hip_runtime.h>

#include "rckangaroo/types.hpp"

namespace rckangaroo::hip_field
{
constexpr int kLimbs = 8;
constexpr u32 kReductionConstant = 977;

__device__ __forceinline__ u32 PrimeLimb(int index)
{
	if (index == 0)
		return 0xFFFFFC2Fu;
	if (index == 1)
		return 0xFFFFFFFEu;
	return 0xFFFFFFFFu;
}

__device__ __forceinline__ bool IsZero(const u32* value)
{
	u32 aggregate = 0;
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
		aggregate |= value[i];
	return aggregate == 0;
}

__device__ __forceinline__ bool GreaterOrEqualPrime(const u32* value)
{
#pragma unroll
	for (int i = kLimbs - 1; i >= 0; --i)
	{
		const u32 prime = PrimeLimb(i);
		if (value[i] != prime)
			return value[i] > prime;
	}
	return true;
}

__device__ __forceinline__ void SubtractPrime(u32* value)
{
	u64 borrow = 0;
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
	{
		const u64 subtrahend = static_cast<u64>(PrimeLimb(i)) + borrow;
		const u64 current = value[i];
		value[i] = static_cast<u32>(current - subtrahend);
		borrow = current < subtrahend;
	}
}

__device__ __forceinline__ void AddWord(u32* value, int count, int index, u64 addend)
{
	while (addend != 0 && index < count)
	{
		const u64 sum = static_cast<u64>(value[index]) + static_cast<u32>(addend);
		value[index] = static_cast<u32>(sum);
		addend = (addend >> 32) + (sum >> 32);
		++index;
	}
}

__device__ __forceinline__ void FoldHighWords(u32* value)
{
	u64 accum[11]{};
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
		accum[i] = value[i];

#pragma unroll
	for (int i = kLimbs; i < 11; ++i)
	{
		const u64 high = value[i];
		accum[i - kLimbs] += high * kReductionConstant;
		accum[i - kLimbs + 1] += high;
	}

#pragma unroll
	for (int i = 0; i < 10; ++i)
	{
		accum[i + 1] += accum[i] >> 32;
		value[i] = static_cast<u32>(accum[i]);
	}
	value[10] = static_cast<u32>(accum[10]);
}

__device__ __forceinline__ void ReduceProduct(u32* result, const u32* product)
{
	u64 accum[11]{};
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
	{
		accum[i] += product[i];
		accum[i] += static_cast<u64>(product[i + kLimbs]) * kReductionConstant;
		accum[i + 1] += product[i + kLimbs];
	}

	u32 folded[11]{};
#pragma unroll
	for (int i = 0; i < 10; ++i)
	{
		accum[i + 1] += accum[i] >> 32;
		folded[i] = static_cast<u32>(accum[i]);
	}
	folded[10] = static_cast<u32>(accum[10]);

	// The first fold leaves at most two high words. Fixed extra folds avoid
	// data-dependent control flow while covering the final carry at bit 256.
#pragma unroll
	for (int pass = 0; pass < 4; ++pass)
		FoldHighWords(folded);

#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
		result[i] = folded[i];
	if (GreaterOrEqualPrime(result))
		SubtractPrime(result);
}

__device__ __forceinline__ void Add(u32* result, const u32* lhs, const u32* rhs)
{
	u32 out[kLimbs];
	u64 carry = 0;
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
	{
		const u64 sum = static_cast<u64>(lhs[i]) + rhs[i] + carry;
		out[i] = static_cast<u32>(sum);
		carry = sum >> 32;
	}

	if (carry != 0 || GreaterOrEqualPrime(out))
		SubtractPrime(out);
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
		result[i] = out[i];
}

__device__ __forceinline__ void Subtract(u32* result, const u32* lhs, const u32* rhs)
{
	u32 out[kLimbs];
	u64 borrow = 0;
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
	{
		const u64 subtrahend = static_cast<u64>(rhs[i]) + borrow;
		const u64 current = lhs[i];
		out[i] = static_cast<u32>(current - subtrahend);
		borrow = current < subtrahend;
	}

	if (borrow != 0)
	{
		u64 carry = 0;
#pragma unroll
		for (int i = 0; i < kLimbs; ++i)
		{
			const u64 sum = static_cast<u64>(out[i]) + PrimeLimb(i) + carry;
			out[i] = static_cast<u32>(sum);
			carry = sum >> 32;
		}
	}
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
		result[i] = out[i];
}

__device__ __forceinline__ void Negate(u32* value)
{
	if (IsZero(value))
		return;
	u64 borrow = 0;
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
	{
		const u64 subtrahend = static_cast<u64>(value[i]) + borrow;
		const u64 prime = PrimeLimb(i);
		value[i] = static_cast<u32>(prime - subtrahend);
		borrow = prime < subtrahend;
	}
}

__device__ __forceinline__ void Multiply(u32* result, const u32* lhs, const u32* rhs)
{
	u32 left[kLimbs];
	u32 right[kLimbs];
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
	{
		left[i] = lhs[i];
		right[i] = rhs[i];
	}

	u32 product[16]{};
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
	{
#pragma unroll
		for (int j = 0; j < kLimbs; ++j)
			AddWord(product, 16, i + j, static_cast<u64>(left[i]) * right[j]);
	}
	ReduceProduct(result, product);
}

__device__ __forceinline__ void Invert(u32* value)
{
	constexpr u32 exponent[kLimbs] = {
		0xFFFFFC2Du,
		0xFFFFFFFEu,
		0xFFFFFFFFu,
		0xFFFFFFFFu,
		0xFFFFFFFFu,
		0xFFFFFFFFu,
		0xFFFFFFFFu,
		0xFFFFFFFFu,
	};
	u32 base[kLimbs];
	u32 result[kLimbs] = {1, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
		base[i] = value[i];

	for (int bit = 255; bit >= 0; --bit)
	{
		Multiply(result, result, result);
		if ((exponent[bit / 32] >> (bit % 32)) & 1u)
			Multiply(result, result, base);
	}
#pragma unroll
	for (int i = 0; i < kLimbs; ++i)
		value[i] = result[i];
}

__device__ __forceinline__ void Add192(u64* result, const u64* value)
{
	const u64 r0 = result[0];
	result[0] += value[0];
	u64 carry = result[0] < r0;
	const u64 r1 = result[1];
	result[1] += value[1] + carry;
	carry = result[1] < r1 || (carry != 0 && result[1] == r1);
	result[2] += value[2] + carry;
}

__device__ __forceinline__ void Subtract192(u64* result, const u64* value)
{
	const u64 r0 = result[0];
	result[0] -= value[0];
	u64 borrow = r0 < value[0];
	const u64 r1 = result[1];
	const u64 subtrahend = value[1] + borrow;
	result[1] -= subtrahend;
	borrow = r1 < subtrahend || (borrow != 0 && subtrahend == 0);
	result[2] -= value[2] + borrow;
}
} // namespace rckangaroo::hip_field

#define Add192to192(res, val) { rckangaroo::hip_field::Add192((res), (val)); }
#define Sub192from192(res, val) { rckangaroo::hip_field::Subtract192((res), (val)); }
#define Copy_int4_x2(dst, src) { \
	(dst)[0] = (src)[0]; (dst)[1] = (src)[1]; (dst)[2] = (src)[2]; (dst)[3] = (src)[3]; }
#define Copy_u64_x4(dst, src) Copy_int4_x2((dst), (src))
#define st_cs_b16(addr, val) (*(addr) = static_cast<u16>(val))
