#pragma once

#include "field/variants.hpp"

__device__ __forceinline__ void AddModP(u64* result, const u64* lhs, const u64* rhs)
{
	rckangaroo::hip_field::AddCarryChain(reinterpret_cast<u32*>(result),
		reinterpret_cast<const u32*>(lhs), reinterpret_cast<const u32*>(rhs));
}

__device__ __forceinline__ void SubModP(u64* result, const u64* lhs, const u64* rhs)
{
	rckangaroo::hip_field::Subtract(reinterpret_cast<u32*>(result),
		reinterpret_cast<const u32*>(lhs), reinterpret_cast<const u32*>(rhs));
}

__device__ __forceinline__ void NegModP(u64* value)
{
	rckangaroo::hip_field::Negate(reinterpret_cast<u32*>(value));
}

__device__ __forceinline__ void MulModP(u64* result, const u64* lhs, const u64* rhs)
{
	rckangaroo::hip_field::MultiplyCombaMad(reinterpret_cast<u32*>(result),
		reinterpret_cast<const u32*>(lhs), reinterpret_cast<const u32*>(rhs));
}

__device__ __forceinline__ void SqrModP(u64* result, const u64* value)
{
	MulModP(result, value, value);
}

__device__ __forceinline__ void InvModP(u32* value)
{
	rckangaroo::hip_field::InvertAdditionChain<
		rckangaroo::hip_field::MultiplyVariant::CombaMad>(value);
}
