#include <hip/hip_runtime.h>

#include <boost/multiprecision/cpp_int.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "field/variants.hpp"

namespace
{
using boost::multiprecision::cpp_int;
using rckangaroo::hip_field::MultiplyVariant;

constexpr int kLimbCount = 8;
constexpr int kVectorCount = 4096;
constexpr int kInverseVectorCount = 256;

struct FieldElement
{
	u32 limb[kLimbCount];
};

struct FieldInput
{
	FieldElement lhs;
	FieldElement rhs;
};

__global__ void RunCarryAdd(const FieldInput* input, FieldElement* output, int count)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count)
		rckangaroo::hip_field::AddCarryChain(
			output[index].limb, input[index].lhs.limb, input[index].rhs.limb);
}

template<MultiplyVariant Variant, bool Square>
__global__ void RunMultiply(const FieldInput* input, FieldElement* output, int count)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count)
		return;
	const u32* rhs = Square ? input[index].lhs.limb : input[index].rhs.limb;
	if constexpr (Variant == MultiplyVariant::CombaMad)
		rckangaroo::hip_field::MultiplyCombaMad(
			output[index].limb, input[index].lhs.limb, rhs);
	else
		rckangaroo::hip_field::MultiplyCombaExplicit(
			output[index].limb, input[index].lhs.limb, rhs);
}

template<MultiplyVariant Variant>
__global__ void RunInverse(const FieldInput* input, FieldElement* output, int count)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count)
		return;
#pragma unroll
	for (int limb = 0; limb < kLimbCount; ++limb)
		output[index].limb[limb] = input[index].lhs.limb[limb];
	rckangaroo::hip_field::InvertAdditionChain<Variant>(output[index].limb);
}

bool CheckHip(hipError_t status, const char* operation)
{
	if (status == hipSuccess)
		return true;
	std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
	return false;
}

cpp_int Prime()
{
	return (cpp_int(1) << 256) - (cpp_int(1) << 32) - 977;
}

void Store(FieldElement& element, cpp_int value)
{
	for (int limb = 0; limb < kLimbCount; ++limb)
	{
		element.limb[limb] = static_cast<u32>(value & 0xFFFFFFFFu);
		value >>= 32;
	}
}

cpp_int Load(const FieldElement& element)
{
	cpp_int value = 0;
	for (int limb = kLimbCount - 1; limb >= 0; --limb)
	{
		value <<= 32;
		value += element.limb[limb];
	}
	return value;
}

u64 NextRandom(u64& state)
{
	state ^= state << 13;
	state ^= state >> 7;
	state ^= state << 17;
	return state;
}

cpp_int RandomFieldElement(u64& state, const cpp_int& prime)
{
	cpp_int value = 0;
	for (int word = 0; word < 4; ++word)
	{
		value <<= 64;
		value += NextRandom(state);
	}
	return value % prime;
}

enum class ExpectedOperation
{
	Add,
	Multiply,
	Square,
	Inverse,
};

bool Verify(const char* name, ExpectedOperation operation, const std::vector<FieldInput>& input,
	const std::vector<FieldElement>& output, int count, const cpp_int& prime)
{
	for (int index = 0; index < count; ++index)
	{
		const cpp_int lhs = Load(input[index].lhs);
		const cpp_int rhs = Load(input[index].rhs);
		cpp_int expected;
		switch (operation)
		{
		case ExpectedOperation::Add:
			expected = (lhs + rhs) % prime;
			break;
		case ExpectedOperation::Multiply:
			expected = (lhs * rhs) % prime;
			break;
		case ExpectedOperation::Square:
			expected = (lhs * lhs) % prime;
			break;
		case ExpectedOperation::Inverse:
			expected = lhs == 0 ? cpp_int(0) : boost::multiprecision::powm(lhs, prime - 2, prime);
			break;
		}
		if (Load(output[index]) != expected)
		{
			std::fprintf(stderr, "%s mismatch at vector %d\n", name, index);
			return false;
		}
	}
	return true;
}
} // namespace

int main()
{
	int device_count = 0;
	if (!CheckHip(hipGetDeviceCount(&device_count), "hipGetDeviceCount") || device_count == 0)
	{
		std::fprintf(stderr, "No HIP device is available.\n");
		return EXIT_FAILURE;
	}
	if (!CheckHip(hipSetDevice(0), "hipSetDevice"))
		return EXIT_FAILURE;

	const cpp_int prime = Prime();
	std::vector<FieldInput> input(kVectorCount);
	std::vector<FieldElement> output(kVectorCount);
	u64 random_state = 0xA0761D6478BD642Full;
	for (int index = 0; index < kVectorCount; ++index)
	{
		cpp_int lhs = RandomFieldElement(random_state, prime);
		cpp_int rhs = RandomFieldElement(random_state, prime);
		if (index < 8)
			lhs = index < 4 ? cpp_int(index) : prime - (index - 3);
		if (index >= 8 && index < 16)
			rhs = index < 12 ? cpp_int(index - 8) : prime - (index - 11);
		if (index == 16)
		{
			lhs = prime - 1;
			rhs = prime - 1;
		}
		Store(input[index].lhs, lhs);
		Store(input[index].rhs, rhs);
	}

	FieldInput* device_input = nullptr;
	FieldElement* device_output = nullptr;
	if (!CheckHip(hipMalloc(&device_input, input.size() * sizeof(FieldInput)), "hipMalloc(input)") ||
		!CheckHip(hipMalloc(&device_output, output.size() * sizeof(FieldElement)), "hipMalloc(output)") ||
		!CheckHip(hipMemcpy(device_input, input.data(), input.size() * sizeof(FieldInput),
			hipMemcpyHostToDevice), "hipMemcpy(input)"))
		return EXIT_FAILURE;

	constexpr int block_size = 256;
	const int block_count = (kVectorCount + block_size - 1) / block_size;
	auto finish_and_verify = [&](const char* name, ExpectedOperation operation, int count)
	{
		if (!CheckHip(hipGetLastError(), name) ||
			!CheckHip(hipDeviceSynchronize(), "kernel synchronize") ||
			!CheckHip(hipMemcpy(output.data(), device_output,
				static_cast<size_t>(count) * sizeof(FieldElement), hipMemcpyDeviceToHost),
				"hipMemcpy(output)"))
			return false;
		return Verify(name, operation, input, output, count, prime);
	};

	bool passed = true;
	hipLaunchKernelGGL(RunCarryAdd, dim3(block_count), dim3(block_size), 0, nullptr,
		device_input, device_output, kVectorCount);
	passed &= finish_and_verify("carry-chain add", ExpectedOperation::Add, kVectorCount);

	hipLaunchKernelGGL((RunMultiply<MultiplyVariant::CombaMad, false>),
		dim3(block_count), dim3(block_size), 0, nullptr,
		device_input, device_output, kVectorCount);
	passed &= finish_and_verify("Comba MAD multiply", ExpectedOperation::Multiply, kVectorCount);
	hipLaunchKernelGGL((RunMultiply<MultiplyVariant::CombaMad, true>),
		dim3(block_count), dim3(block_size), 0, nullptr,
		device_input, device_output, kVectorCount);
	passed &= finish_and_verify("Comba MAD square", ExpectedOperation::Square, kVectorCount);

	hipLaunchKernelGGL((RunMultiply<MultiplyVariant::CombaExplicit, false>),
		dim3(block_count), dim3(block_size), 0, nullptr,
		device_input, device_output, kVectorCount);
	passed &= finish_and_verify("explicit multiply", ExpectedOperation::Multiply, kVectorCount);
	hipLaunchKernelGGL((RunMultiply<MultiplyVariant::CombaExplicit, true>),
		dim3(block_count), dim3(block_size), 0, nullptr,
		device_input, device_output, kVectorCount);
	passed &= finish_and_verify("explicit square", ExpectedOperation::Square, kVectorCount);

	const int inverse_block_count = (kInverseVectorCount + block_size - 1) / block_size;
	hipLaunchKernelGGL((RunInverse<MultiplyVariant::Portable>),
		dim3(inverse_block_count), dim3(block_size), 0, nullptr,
		device_input, device_output, kInverseVectorCount);
	passed &= finish_and_verify("portable addition-chain inverse",
		ExpectedOperation::Inverse, kInverseVectorCount);
	hipLaunchKernelGGL((RunInverse<MultiplyVariant::CombaMad>),
		dim3(inverse_block_count), dim3(block_size), 0, nullptr,
		device_input, device_output, kInverseVectorCount);
	passed &= finish_and_verify("Comba MAD addition-chain inverse",
		ExpectedOperation::Inverse, kInverseVectorCount);
	hipLaunchKernelGGL((RunInverse<MultiplyVariant::CombaExplicit>),
		dim3(inverse_block_count), dim3(block_size), 0, nullptr,
		device_input, device_output, kInverseVectorCount);
	passed &= finish_and_verify("explicit addition-chain inverse",
		ExpectedOperation::Inverse, kInverseVectorCount);

	passed &= CheckHip(hipFree(device_output), "hipFree(output)");
	passed &= CheckHip(hipFree(device_input), "hipFree(input)");
	if (!passed)
		return EXIT_FAILURE;
	std::printf("HIP field variants passed %d arithmetic and %d inversion vectors.\n",
		kVectorCount, kInverseVectorCount);
	return EXIT_SUCCESS;
}
