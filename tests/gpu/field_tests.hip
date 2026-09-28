#include <hip/hip_runtime.h>

#include <boost/multiprecision/cpp_int.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "field/portable.hpp"

namespace
{
using boost::multiprecision::cpp_int;

constexpr int kLimbCount = 8;
constexpr int kVectorCount = 256;

struct FieldInput
{
	u32 lhs[kLimbCount];
	u32 rhs[kLimbCount];
};

struct FieldOutput
{
	u32 add[kLimbCount];
	u32 subtract[kLimbCount];
	u32 multiply[kLimbCount];
	u32 square[kLimbCount];
	u32 inverse[kLimbCount];
};

__global__ void RunFieldOperations(const FieldInput* input, FieldOutput* output, int count)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count)
		return;

	rckangaroo::hip_field::Add(output[index].add, input[index].lhs, input[index].rhs);
	rckangaroo::hip_field::Subtract(output[index].subtract, input[index].lhs, input[index].rhs);
	rckangaroo::hip_field::Multiply(output[index].multiply, input[index].lhs, input[index].rhs);
	rckangaroo::hip_field::Multiply(output[index].square, input[index].lhs, input[index].lhs);
	for (int limb = 0; limb < kLimbCount; ++limb)
		output[index].inverse[limb] = input[index].lhs[limb];
	rckangaroo::hip_field::Invert(output[index].inverse);
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

void Store(std::array<u32, kLimbCount>& limbs, cpp_int value)
{
	for (int limb = 0; limb < kLimbCount; ++limb)
	{
		limbs[limb] = static_cast<u32>(value & 0xFFFFFFFFu);
		value >>= 32;
	}
}

cpp_int Load(const u32* limbs)
{
	cpp_int value = 0;
	for (int limb = kLimbCount - 1; limb >= 0; --limb)
	{
		value <<= 32;
		value += limbs[limb];
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

bool Compare(const char* operation, int index, const u32* actual, const cpp_int& expected)
{
	const cpp_int actual_value = Load(actual);
	if (actual_value == expected)
		return true;
	std::fprintf(stderr, "%s mismatch at vector %d\n", operation, index);
	return false;
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
	std::vector<FieldOutput> output(kVectorCount);
	u64 random_state = 0xD1B54A32D192ED03ull;

	for (int index = 0; index < kVectorCount; ++index)
	{
		cpp_int lhs = RandomFieldElement(random_state, prime);
		cpp_int rhs = RandomFieldElement(random_state, prime);
		if (index == 0)
			lhs = 0;
		else if (index == 1)
			lhs = 1;
		else if (index == 2)
			lhs = prime - 1;
		if (index == 3)
			rhs = prime - 1;

		std::array<u32, kLimbCount> lhs_limbs{};
		std::array<u32, kLimbCount> rhs_limbs{};
		Store(lhs_limbs, lhs);
		Store(rhs_limbs, rhs);
		for (int limb = 0; limb < kLimbCount; ++limb)
		{
			input[index].lhs[limb] = lhs_limbs[limb];
			input[index].rhs[limb] = rhs_limbs[limb];
		}
	}

	FieldInput* device_input = nullptr;
	FieldOutput* device_output = nullptr;
	if (!CheckHip(hipMalloc(&device_input, input.size() * sizeof(FieldInput)), "hipMalloc(input)") ||
		!CheckHip(hipMalloc(&device_output, output.size() * sizeof(FieldOutput)), "hipMalloc(output)"))
		return EXIT_FAILURE;
	if (!CheckHip(hipMemcpy(device_input, input.data(), input.size() * sizeof(FieldInput),
		hipMemcpyHostToDevice), "hipMemcpy(input)"))
		return EXIT_FAILURE;

	constexpr int block_size = 128;
	const int block_count = (kVectorCount + block_size - 1) / block_size;
	hipLaunchKernelGGL(RunFieldOperations, dim3(block_count), dim3(block_size), 0, nullptr,
		device_input, device_output, kVectorCount);
	if (!CheckHip(hipGetLastError(), "RunFieldOperations launch") ||
		!CheckHip(hipDeviceSynchronize(), "RunFieldOperations synchronize") ||
		!CheckHip(hipMemcpy(output.data(), device_output, output.size() * sizeof(FieldOutput),
			hipMemcpyDeviceToHost), "hipMemcpy(output)"))
		return EXIT_FAILURE;

	bool passed = true;
	for (int index = 0; index < kVectorCount; ++index)
	{
		const cpp_int lhs = Load(input[index].lhs);
		const cpp_int rhs = Load(input[index].rhs);
		passed &= Compare("add", index, output[index].add, (lhs + rhs) % prime);
		passed &= Compare("subtract", index, output[index].subtract, (lhs - rhs + prime) % prime);
		passed &= Compare("multiply", index, output[index].multiply, (lhs * rhs) % prime);
		passed &= Compare("square", index, output[index].square, (lhs * lhs) % prime);
		const cpp_int expected_inverse = lhs == 0 ? cpp_int(0) : boost::multiprecision::powm(lhs, prime - 2, prime);
		passed &= Compare("inverse", index, output[index].inverse, expected_inverse);
	}

	passed &= CheckHip(hipFree(device_output), "hipFree(output)");
	passed &= CheckHip(hipFree(device_input), "hipFree(input)");
	if (!passed)
		return EXIT_FAILURE;
	std::printf("Portable HIP field tests passed on %d vectors.\n", kVectorCount);
	return EXIT_SUCCESS;
}
