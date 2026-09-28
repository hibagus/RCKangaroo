#include <hip/hip_runtime.h>

#include <boost/multiprecision/cpp_int.hpp>

#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <vector>

#include "field/wave64.hpp"

namespace
{
using boost::multiprecision::cpp_int;

constexpr int kLimbs = 8;
constexpr int kWaveWidth = 64;
constexpr int kBlockCount = 32;
constexpr int kElementCount = kBlockCount * kWaveWidth;

struct FieldElement
{
	u32 limb[kLimbs];
};

__global__ __launch_bounds__(kWaveWidth) void RunWaveInverse(
	FieldElement* values, const unsigned* active_counts)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	rckangaroo::hip_field::InvertWave64(values[index].limb, active_counts[blockIdx.x]);
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
	for (int limb = 0; limb < kLimbs; ++limb)
	{
		element.limb[limb] = static_cast<u32>(value & 0xFFFFFFFFu);
		value >>= 32;
	}
}

cpp_int Load(const FieldElement& element)
{
	cpp_int value = 0;
	for (int limb = kLimbs - 1; limb >= 0; --limb)
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
	hipDeviceProp_t properties{};
	if (!CheckHip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties"))
		return EXIT_FAILURE;
	if (properties.warpSize != kWaveWidth)
	{
		std::fprintf(stderr, "Wave64 test requires warpSize=64, got %d.\n", properties.warpSize);
		return EXIT_FAILURE;
	}

	const cpp_int prime = Prime();
	std::vector<FieldElement> input(kElementCount);
	std::vector<FieldElement> output(kElementCount);
	std::vector<unsigned> active_counts(kBlockCount, kWaveWidth);
	constexpr unsigned edge_counts[] = {1, 2, 3, 7, 16, 31, 32, 33, 47, 63, 64};
	for (size_t index = 0; index < std::size(edge_counts); ++index)
		active_counts[index] = edge_counts[index];

	u64 random_state = 0xE7037ED1A0B428DBull;
	for (int block = 0; block < kBlockCount; ++block)
	{
		for (int lane = 0; lane < kWaveWidth; ++lane)
		{
			cpp_int value = RandomFieldElement(random_state, prime);
			if (lane < static_cast<int>(active_counts[block]) &&
				((block == 0 && lane == 0) || ((block + lane) % 53 == 0)))
				value = 0;
			Store(input[block * kWaveWidth + lane], value);
		}
	}

	FieldElement* device_values = nullptr;
	unsigned* device_active_counts = nullptr;
	if (!CheckHip(hipMalloc(&device_values, input.size() * sizeof(FieldElement)),
			"hipMalloc(values)") ||
		!CheckHip(hipMalloc(&device_active_counts,
			active_counts.size() * sizeof(unsigned)), "hipMalloc(active_counts)") ||
		!CheckHip(hipMemcpy(device_values, input.data(), input.size() * sizeof(FieldElement),
			hipMemcpyHostToDevice), "hipMemcpy(values)") ||
		!CheckHip(hipMemcpy(device_active_counts, active_counts.data(),
			active_counts.size() * sizeof(unsigned), hipMemcpyHostToDevice),
			"hipMemcpy(active_counts)"))
		return EXIT_FAILURE;

	hipLaunchKernelGGL(RunWaveInverse, dim3(kBlockCount), dim3(kWaveWidth), 0, nullptr,
		device_values, device_active_counts);
	if (!CheckHip(hipGetLastError(), "RunWaveInverse launch") ||
		!CheckHip(hipDeviceSynchronize(), "RunWaveInverse synchronize") ||
		!CheckHip(hipMemcpy(output.data(), device_values,
			output.size() * sizeof(FieldElement), hipMemcpyDeviceToHost),
			"hipMemcpy(output)"))
		return EXIT_FAILURE;

	bool passed = true;
	for (int block = 0; block < kBlockCount; ++block)
	{
		for (int lane = 0; lane < kWaveWidth; ++lane)
		{
			const int index = block * kWaveWidth + lane;
			const cpp_int source = Load(input[index]);
			const cpp_int expected = lane < static_cast<int>(active_counts[block])
				? (source == 0 ? cpp_int(0) : boost::multiprecision::powm(source, prime - 2, prime))
				: source;
			if (Load(output[index]) != expected)
			{
				std::fprintf(stderr, "wave inverse mismatch at block %d lane %d (active %u)\n",
					block, lane, active_counts[block]);
				passed = false;
				break;
			}
		}
	}
	passed &= CheckHip(hipFree(device_active_counts), "hipFree(active_counts)");
	passed &= CheckHip(hipFree(device_values), "hipFree(values)");
	if (!passed)
		return EXIT_FAILURE;
	std::printf("Wave64 inversion passed %d values across full and partial waves.\n",
		kElementCount);
	return EXIT_SUCCESS;
}
