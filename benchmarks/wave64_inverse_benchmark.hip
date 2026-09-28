#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "field/wave64.hpp"

namespace
{
constexpr int kLimbs = 8;
constexpr int kThreadsPerBlock = 256;

struct FieldElement
{
	u32 limb[kLimbs];
};

struct Options
{
	int device = 0;
	int samples = 5;
	int iterations = 8;
	int blocks = 0;
};

extern "C" __global__ void InversePerLane(FieldElement* state, int iterations)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	u32 value[kLimbs];
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		value[limb] = state[index].limb[limb];
	for (int iteration = 0; iteration < iterations; ++iteration)
		rckangaroo::hip_field::InvertAdditionChain<
			rckangaroo::hip_field::MultiplyVariant::CombaMad>(value);
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		state[index].limb[limb] = value[limb];
}

extern "C" __global__ void InverseWave64(FieldElement* state, int iterations)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	u32 value[kLimbs];
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		value[limb] = state[index].limb[limb];
	for (int iteration = 0; iteration < iterations; ++iteration)
		rckangaroo::hip_field::InvertWave64(value);
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		state[index].limb[limb] = value[limb];
}

bool CheckHip(hipError_t status, const char* operation)
{
	if (status == hipSuccess)
		return true;
	std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
	return false;
}

bool ParseInteger(const char* text, int minimum, int* value)
{
	char* end = nullptr;
	const long parsed = std::strtol(text, &end, 10);
	if (!text[0] || *end || parsed < minimum || parsed > 1'000'000)
		return false;
	*value = static_cast<int>(parsed);
	return true;
}

bool ParseOptions(int argc, char** argv, Options* options)
{
	for (int index = 1; index < argc; ++index)
	{
		const std::string_view argument = argv[index];
		if (argument == "--help")
			return false;
		if (index + 1 >= argc)
			return false;
		int value = 0;
		if (!ParseInteger(argv[++index], 0, &value))
			return false;
		if (argument == "--device")
			options->device = value;
		else if (argument == "--samples" && value > 0)
			options->samples = value;
		else if (argument == "--iterations" && value > 0)
			options->iterations = value;
		else if (argument == "--blocks" && value > 0)
			options->blocks = value;
		else
			return false;
	}
	return options->samples <= 100;
}

double Median(std::vector<double> values)
{
	std::sort(values.begin(), values.end());
	const size_t middle = values.size() / 2;
	return values.size() & 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

struct Result
{
	const char* name;
	double median_ms;
	double median_ginversions;
	double mad_ginversions;
};
} // namespace

int main(int argc, char** argv)
{
	if (argc == 2 && std::string_view(argv[1]) == "--help")
	{
		std::printf("Usage: %s [--device N] [--samples N] [--iterations N] [--blocks N]\n",
			argv[0]);
		return EXIT_SUCCESS;
	}
	Options options;
	if (!ParseOptions(argc, argv, &options))
	{
		std::fprintf(stderr, "Invalid arguments. Use --help for usage.\n");
		return EXIT_FAILURE;
	}
	if (!CheckHip(hipSetDevice(options.device), "hipSetDevice"))
		return EXIT_FAILURE;
	hipDeviceProp_t properties{};
	if (!CheckHip(hipGetDeviceProperties(&properties, options.device), "hipGetDeviceProperties"))
		return EXIT_FAILURE;
	if (properties.warpSize != 64)
	{
		std::fprintf(stderr, "Wave64 benchmark requires warpSize=64, got %d.\n", properties.warpSize);
		return EXIT_FAILURE;
	}

	const int block_count = options.blocks ? options.blocks : properties.multiProcessorCount;
	const int element_count = block_count * kThreadsPerBlock;
	std::vector<FieldElement> initial(element_count);
	constexpr u32 generator_x[kLimbs] = {
		0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
		0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu};
	for (int index = 0; index < element_count; ++index)
	{
		for (int limb = 0; limb < kLimbs; ++limb)
			initial[index].limb[limb] = generator_x[limb];
		initial[index].limb[0] ^= static_cast<u32>(index * 0x9E3779B9u);
	}

	FieldElement* device_state = nullptr;
	const size_t bytes = initial.size() * sizeof(FieldElement);
	hipStream_t stream = nullptr;
	hipEvent_t start = nullptr;
	hipEvent_t stop = nullptr;
	if (!CheckHip(hipMalloc(&device_state, bytes), "hipMalloc(state)") ||
		!CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreate") ||
		!CheckHip(hipEventCreate(&start), "hipEventCreate(start)") ||
		!CheckHip(hipEventCreate(&stop), "hipEventCreate(stop)"))
		return EXIT_FAILURE;

	std::vector<Result> results;
	const dim3 grid(block_count);
	const dim3 block(kThreadsPerBlock);
	auto measure = [&](const char* name, auto launch)
	{
		if (!CheckHip(hipMemcpyAsync(device_state, initial.data(), bytes,
			hipMemcpyHostToDevice, stream), "hipMemcpyAsync(reset)"))
			return false;
		launch();
		if (!CheckHip(hipGetLastError(), "warm-up launch") ||
			!CheckHip(hipStreamSynchronize(stream), "warm-up synchronize"))
			return false;

		std::vector<double> milliseconds;
		std::vector<double> rates;
		for (int sample = 0; sample < options.samples; ++sample)
		{
			if (!CheckHip(hipEventRecord(start, stream), "hipEventRecord(start)"))
				return false;
			launch();
			if (!CheckHip(hipGetLastError(), "benchmark launch") ||
				!CheckHip(hipEventRecord(stop, stream), "hipEventRecord(stop)") ||
				!CheckHip(hipEventSynchronize(stop), "hipEventSynchronize(stop)"))
				return false;
			float elapsed_ms = 0.0f;
			if (!CheckHip(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime"))
				return false;
			const double inversions = static_cast<double>(element_count) * options.iterations;
			milliseconds.push_back(elapsed_ms);
			rates.push_back(inversions / (static_cast<double>(elapsed_ms) * 1.0e6));
		}
		const double median_ms = Median(milliseconds);
		const double median_rate = Median(rates);
		std::vector<double> deviations;
		for (const double rate : rates)
			deviations.push_back(std::abs(rate - median_rate));
		results.push_back({name, median_ms, median_rate, Median(deviations)});
		std::printf("%-20s %10.3f ms  %10.3f Ginversion/s\n",
			name, median_ms, median_rate);
		return true;
	};

	bool passed = true;
	passed &= measure("per_lane", [&] {
		hipLaunchKernelGGL(InversePerLane, grid, block, 0, stream,
			device_state, options.iterations);
	});
	passed &= measure("wave64", [&] {
		hipLaunchKernelGGL(InverseWave64, grid, block, 0, stream,
			device_state, options.iterations);
	});

	std::printf("RCK_WAVE64_BENCHMARK_JSON={\"schema\":1,\"device\":%d,"
		"\"gpu\":\"%s\",\"arch\":\"%s\",\"blocks\":%d,"
		"\"threads_per_block\":%d,\"iterations\":%d,\"results\":[",
		options.device, properties.name, properties.gcnArchName, block_count,
		kThreadsPerBlock, options.iterations);
	for (size_t index = 0; index < results.size(); ++index)
	{
		const Result& result = results[index];
		std::printf("%s{\"name\":\"%s\",\"median_ms\":%.6f,"
			"\"median_ginversions\":%.6f,\"mad_ginversions\":%.6f}",
			index ? "," : "", result.name, result.median_ms,
			result.median_ginversions, result.mad_ginversions);
	}
	std::printf("]}\n");

	passed &= CheckHip(hipEventDestroy(stop), "hipEventDestroy(stop)");
	passed &= CheckHip(hipEventDestroy(start), "hipEventDestroy(start)");
	passed &= CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
	passed &= CheckHip(hipFree(device_state), "hipFree(state)");
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
