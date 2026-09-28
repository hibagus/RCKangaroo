#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "field/variants.hpp"

namespace
{
using rckangaroo::hip_field::MultiplyVariant;

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
	int multiply_iterations = 256;
	int inverse_iterations = 1;
	int blocks = 0;
};

template<MultiplyVariant Variant>
__device__ __forceinline__ void MultiplySelected(u32* result, const u32* lhs, const u32* rhs)
{
	if constexpr (Variant == MultiplyVariant::Portable)
		rckangaroo::hip_field::Multiply(result, lhs, rhs);
	else if constexpr (Variant == MultiplyVariant::CombaMad)
		rckangaroo::hip_field::MultiplyCombaMad(result, lhs, rhs);
	else
		rckangaroo::hip_field::MultiplyCombaExplicit(result, lhs, rhs);
}

template<MultiplyVariant Variant, bool Square>
__device__ __forceinline__ void RunMultiplyLoop(
	FieldElement* state, const FieldElement* multiplier, int iterations)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	u32 value[kLimbs];
	u32 rhs[kLimbs];
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
	{
		value[limb] = state[index].limb[limb];
		rhs[limb] = multiplier[index].limb[limb];
	}
	for (int iteration = 0; iteration < iterations; ++iteration)
	{
		u32 next[kLimbs];
		MultiplySelected<Variant>(next, value, Square ? value : rhs);
#pragma unroll
		for (int limb = 0; limb < kLimbs; ++limb)
			value[limb] = next[limb];
	}
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		state[index].limb[limb] = value[limb];
}

extern "C" __global__ void FieldMulPortable(
	FieldElement* state, const FieldElement* multiplier, int iterations)
{
	RunMultiplyLoop<MultiplyVariant::Portable, false>(state, multiplier, iterations);
}

extern "C" __global__ void FieldMulCombaMad(
	FieldElement* state, const FieldElement* multiplier, int iterations)
{
	RunMultiplyLoop<MultiplyVariant::CombaMad, false>(state, multiplier, iterations);
}

extern "C" __global__ void FieldMulCombaExplicit(
	FieldElement* state, const FieldElement* multiplier, int iterations)
{
	RunMultiplyLoop<MultiplyVariant::CombaExplicit, false>(state, multiplier, iterations);
}

extern "C" __global__ void FieldSquarePortable(
	FieldElement* state, const FieldElement* multiplier, int iterations)
{
	RunMultiplyLoop<MultiplyVariant::Portable, true>(state, multiplier, iterations);
}

extern "C" __global__ void FieldSquareCombaMad(
	FieldElement* state, const FieldElement* multiplier, int iterations)
{
	RunMultiplyLoop<MultiplyVariant::CombaMad, true>(state, multiplier, iterations);
}

extern "C" __global__ void FieldSquareCombaExplicit(
	FieldElement* state, const FieldElement* multiplier, int iterations)
{
	RunMultiplyLoop<MultiplyVariant::CombaExplicit, true>(state, multiplier, iterations);
}

enum class InverseVariant
{
	BinaryExponent,
	ChainPortable,
	ChainCombaMad,
	ChainCombaExplicit,
};

template<InverseVariant Variant>
__device__ __forceinline__ void RunInverseLoop(FieldElement* state, int iterations)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	u32 value[kLimbs];
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		value[limb] = state[index].limb[limb];
	for (int iteration = 0; iteration < iterations; ++iteration)
	{
		if constexpr (Variant == InverseVariant::BinaryExponent)
			rckangaroo::hip_field::Invert(value);
		else if constexpr (Variant == InverseVariant::ChainPortable)
			rckangaroo::hip_field::InvertAdditionChain<MultiplyVariant::Portable>(value);
		else if constexpr (Variant == InverseVariant::ChainCombaMad)
			rckangaroo::hip_field::InvertAdditionChain<MultiplyVariant::CombaMad>(value);
		else
			rckangaroo::hip_field::InvertAdditionChain<MultiplyVariant::CombaExplicit>(value);
	}
#pragma unroll
	for (int limb = 0; limb < kLimbs; ++limb)
		state[index].limb[limb] = value[limb];
}

extern "C" __global__ void FieldInverseBinaryExponent(FieldElement* state, int iterations)
{
	RunInverseLoop<InverseVariant::BinaryExponent>(state, iterations);
}

extern "C" __global__ void FieldInverseChainPortable(FieldElement* state, int iterations)
{
	RunInverseLoop<InverseVariant::ChainPortable>(state, iterations);
}

extern "C" __global__ void FieldInverseChainCombaMad(FieldElement* state, int iterations)
{
	RunInverseLoop<InverseVariant::ChainCombaMad>(state, iterations);
}

extern "C" __global__ void FieldInverseChainCombaExplicit(FieldElement* state, int iterations)
{
	RunInverseLoop<InverseVariant::ChainCombaExplicit>(state, iterations);
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
		else if (argument == "--multiply-iterations" && value > 0)
			options->multiply_iterations = value;
		else if (argument == "--inverse-iterations" && value > 0)
			options->inverse_iterations = value;
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
	int iterations;
	double median_ms;
	double median_gops;
	double mad_gops;
};
} // namespace

int main(int argc, char** argv)
{
	if (argc == 2 && std::string_view(argv[1]) == "--help")
	{
		std::printf("Usage: %s [--device N] [--samples N] [--blocks N] "
			"[--multiply-iterations N] [--inverse-iterations N]\n", argv[0]);
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
	const int block_count = options.blocks ? options.blocks : properties.multiProcessorCount;
	const int element_count = block_count * kThreadsPerBlock;

	std::vector<FieldElement> initial(element_count);
	std::vector<FieldElement> multiplier(element_count);
	constexpr u32 generator_x[kLimbs] = {
		0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
		0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu};
	constexpr u32 generator_y[kLimbs] = {
		0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
		0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u};
	for (int index = 0; index < element_count; ++index)
	{
		for (int limb = 0; limb < kLimbs; ++limb)
		{
			initial[index].limb[limb] = generator_x[limb];
			multiplier[index].limb[limb] = generator_y[limb];
		}
		initial[index].limb[0] ^= static_cast<u32>(index * 0x9E3779B9u);
		multiplier[index].limb[0] ^= static_cast<u32>(index * 0x85EBCA6Bu);
	}

	FieldElement* device_state = nullptr;
	FieldElement* device_multiplier = nullptr;
	const size_t bytes = initial.size() * sizeof(FieldElement);
	hipStream_t stream = nullptr;
	hipEvent_t start = nullptr;
	hipEvent_t stop = nullptr;
	if (!CheckHip(hipMalloc(&device_state, bytes), "hipMalloc(state)") ||
		!CheckHip(hipMalloc(&device_multiplier, bytes), "hipMalloc(multiplier)") ||
		!CheckHip(hipMemcpy(device_multiplier, multiplier.data(), bytes, hipMemcpyHostToDevice),
			"hipMemcpy(multiplier)") ||
		!CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreate") ||
		!CheckHip(hipEventCreate(&start), "hipEventCreate(start)") ||
		!CheckHip(hipEventCreate(&stop), "hipEventCreate(stop)"))
		return EXIT_FAILURE;

	std::vector<Result> results;
	auto measure = [&](const char* name, int iterations, auto launch)
	{
		if (!CheckHip(hipMemcpyAsync(device_state, initial.data(), bytes,
			hipMemcpyHostToDevice, stream), "hipMemcpyAsync(reset)"))
			return false;
		launch();
		if (!CheckHip(hipGetLastError(), "warm-up launch") ||
			!CheckHip(hipStreamSynchronize(stream), "warm-up synchronize"))
			return false;

		std::vector<double> milliseconds;
		std::vector<double> gops;
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
			const double operations = static_cast<double>(element_count) * iterations;
			milliseconds.push_back(elapsed_ms);
			gops.push_back(operations / (static_cast<double>(elapsed_ms) * 1.0e6));
		}
		const double median_ms = Median(milliseconds);
		const double median_gops = Median(gops);
		std::vector<double> deviations;
		for (const double value : gops)
			deviations.push_back(std::abs(value - median_gops));
		results.push_back({name, iterations, median_ms, median_gops, Median(deviations)});
		std::printf("%-30s %10.3f ms  %10.3f Gop/s\n", name, median_ms, median_gops);
		return true;
	};

	const dim3 grid(block_count);
	const dim3 block(kThreadsPerBlock);
	bool passed = true;
#define MEASURE_MULTIPLY(name, kernel) \
	passed &= measure((name), options.multiply_iterations, [&] { \
		hipLaunchKernelGGL((kernel), grid, block, 0, stream, device_state, \
			device_multiplier, options.multiply_iterations); \
	})
	MEASURE_MULTIPLY("multiply_portable", FieldMulPortable);
	MEASURE_MULTIPLY("multiply_comba_mad", FieldMulCombaMad);
	MEASURE_MULTIPLY("multiply_comba_explicit", FieldMulCombaExplicit);
	MEASURE_MULTIPLY("square_portable", FieldSquarePortable);
	MEASURE_MULTIPLY("square_comba_mad", FieldSquareCombaMad);
	MEASURE_MULTIPLY("square_comba_explicit", FieldSquareCombaExplicit);
#undef MEASURE_MULTIPLY

#define MEASURE_INVERSE(name, kernel) \
	passed &= measure((name), options.inverse_iterations, [&] { \
		hipLaunchKernelGGL((kernel), grid, block, 0, stream, device_state, \
			options.inverse_iterations); \
	})
	MEASURE_INVERSE("inverse_binary_exponent", FieldInverseBinaryExponent);
	MEASURE_INVERSE("inverse_chain_portable", FieldInverseChainPortable);
	MEASURE_INVERSE("inverse_chain_comba_mad", FieldInverseChainCombaMad);
	MEASURE_INVERSE("inverse_chain_comba_explicit", FieldInverseChainCombaExplicit);
#undef MEASURE_INVERSE

	std::printf("RCK_FIELD_BENCHMARK_JSON={\"schema\":1,\"device\":%d,\"gpu\":\"%s\","
		"\"arch\":\"%s\",\"blocks\":%d,\"threads_per_block\":%d,\"results\":[",
		options.device, properties.name, properties.gcnArchName, block_count, kThreadsPerBlock);
	for (size_t index = 0; index < results.size(); ++index)
	{
		const Result& result = results[index];
		std::printf("%s{\"name\":\"%s\",\"iterations\":%d,\"median_ms\":%.6f,"
			"\"median_gops\":%.6f,\"mad_gops\":%.6f}", index ? "," : "",
			result.name, result.iterations, result.median_ms,
			result.median_gops, result.mad_gops);
	}
	std::printf("]}\n");

	passed &= CheckHip(hipEventDestroy(stop), "hipEventDestroy(stop)");
	passed &= CheckHip(hipEventDestroy(start), "hipEventDestroy(start)");
	passed &= CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
	passed &= CheckHip(hipFree(device_multiplier), "hipFree(multiplier)");
	passed &= CheckHip(hipFree(device_state), "hipFree(state)");
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
