#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#include "rckangaroo/config.hpp"
#include "rckangaroo/gpu/kernels.hpp"
#include "rckangaroo/gpu/tuning.hpp"

namespace
{
struct BenchmarkOptions
{
	int device_index = 0;
	int warmup_ms = 1000;
	int sample_ms = 1000;
	int sample_count = 5;
	u32 block_count = 0;
	u32 group_count = PNT_GROUP_CNT;
	bool group_count_explicit = false;
};

constexpr std::array<u64, 4> kGeneratorX = {
	0x59F2815B16F81798ull, 0x029BFCDB2DCE28D9ull,
	0x55A06295CE870B07ull, 0x79BE667EF9DCBBACull};
constexpr std::array<u64, 4> kGeneratorY = {
	0x9C47D08FFB10D4B8ull, 0xFD17B448A6855419ull,
	0x5DA4FBFC0E1108A8ull, 0x483ADA7726A3C465ull};
constexpr std::array<u64, 4> kDoubleGeneratorX = {
	0xABAC09B95C709EE5ull, 0x5C778E4B8CEF3CA7ull,
	0x3045406E95C07CD8ull, 0xC6047F9441ED7D6Dull};
constexpr std::array<u64, 4> kDoubleGeneratorY = {
	0x236431A950CFE52Aull, 0xF7F632653266D0E1ull,
	0xA3C58419466CEAEEull, 0x1AE168FEA63DC339ull};

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

bool ParseOptions(int argc, char** argv, BenchmarkOptions* options)
{
	for (int index = 1; index < argc; ++index)
	{
		const std::string_view argument = argv[index];
		if (argument == "--help")
		{
			std::printf("Usage: %s [device] [--device N] [--warmup-ms N] "
				"[--sample-ms N] [--samples N] [--blocks N] [--groups N]\n", argv[0]);
			return false;
		}
		if (index == 1 && !argument.starts_with("-"))
		{
			if (!ParseInteger(argv[index], 0, &options->device_index))
				return false;
			continue;
		}
		if (index + 1 >= argc)
			return false;
		int value = 0;
		if (!ParseInteger(argv[index + 1], 0, &value))
			return false;
		if (argument == "--device")
			options->device_index = value;
		else if (argument == "--warmup-ms")
			options->warmup_ms = value;
		else if (argument == "--sample-ms" && value > 0)
			options->sample_ms = value;
		else if (argument == "--samples" && value > 0)
			options->sample_count = value;
		else if (argument == "--blocks" && value > 0)
			options->block_count = static_cast<u32>(value);
		else if (argument == "--groups" && value >= 2 && value <= 32 && !(value & 1))
		{
			options->group_count = static_cast<u32>(value);
			options->group_count_explicit = true;
		}
		else
			return false;
		++index;
	}
	return options->sample_count <= 1000;
}

double Median(std::vector<double> values)
{
	std::sort(values.begin(), values.end());
	const size_t middle = values.size() / 2;
	if (values.size() & 1)
		return values[middle];
	return (values[middle - 1] + values[middle]) / 2.0;
}
} // namespace

int main(int argc, char** argv)
{
	if (argc == 2 && std::string_view(argv[1]) == "--help")
	{
		std::printf("Usage: %s [device] [--device N] [--warmup-ms N] "
			"[--sample-ms N] [--samples N] [--blocks N] [--groups N]\n", argv[0]);
		return EXIT_SUCCESS;
	}
	BenchmarkOptions options;
	if (!ParseOptions(argc, argv, &options))
	{
		std::fprintf(stderr, "Invalid benchmark arguments. Use --help for usage.\n");
		return EXIT_FAILURE;
	}
	if (!CheckHip(hipSetDevice(options.device_index), "hipSetDevice"))
		return EXIT_FAILURE;
	hipDeviceProp_t properties{};
	if (!CheckHip(hipGetDeviceProperties(&properties, options.device_index), "hipGetDeviceProperties"))
		return EXIT_FAILURE;
	const rckangaroo::hip::ArchitectureTuning tuning =
		rckangaroo::hip::SelectArchitectureTuning(properties.gcnArchName);
	if (!options.group_count_explicit)
		options.group_count = tuning.point_group_count;

	const u32 block_count = options.block_count ? options.block_count :
		properties.multiProcessorCount * BLOCKS_PER_CU;
	const u32 kangaroo_count = block_count * BLOCK_SIZE * options.group_count;
	const size_t coordinate_words = static_cast<size_t>(kangaroo_count) * 4;
	std::vector<u64> initial_l2(static_cast<size_t>(kangaroo_count) * 12, 0);
	for (u32 index = 0; index < kangaroo_count; ++index)
	{
		std::memcpy(initial_l2.data() + 4 * index, kGeneratorX.data(), 4 * sizeof(u64));
		std::memcpy(initial_l2.data() + coordinate_words + 4 * index,
			kGeneratorY.data(), 4 * sizeof(u64));
	}

	std::vector<u64> jump_table_1(static_cast<size_t>(JMP_CNT) * 12, 0);
	std::vector<u64> jump_table_2(static_cast<size_t>(JMP_CNT) * 8, 0);
	for (int index = 0; index < JMP_CNT; ++index)
	{
		std::memcpy(jump_table_1.data() + 12 * index, kDoubleGeneratorX.data(), 4 * sizeof(u64));
		std::memcpy(jump_table_1.data() + 12 * index + 4, kDoubleGeneratorY.data(), 4 * sizeof(u64));
		std::memcpy(jump_table_2.data() + 8 * index, kDoubleGeneratorX.data(), 4 * sizeof(u64));
		std::memcpy(jump_table_2.data() + 8 * index + 4, kDoubleGeneratorY.data(), 4 * sizeof(u64));
	}

	TKparams params{};
	params.BlockCnt = block_count;
	params.BlockSize = BLOCK_SIZE;
	params.GroupCnt = options.group_count;
	params.KangCnt = kangaroo_count;
	params.iter_cnt = 1;
	params.dp_mask = 0xFFFFFFFFu;
	params.KernelA_LDS_Size = tuning.kernel_a_lds_bytes;
	const size_t l2_bytes = initial_l2.size() * sizeof(u64);
	const size_t l1s2_bytes = static_cast<size_t>(block_count) * BLOCK_SIZE * sizeof(u32);
	const size_t jump_list_bytes = static_cast<size_t>(kangaroo_count) *
		(params.iter_cnt + MD_LEN) * sizeof(u16);
	const size_t last_points_bytes = static_cast<size_t>(kangaroo_count) * MD_LEN * 64;

	if (!CheckHip(hipMalloc(&params.L2, l2_bytes), "hipMalloc(L2)") ||
		!CheckHip(hipMalloc(&params.Jumps1, jump_table_1.size() * sizeof(u64)), "hipMalloc(Jumps1)") ||
		!CheckHip(hipMalloc(&params.L1S2, l1s2_bytes), "hipMalloc(L1S2)") ||
		!CheckHip(hipMalloc(&params.JumpsList, jump_list_bytes), "hipMalloc(JumpsList)") ||
		!CheckHip(hipMalloc(&params.LastPnts, last_points_bytes), "hipMalloc(LastPnts)"))
		return EXIT_FAILURE;

	hipStream_t stream = nullptr;
	hipEvent_t start = nullptr;
	hipEvent_t stop = nullptr;
	if (!CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags") ||
		!CheckHip(hipEventCreate(&start), "hipEventCreate(start)") ||
		!CheckHip(hipEventCreate(&stop), "hipEventCreate(stop)") ||
		!CheckHip(hipMemcpy(params.Jumps1, jump_table_1.data(), jump_table_1.size() * sizeof(u64),
			hipMemcpyHostToDevice), "hipMemcpy(Jumps1)") ||
		!CheckHip(SetGpuParams(params, jump_table_2.data()), "SetGpuParams"))
		return EXIT_FAILURE;

	auto measure_launch = [&]() -> float
	{
		if (!CheckHip(hipMemcpyAsync(params.L2, initial_l2.data(), l2_bytes,
			hipMemcpyHostToDevice, stream), "hipMemcpyAsync(L2 reset)") ||
			!CheckHip(hipMemsetAsync(params.L1S2, 0, l1s2_bytes, stream), "hipMemsetAsync(L1S2)") ||
			!CheckHip(hipStreamSynchronize(stream), "hipStreamSynchronize(reset)") ||
			!CheckHip(hipEventRecord(start, stream), "hipEventRecord(start)") ||
			!CheckHip(LaunchKernelA(params, stream), "LaunchKernelA") ||
			!CheckHip(hipEventRecord(stop, stream), "hipEventRecord(stop)") ||
			!CheckHip(hipEventSynchronize(stop), "hipEventSynchronize(stop)"))
			return -1.0f;
		float milliseconds = 0.0f;
		if (!CheckHip(hipEventElapsedTime(&milliseconds, start, stop), "hipEventElapsedTime"))
			return -1.0f;
		return milliseconds;
	};

	double warmup_elapsed_ms = 0.0;
	int warmup_launches = 0;
	while (warmup_elapsed_ms < options.warmup_ms)
	{
		const float elapsed = measure_launch();
		if (elapsed < 0.0f)
			return EXIT_FAILURE;
		warmup_elapsed_ms += elapsed;
		++warmup_launches;
	}

	std::vector<double> sample_mkeys;
	std::vector<double> sample_average_ms;
	std::vector<int> sample_launches;
	for (int sample = 0; sample < options.sample_count; ++sample)
	{
		double elapsed_ms = 0.0;
		int launches = 0;
		while (elapsed_ms < options.sample_ms)
		{
			const float elapsed = measure_launch();
			if (elapsed < 0.0f)
				return EXIT_FAILURE;
			elapsed_ms += elapsed;
			++launches;
		}
		const double points = static_cast<double>(kangaroo_count) * launches;
		sample_mkeys.push_back(points / (elapsed_ms * 1000.0));
		sample_average_ms.push_back(elapsed_ms / launches);
		sample_launches.push_back(launches);
	}

	const double median_ms = Median(sample_average_ms);
	const double median_mkeys = Median(sample_mkeys);
	std::vector<double> deviations;
	deviations.reserve(sample_mkeys.size());
	for (const double value : sample_mkeys)
		deviations.push_back(std::abs(value - median_mkeys));
	const double mad_mkeys = Median(deviations);
	std::printf("GPU %d: %s (%s), %u CUs\n", options.device_index, properties.name,
		properties.gcnArchName, properties.multiProcessorCount);
	std::printf("Geometry: %u blocks, %u threads, %u groups, %u kangaroos, 1 step\n",
		block_count, BLOCK_SIZE, options.group_count, kangaroo_count);
	std::printf("Protocol: %d ms warm-up (%d launches), %d x %d ms samples\n",
		options.warmup_ms, warmup_launches, options.sample_count, options.sample_ms);
	for (int sample = 0; sample < options.sample_count; ++sample)
		std::printf("Sample %d: %.3f MKeys/s, %.3f ms/launch, %d launches\n",
			sample + 1, sample_mkeys[sample], sample_average_ms[sample], sample_launches[sample]);
	std::printf("KernelA portable baseline: %.3f ms, %.3f MKeys/s median, %.3f MAD\n",
		median_ms, median_mkeys, mad_mkeys);
	std::printf("RCK_BENCHMARK_JSON={\"schema\":1,\"device\":%d,\"gpu\":\"%s\","
		"\"arch\":\"%s\",\"compute_units\":%d,\"blocks\":%u,"
		"\"configured_blocks_per_cu\":%u,\"threads\":%u,\"groups\":%u,"
		"\"configured_groups\":%u,\"table_mode\":%d,\"lds_bytes\":%zu,"
		"\"state_layout\":%d,"
		"\"iterations\":1,\"kangaroos\":%u,\"warmup_ms\":%d,\"sample_ms\":%d,"
		"\"samples\":%d,\"median_kernel_ms\":%.6f,\"median_mkeys_per_second\":%.6f,"
		"\"mad_mkeys_per_second\":%.6f,\"sample_mkeys_per_second\":[",
		options.device_index, properties.name, properties.gcnArchName,
		properties.multiProcessorCount, block_count, BLOCKS_PER_CU, BLOCK_SIZE,
		options.group_count, tuning.point_group_count, tuning.kernel_a_table_mode,
		static_cast<size_t>(tuning.kernel_a_lds_bytes), tuning.state_layout, kangaroo_count,
		options.warmup_ms, options.sample_ms,
		options.sample_count, median_ms, median_mkeys, mad_mkeys);
	for (int sample = 0; sample < options.sample_count; ++sample)
		std::printf("%s%.6f", sample ? "," : "", sample_mkeys[sample]);
	std::printf("]}\n");

	bool passed = true;
	passed &= CheckHip(hipEventDestroy(stop), "hipEventDestroy(stop)");
	passed &= CheckHip(hipEventDestroy(start), "hipEventDestroy(start)");
	passed &= CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
	passed &= CheckHip(hipFree(params.LastPnts), "hipFree(LastPnts)");
	passed &= CheckHip(hipFree(params.JumpsList), "hipFree(JumpsList)");
	passed &= CheckHip(hipFree(params.L1S2), "hipFree(L1S2)");
	passed &= CheckHip(hipFree(params.Jumps1), "hipFree(Jumps1)");
	passed &= CheckHip(hipFree(params.L2), "hipFree(L2)");
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
