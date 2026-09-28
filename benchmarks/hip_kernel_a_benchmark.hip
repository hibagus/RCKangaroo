#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rckangaroo/config.hpp"
#include "rckangaroo/gpu/kernels.hpp"

namespace
{
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
} // namespace

int main(int argc, char** argv)
{
	const int device_index = argc > 1 ? std::atoi(argv[1]) : 0;
	if (!CheckHip(hipSetDevice(device_index), "hipSetDevice"))
		return EXIT_FAILURE;
	hipDeviceProp_t properties{};
	if (!CheckHip(hipGetDeviceProperties(&properties, device_index), "hipGetDeviceProperties"))
		return EXIT_FAILURE;

	const u32 block_count = properties.multiProcessorCount;
	const u32 kangaroo_count = block_count * BLOCK_SIZE * PNT_GROUP_CNT;
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
	params.GroupCnt = PNT_GROUP_CNT;
	params.KangCnt = kangaroo_count;
	params.iter_cnt = 1;
	params.dp_mask = 0xFFFFFFFFu;
	params.KernelA_LDS_Size = 8 * JMP_CNT * sizeof(u64);
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

	constexpr int sample_count = 5;
	std::array<float, sample_count> milliseconds{};
	for (int sample = -1; sample < sample_count; ++sample)
	{
		if (!CheckHip(hipMemcpyAsync(params.L2, initial_l2.data(), l2_bytes,
			hipMemcpyHostToDevice, stream), "hipMemcpyAsync(L2 reset)") ||
			!CheckHip(hipMemsetAsync(params.L1S2, 0, l1s2_bytes, stream), "hipMemsetAsync(L1S2)") ||
			!CheckHip(hipStreamSynchronize(stream), "hipStreamSynchronize(reset)") ||
			!CheckHip(hipEventRecord(start, stream), "hipEventRecord(start)") ||
			!CheckHip(LaunchKernelA(params, stream), "LaunchKernelA") ||
			!CheckHip(hipEventRecord(stop, stream), "hipEventRecord(stop)") ||
			!CheckHip(hipEventSynchronize(stop), "hipEventSynchronize(stop)"))
			return EXIT_FAILURE;
		if (sample >= 0 && !CheckHip(hipEventElapsedTime(&milliseconds[sample], start, stop),
			"hipEventElapsedTime"))
			return EXIT_FAILURE;
	}

	std::sort(milliseconds.begin(), milliseconds.end());
	const float median_ms = milliseconds[sample_count / 2];
	const double mkeys_per_second = static_cast<double>(kangaroo_count) / (median_ms * 1000.0);
	std::printf("GPU %d: %s (%s), %u CUs\n", device_index, properties.name,
		properties.gcnArchName, block_count);
	std::printf("KernelA portable baseline: %.3f ms, %.3f MKeys/s (%u kangaroos, one step)\n",
		median_ms, mkeys_per_second, kangaroo_count);

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
