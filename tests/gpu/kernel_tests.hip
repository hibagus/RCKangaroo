#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rckangaroo/config.hpp"
#include "rckangaroo/ec.hpp"
#include "rckangaroo/gpu/kernels.hpp"
#include "rckangaroo/gpu/tuning.hpp"

namespace
{
constexpr u32 kBlockCount = 1;
constexpr u32 kKangarooCount = kBlockCount * BLOCK_SIZE * PNT_GROUP_CNT;

bool CheckHip(hipError_t status, const char* operation)
{
	if (status == hipSuccess)
		return true;
	std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
	return false;
}

bool EqualCoordinate(const u64* actual, const EcInt& expected)
{
	return std::memcmp(actual, expected.data, 4 * sizeof(u64)) == 0;
}

void StorePoint(std::vector<u64>& soa, u32 kangaroo_count, u32 index, const EcPoint& point)
{
	const size_t coordinate_stride = static_cast<size_t>(kangaroo_count) * 4;
	std::memcpy(soa.data() + 4 * index, point.x.data, 4 * sizeof(u64));
	std::memcpy(soa.data() + coordinate_stride + 4 * index, point.y.data, 4 * sizeof(u64));
}

bool TestKernelGen(hipStream_t stream)
{
	std::vector<u64> host_l2(static_cast<size_t>(kKangarooCount) * 12, 0);
	const size_t distance_offset = static_cast<size_t>(kKangarooCount) * 8;
	for (u32 index = 0; index < kKangarooCount; ++index)
		host_l2[distance_offset + 4 * index] = (index & 1u) + 1u;

	u64* device_l2 = nullptr;
	if (!CheckHip(hipMalloc(&device_l2, host_l2.size() * sizeof(u64)), "hipMalloc(KernelGen L2)"))
		return false;
	if (!CheckHip(hipMemcpyAsync(device_l2, host_l2.data(), host_l2.size() * sizeof(u64),
		hipMemcpyHostToDevice, stream), "hipMemcpyAsync(KernelGen input)"))
		return false;

	TKparams params{};
	params.L2 = device_l2;
	params.BlockCnt = kBlockCount;
	params.BlockSize = BLOCK_SIZE;
	params.GroupCnt = PNT_GROUP_CNT;
	params.KangCnt = kKangarooCount;
	params.IsGenMode = 1;
	if (!CheckHip(LaunchKernelGen(params, stream), "LaunchKernelGen") ||
		!CheckHip(hipStreamSynchronize(stream), "hipStreamSynchronize(KernelGen)"))
		return false;
	if (!CheckHip(hipMemcpy(host_l2.data(), device_l2, host_l2.size() * sizeof(u64),
		hipMemcpyDeviceToHost), "hipMemcpy(KernelGen output)"))
		return false;

	Ec ec;
	EcInt scalar_one;
	EcInt scalar_two;
	scalar_one.Set(1);
	scalar_two.Set(2);
	EcPoint expected[2] = {ec.MultiplyG(scalar_one), ec.MultiplyG(scalar_two)};
	const size_t coordinate_stride = static_cast<size_t>(kKangarooCount) * 4;
	bool passed = true;
	for (u32 index = 0; index < kKangarooCount; ++index)
	{
		const EcPoint& point = expected[index & 1u];
		if (!EqualCoordinate(host_l2.data() + 4 * index, point.x) ||
			!EqualCoordinate(host_l2.data() + coordinate_stride + 4 * index, point.y))
		{
			std::fprintf(stderr, "KernelGen mismatch at kangaroo %u\n", index);
			passed = false;
			break;
		}
	}
	passed &= CheckHip(hipFree(device_l2), "hipFree(KernelGen L2)");
	return passed;
}

bool TestKernelAOneStep(hipStream_t stream, u32 group_count, u32 kernel_a_lds_bytes)
{
	Ec ec;
	EcInt scalar_one;
	EcInt scalar_two;
	EcInt scalar_three;
	scalar_one.Set(1);
	scalar_two.Set(2);
	scalar_three.Set(3);
	EcPoint generator = ec.MultiplyG(scalar_one);
	EcPoint jump = ec.MultiplyG(scalar_two);
	EcPoint expected = ec.MultiplyG(scalar_three);

	const u32 kangaroo_count = kBlockCount * BLOCK_SIZE * group_count;
	std::vector<u64> host_l2(static_cast<size_t>(kangaroo_count) * 12, 0);
	for (u32 index = 0; index < kangaroo_count; ++index)
		StorePoint(host_l2, kangaroo_count, index, generator);

	std::vector<u64> jump_table_1(static_cast<size_t>(JMP_CNT) * 12, 0);
	std::vector<u64> jump_table_2(static_cast<size_t>(JMP_CNT) * 8, 0);
	for (int index = 0; index < JMP_CNT; ++index)
	{
		std::memcpy(jump_table_1.data() + 12 * index, jump.x.data, 4 * sizeof(u64));
		std::memcpy(jump_table_1.data() + 12 * index + 4, jump.y.data, 4 * sizeof(u64));
		std::memcpy(jump_table_2.data() + 8 * index, jump.x.data, 4 * sizeof(u64));
		std::memcpy(jump_table_2.data() + 8 * index + 4, jump.y.data, 4 * sizeof(u64));
	}

	TKparams params{};
	params.BlockCnt = kBlockCount;
	params.BlockSize = BLOCK_SIZE;
	params.GroupCnt = group_count;
	params.KangCnt = kangaroo_count;
	params.iter_cnt = 1;
	params.dp_mask = 0xFFFFFFFFu;
	params.KernelA_LDS_Size = kernel_a_lds_bytes;

	const size_t l2_bytes = host_l2.size() * sizeof(u64);
	const size_t last_points_bytes = static_cast<size_t>(kangaroo_count) * MD_LEN * 64;
	const size_t jump_list_bytes = static_cast<size_t>(kangaroo_count) * sizeof(u16);
	const size_t dp_table_bytes = static_cast<size_t>(kangaroo_count) *
		(sizeof(u32) + DPTABLE_MAX_CNT * 16);
	if (!CheckHip(hipMalloc(&params.L2, l2_bytes), "hipMalloc(KernelA L2)") ||
		!CheckHip(hipMalloc(&params.Jumps1, jump_table_1.size() * sizeof(u64)), "hipMalloc(Jumps1)") ||
		!CheckHip(hipMalloc(&params.L1S2, BLOCK_SIZE * sizeof(u32)), "hipMalloc(L1S2)") ||
		!CheckHip(hipMalloc(&params.JumpsList, jump_list_bytes), "hipMalloc(JumpsList)") ||
		!CheckHip(hipMalloc(&params.LastPnts, last_points_bytes), "hipMalloc(LastPnts)") ||
		!CheckHip(hipMalloc(&params.DPTable, dp_table_bytes), "hipMalloc(DPTable)"))
		return false;

	if (!CheckHip(hipMemcpyAsync(params.L2, host_l2.data(), l2_bytes, hipMemcpyHostToDevice, stream),
		"hipMemcpyAsync(KernelA L2)") ||
		!CheckHip(hipMemcpyAsync(params.Jumps1, jump_table_1.data(), jump_table_1.size() * sizeof(u64),
			hipMemcpyHostToDevice, stream), "hipMemcpyAsync(Jumps1)") ||
		!CheckHip(hipMemsetAsync(params.L1S2, 0, BLOCK_SIZE * sizeof(u32), stream), "hipMemsetAsync(L1S2)") ||
		!CheckHip(hipMemsetAsync(params.DPTable, 0, dp_table_bytes, stream), "hipMemsetAsync(DPTable)") ||
		!CheckHip(SetGpuParams(params, jump_table_2.data()), "SetGpuParams"))
		return false;

	if (!CheckHip(LaunchKernelA(params, stream), "LaunchKernelA(one step)") ||
		!CheckHip(hipStreamSynchronize(stream), "hipStreamSynchronize(KernelA one step)"))
		return false;
	if (!CheckHip(hipMemcpy(host_l2.data(), params.L2, l2_bytes, hipMemcpyDeviceToHost),
		"hipMemcpy(KernelA output)"))
		return false;

	const size_t coordinate_stride = static_cast<size_t>(kangaroo_count) * 4;
	bool passed = true;
	for (u32 index = 0; index < kangaroo_count; ++index)
	{
		if (!EqualCoordinate(host_l2.data() + 4 * index, expected.x) ||
			!EqualCoordinate(host_l2.data() + coordinate_stride + 4 * index, expected.y))
		{
			std::fprintf(stderr,
				"KernelA one-step mismatch at kangaroo %u with %u groups\n",
				index, group_count);
			passed = false;
			break;
		}
	}

#define RCK_FREE(member) passed &= CheckHip(hipFree(params.member), "hipFree(" #member ")")
	RCK_FREE(DPTable);
	RCK_FREE(LastPnts);
	RCK_FREE(JumpsList);
	RCK_FREE(L1S2);
	RCK_FREE(Jumps1);
	RCK_FREE(L2);
#undef RCK_FREE
	return passed;
}
} // namespace

int main()
{
	if (!CheckHip(hipSetDevice(0), "hipSetDevice"))
		return EXIT_FAILURE;
	hipDeviceProp_t properties{};
	if (!CheckHip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties"))
		return EXIT_FAILURE;
	const rckangaroo::hip::ArchitectureTuning tuning =
		rckangaroo::hip::SelectArchitectureTuning(properties.gcnArchName);
	hipStream_t stream = nullptr;
	if (!CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags"))
		return EXIT_FAILURE;

	InitEc();
	bool passed = TestKernelGen(stream);
	constexpr u32 group_counts[] = {8, 12, 16, 24, 32};
	for (const u32 group_count : group_counts)
		passed &= TestKernelAOneStep(stream, group_count, tuning.kernel_a_lds_bytes);
	DeInitEc();
	const bool destroyed = CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
	if (!passed || !destroyed)
		return EXIT_FAILURE;
	std::printf("KernelGen and KernelA one-step CPU comparisons passed for groups 8-32.\n");
	return EXIT_SUCCESS;
}
