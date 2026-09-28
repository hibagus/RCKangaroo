// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <hip/hip_runtime.h>

#include "rckangaroo/gpu/kangaroo.hpp"
#include "rckangaroo/gpu/kernels.hpp"
#include "rckangaroo/gpu/tuning.hpp"

void AddPointsToList(u32* data, int cnt, u32 KangCnt, u64 ops_cnt, int JumperInd);
extern bool gGenMode; //tames generation mode

namespace
{
struct ProfileStats
{
	float median = 0.0f;
	float mad = 0.0f;
	float minimum = 0.0f;
	float maximum = 0.0f;
};

bool CheckHip(hipError_t status, int device_index, const char* operation)
{
	if (status == hipSuccess)
		return true;
	fprintf(stderr, "GPU %d, %s failed: %s\n", device_index, operation, hipGetErrorString(status));
	return false;
}

float Median(std::vector<float> values)
{
	std::sort(values.begin(), values.end());
	const size_t middle = values.size() / 2;
	if (values.size() & 1)
		return values[middle];
	return (values[middle - 1] + values[middle]) / 2.0f;
}

ProfileStats Summarize(const std::vector<float>& values)
{
	if (values.empty())
		return {};
	ProfileStats result;
	result.median = Median(values);
	result.minimum = *std::min_element(values.begin(), values.end());
	result.maximum = *std::max_element(values.begin(), values.end());
	std::vector<float> deviations;
	deviations.reserve(values.size());
	for (const float value : values)
		deviations.push_back(std::abs(value - result.median));
	result.mad = Median(deviations);
	return result;
}
}

void RCGpuKang::ApplyArchitectureTuning(const char* architecture)
{
	const rckangaroo::hip::ArchitectureTuning tuning =
		rckangaroo::hip::SelectArchitectureTuning(architecture);
	PointGroupCnt = static_cast<int>(tuning.point_group_count);
	KernelStepCnt = static_cast<int>(tuning.step_count);
	KernelATableMode = tuning.kernel_a_table_mode;
	KernelALdsBytes = tuning.kernel_a_lds_bytes;
	StateLayout = tuning.state_layout;
}

int RCGpuKang::CalcKangCnt()
{
	Kparams.BlockCnt = mpCnt * BLOCKS_PER_CU;
	Kparams.BlockSize = BLOCK_SIZE;
	Kparams.GroupCnt = PointGroupCnt;
	return Kparams.BlockSize* Kparams.GroupCnt* Kparams.BlockCnt;
}

//executes in main thread
bool RCGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
	PntToSolve = _PntToSolve;
	Range = _Range;
	DP = _DP;
	EcJumps1 = _EcJumps1;
	EcJumps2 = _EcJumps2;
	EcJumps3 = _EcJumps3;
	StopFlag = false;
	Failed = false;
	u64 total_mem = 0;
	memset(dbg, 0, sizeof(dbg));
	memset(SpeedStats, 0, sizeof(SpeedStats));
	cur_stats_ind = 0;
	KernelGenMilliseconds = 0.0f;
	KernelAMilliseconds.clear();
	KernelBMilliseconds.clear();
	KernelCMilliseconds.clear();
	EndToEndMKeys.clear();

	hipError_t err = hipSetDevice(DeviceIndex);
	if (!CheckHip(err, DeviceIndex, "hipSetDevice"))
		return false;
	err = hipStreamCreateWithFlags(&Stream, hipStreamNonBlocking);
	if (!CheckHip(err, DeviceIndex, "hipStreamCreateWithFlags"))
		return false;
	const char* profile_environment = std::getenv("RCK_PROFILE");
	ProfilingEnabled = profile_environment && profile_environment[0] &&
		std::strcmp(profile_environment, "0") != 0;
	if (ProfilingEnabled)
	{
		if (!CheckHip(hipEventCreate(&ProfileStart), DeviceIndex, "hipEventCreate(profile start)") ||
			!CheckHip(hipEventCreate(&ProfileAfterA), DeviceIndex, "hipEventCreate(profile A)") ||
			!CheckHip(hipEventCreate(&ProfileAfterB), DeviceIndex, "hipEventCreate(profile B)") ||
			!CheckHip(hipEventCreate(&ProfileAfterC), DeviceIndex, "hipEventCreate(profile C)"))
			return false;
	}

	Kparams.BlockCnt = mpCnt * BLOCKS_PER_CU;
	Kparams.BlockSize = BLOCK_SIZE;
	Kparams.GroupCnt = PointGroupCnt;
	KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
	Kparams.KangCnt = KangCnt;
	Kparams.DP = DP;
	Kparams.KernelA_LDS_Size = KernelALdsBytes;
	Kparams.KernelB_LDS_Size = 48 * 1024;
	Kparams.KernelC_LDS_Size = 12 * JMP_CNT * sizeof(u64);
	Kparams.IsGenMode = gGenMode;
	Kparams.dp_mask = (u32)((1ull << DP) - 1);
	Kparams.iter_cnt = KernelStepCnt;
	Kparams.StopThr = (int)(0.5 * (Kparams.BlockCnt * 8)); //at the end, work will be stopped when number of finished producers is higher than this value. Must be <(WarpCnt-32)

// Allocate GPU memory.
	u64 size;
	const size_t L2size = static_cast<size_t>(Kparams.KangCnt) * (3 * 32);
	total_mem += L2size;
	err = hipMalloc((void**)&Kparams.L2, L2size);
	if (err != hipSuccess)
	{
		printf("GPU %d, Allocate L2 memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
	size = MAX_DP_CNT * GPU_DP_SIZE + 16;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.DPs_out, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate GpuOut memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = hipMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = hipMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = hipMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate Jumps3 memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	size = 2 * (u64)KangCnt * (Kparams.iter_cnt + MD_LEN);
	total_mem += size;
	err = hipMalloc((void**)&Kparams.JumpsList, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate JumpsList memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
	total_mem += size;
	err = hipMalloc((void**)&Kparams.DPTable, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate DPTable memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	size = static_cast<u64>(Kparams.BlockCnt) * Kparams.BlockSize * sizeof(u32);
	total_mem += size;
	err = hipMalloc((void**)&Kparams.L1S2, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate L1S2 memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * (2 * 32);
	total_mem += size;
	err = hipMalloc((void**)&Kparams.LastPnts, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * sizeof(u64);
	total_mem += size;
	err = hipMalloc((void**)&Kparams.LoopTable, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	total_mem += 1024;
	err = hipMalloc((void**)&Kparams.dbg_buf, 1024);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate dbg_buf memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	size = sizeof(u32) * KangCnt + 8;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.LoopedKangs, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate LoopedKangs memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	size = 32 * KangCnt;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.dists, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate dists memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);
	/////////////////
	size = JMP_CNT * 32 * 2 * 3;
	total_mem += size;
	err = hipMalloc((void**)&Kparams.Jumps12, size);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate Jumps12 memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}

	u32* pJumps12 = (u32*)malloc(JMP_CNT * 32 * 2 * 3); //96KB
	int part_ofs = 4 * JMP_CNT;
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(pJumps12 + i * 4, EcJumps1[i].p.x.data, 16);
		memcpy(pJumps12 + i * 4 + part_ofs, EcJumps1[i].p.x.data + 2, 16);
		memcpy(pJumps12 + i * 4 + 2 * part_ofs, EcJumps1[i].p.y.data, 16);
		memcpy(pJumps12 + i * 4 + 3 * part_ofs, EcJumps1[i].p.y.data + 2, 16);

		EcInt ng = EcJumps1[i].p.y;
		ng.NegModP();
		memcpy(pJumps12 + i * 4 + 4 * part_ofs, ng.data, 16);
		memcpy(pJumps12 + i * 4 + 5 * part_ofs, ng.data + 2, 16);
	}
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4, EcJumps2[i].p.x.data, 16);
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4 + part_ofs, EcJumps2[i].p.x.data + 2, 16);
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4 + 2 * part_ofs, EcJumps2[i].p.y.data, 16);
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4 + 3 * part_ofs, EcJumps2[i].p.y.data + 2, 16);

		EcInt ng = EcJumps2[i].p.y;
		ng.NegModP();
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4 + 4 * part_ofs, ng.data, 16);
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4 + 5 * part_ofs, ng.data + 2, 16);
	}
	err = hipMemcpy(Kparams.Jumps12, pJumps12, JMP_CNT * 32 * 2 * 3, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy Jumps1 failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
	/////////////////
	free(pJumps12);


	total_mem += JMP_CNT * 96;
	err = hipMalloc((void**)&Kparams.JmpDists12, JMP_CNT * 96);
	if (err != hipSuccess)
	{
		printf("GPU %d Allocate JmpDists12 memory failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
	EcInt val193b;
	val193b.SetHexStr("1000000000000000000000000000000000000000000000000");
	u32* jd12 = (u32*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(jd12 + i * 4, EcJumps1[i].dist.data, 16);
		memcpy(jd12 + i * 2 + (32 * 1024 / 4), EcJumps1[i].dist.data + 2, 8);
		EcInt neg = val193b;
		neg.Sub(EcJumps1[i].dist);
		memcpy(jd12 + i * 4 + (8 * 1024 / 4), neg.data, 16);
		memcpy(jd12 + i * 2 + (32 * 1024 / 4) + (4 * 1024 / 4), neg.data + 2, 8);

		memcpy(jd12 + i * 4 + (16 * 1024 / 4), EcJumps2[i].dist.data, 16);
		memcpy(jd12 + i * 2 + (32 * 1024 / 4) + (8 * 1024 / 4), EcJumps2[i].dist.data + 2, 8);
		neg = val193b;
		neg.Sub(EcJumps2[i].dist);
		memcpy(jd12 + i * 4 + (24 * 1024 / 4), neg.data, 16);
		memcpy(jd12 + i * 2 + (32 * 1024 / 4) + (12 * 1024 / 4), neg.data + 2, 8);
	}
	err = hipMemcpy(Kparams.JmpDists12, jd12, JMP_CNT * 96, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy JmpDists12 failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
	free(jd12);
//jmp1
	u64* buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
	}
	err = hipMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy Jumps1 failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
	free(buf);
//jmp2
	buf = (u64*)malloc(JMP_CNT * 96);
	u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
		memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
	}
	err = hipMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy Jumps2 failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
	free(buf);

	err = SetGpuParams(Kparams, jmp2_table);
	if (err != hipSuccess)
	{
		free(jmp2_table);
		printf("GPU %d, SetGpuParams failed: %s!\r\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
	free(jmp2_table);
//jmp3
	buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
	}
	err = hipMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy Jumps3 failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
	free(buf);

	printf("GPU %d: allocated %llu MB, %d kangaroos.\r\n", DeviceIndex, total_mem / (1024 * 1024), KangCnt);
	return true;
}

void RCGpuKang::Release()
{
	free(RndPnts);
	RndPnts = nullptr;
	free(DPs_out);
	DPs_out = nullptr;
	if (Stream)
		CheckHip(hipStreamSynchronize(Stream), DeviceIndex, "hipStreamSynchronize(release)");
#define RCK_HIP_FREE(member) \
	if (Kparams.member) { CheckHip(hipFree(Kparams.member), DeviceIndex, "hipFree(" #member ")"); Kparams.member = nullptr; }
	RCK_HIP_FREE(LoopedKangs)
	RCK_HIP_FREE(dbg_buf)
	RCK_HIP_FREE(LoopTable)
	RCK_HIP_FREE(LastPnts)
	RCK_HIP_FREE(L1S2)
	RCK_HIP_FREE(DPTable)
	RCK_HIP_FREE(JumpsList)
	RCK_HIP_FREE(Jumps3)
	RCK_HIP_FREE(Jumps2)
	RCK_HIP_FREE(Jumps1)
	RCK_HIP_FREE(DPs_out)
	RCK_HIP_FREE(L2)
	RCK_HIP_FREE(dists)
	RCK_HIP_FREE(Jumps12)
	RCK_HIP_FREE(JmpDists12)
#undef RCK_HIP_FREE
	if (ProfileAfterC)
	{
		CheckHip(hipEventDestroy(ProfileAfterC), DeviceIndex, "hipEventDestroy(profile C)");
		ProfileAfterC = nullptr;
	}
	if (ProfileAfterB)
	{
		CheckHip(hipEventDestroy(ProfileAfterB), DeviceIndex, "hipEventDestroy(profile B)");
		ProfileAfterB = nullptr;
	}
	if (ProfileAfterA)
	{
		CheckHip(hipEventDestroy(ProfileAfterA), DeviceIndex, "hipEventDestroy(profile A)");
		ProfileAfterA = nullptr;
	}
	if (ProfileStart)
	{
		CheckHip(hipEventDestroy(ProfileStart), DeviceIndex, "hipEventDestroy(profile start)");
		ProfileStart = nullptr;
	}
	if (Stream)
	{
		CheckHip(hipStreamDestroy(Stream), DeviceIndex, "hipStreamDestroy");
		Stream = nullptr;
	}
}

void RCGpuKang::Stop()
{
	StopFlag = true;
}

void RCGpuKang::DoRestartKangs()
{
	cr.Enter();
	if (lsToRestart.empty())
	{
		cr.Leave();
		return;
	}

	EcInt WildRange, x32;
	x32.Set(1);
	x32.ShiftLeft(Range - 5);
	WildRange.Set(0);
	for (int i = 0; i < 32 + 2; i++) // +2 to smooth edges, PntToSolve must be RealPntToSolve+x32
		WildRange.Add(x32);

	hipError_t err;
	u64 t0 = GetTickCount64();
	for (int i = 0; i < (int)lsToRestart.size(); i++)
	{
		int KangInd = lsToRestart[i];

		EcInt d;
		if (KangInd < KangCnt / 3)
			d.RndMax(x32); //TAME kangs
		else
		{
			d.RndMax(WildRange);
			d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		}
		memcpy(RndPnts[KangInd].priv, d.data, 32);

#ifdef DEBUG_MODE
		EcPoint pnt = ec.MultiplyG_Fast(d);
#else
		EcPoint pnt = ec.MultiplyG(d);
#endif
		if (KangInd >= KangCnt / 3)
			pnt = ec.AddPoints(pnt, PntWild);
		pnt.SaveToBuffer64((u8*)RndPnts[KangInd].x);

		////copy pnt to gpu
		err = hipMemcpy(Kparams.L2 + 4 * KangInd, RndPnts[KangInd].x, 32, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			printf("GPU %d, hipMemcpy failed: %s\n", DeviceIndex, hipGetErrorString(err));
			cr.Leave();
			return;
		}
		err = hipMemcpy(Kparams.L2 + 4 * KangCnt + 4 * KangInd, RndPnts[KangInd].y, 32, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			printf("GPU %d, hipMemcpy failed: %s\n", DeviceIndex, hipGetErrorString(err));
			cr.Leave();
			return;
		}
		err = hipMemcpy(Kparams.dists + 4 * KangInd, RndPnts[KangInd].priv, 24, hipMemcpyHostToDevice);
		if (err != hipSuccess)
		{
			printf("GPU %d, hipMemcpy failed: %s\n", DeviceIndex, hipGetErrorString(err));
			cr.Leave();
			return;
		}
	}

	lsToRestart.clear();
	cr.Leave();
//	printf("DoRestart %d ms\r\n", GetTickCount64() - t0);
}

void RCGpuKang::GenerateRndDistances()
{
	EcInt WildRange, x32;
	x32.Set(1);
	x32.ShiftLeft(Range - 5);
	WildRange.Set(0);
	for (int i = 0; i < 32 + 2; i++) // +2 to smooth edges, PntToSolve must be RealPntToSolve+x32
		WildRange.Add(x32);

	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		if (i < KangCnt / 3)
			d.RndMax(x32); //TAME kangs
		else
		{
			d.RndMax(WildRange);
			d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		}
		memcpy(RndPnts[i].priv, d.data, 24);
	}
}

bool RCGpuKang::Start()
{
	if (Failed)
		return false;

	hipError_t err;
	err = hipSetDevice(DeviceIndex);
	if (!CheckHip(err, DeviceIndex, "hipSetDevice"))
		return false;

	HalfRange.Set(1);
	HalfRange.ShiftLeft(Range - 1);
	PntHalfRange = ec.MultiplyG(HalfRange);
	NegPntHalfRange = PntHalfRange;
	NegPntHalfRange.y.NegModP();

	PntWild = PntToSolve; //to smooth edges PntToSolve = RealPnt+x32 (added in caller)
	PntWild.y.NegModP(); //negate

	RndPnts = (TPointPriv*)malloc(KangCnt * 96);
	GenerateRndDistances();
	// Calculate starting points on the GPU.
	u8 buf_PntWild[64];
	PntWild.SaveToBuffer64(buf_PntWild);
	for (int i = 0; i < KangCnt; i++)
	{
		if (i < KangCnt / 3)
			memset(RndPnts[i].x, 0, 64);
		else
			memcpy(RndPnts[i].x, buf_PntWild, 64);
	}

	u8* gpu_pnts = (u8*)malloc(96 * KangCnt);
	for (int i = 0; i < KangCnt; i++)
	{
		memcpy(gpu_pnts + 32 * i, RndPnts[i].x, 32);
		memcpy(gpu_pnts + 32 * i + 32 * KangCnt, RndPnts[i].y, 32);
		memcpy(gpu_pnts + 32 * i + 64 * KangCnt, RndPnts[i].priv, 32);
	}

	//copy to gpu
	err = hipMemcpy(Kparams.L2, gpu_pnts, KangCnt * 96, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		free(gpu_pnts);
		printf("GPU %d, hipMemcpy gpu_pnts failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
	if (ProfilingEnabled &&
		!CheckHip(hipEventRecord(ProfileStart, Stream), DeviceIndex, "hipEventRecord(KernelGen start)"))
	{
		free(gpu_pnts);
		return false;
	}
	err = LaunchKernelGen(Kparams, Stream);
	if (!CheckHip(err, DeviceIndex, "LaunchKernelGen") ||
		(ProfilingEnabled && !CheckHip(hipEventRecord(ProfileAfterA, Stream), DeviceIndex,
			"hipEventRecord(KernelGen stop)")) ||
		!CheckHip(hipStreamSynchronize(Stream), DeviceIndex, "hipStreamSynchronize(KernelGen)"))
	{
		free(gpu_pnts);
		return false;
	}
	if (ProfilingEnabled && !CheckHip(hipEventElapsedTime(&KernelGenMilliseconds,
		ProfileStart, ProfileAfterA), DeviceIndex, "hipEventElapsedTime(KernelGen)"))
	{
		free(gpu_pnts);
		return false;
	}
	err = hipMemcpy(Kparams.dists, gpu_pnts + 64 * KangCnt, KangCnt * 32, hipMemcpyHostToDevice);
	if (err != hipSuccess)
	{
		printf("GPU %d, hipMemcpy failed: %s\n", DeviceIndex, hipGetErrorString(err));
		return false;
	}
	free(gpu_pnts);

	err = hipMemset(Kparams.L1S2, 0,
		static_cast<size_t>(Kparams.BlockCnt) * Kparams.BlockSize * sizeof(u32));
	if (!CheckHip(err, DeviceIndex, "hipMemset(L1S2)"))
		return false;
	err = hipMemsetAsync(Kparams.dbg_buf, 0, 1024, Stream);
	if (!CheckHip(err, DeviceIndex, "hipMemsetAsync(dbg_buf)"))
		return false;
	err = hipMemsetAsync(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64), Stream);
	if (!CheckHip(err, DeviceIndex, "hipMemsetAsync(LoopTable)"))
		return false;
	return CheckHip(hipStreamSynchronize(Stream), DeviceIndex, "hipStreamSynchronize(start)");
}

#ifdef DEBUG_MODE
int RCGpuKang::Dbg_CheckKangs()
{
	u64 PartStride = static_cast<u64>(Kparams.KangCnt) * 32;
	u64* kangs = (u64*)malloc(Kparams.KangCnt * 64);
	u64* dists = (u64*)malloc(Kparams.KangCnt * 32);
	hipError_t err = hipMemcpy(kangs, Kparams.L2, Kparams.KangCnt * 64, hipMemcpyDeviceToHost);
	if (!CheckHip(err, DeviceIndex, "hipMemcpy(debug kangaroos)"))
	{
		free(kangs);
		free(dists);
		return -1;
	}
	err = hipMemcpy(dists, Kparams.dists, Kparams.KangCnt * 32, hipMemcpyDeviceToHost);
	if (!CheckHip(err, DeviceIndex, "hipMemcpy(debug distances)"))
	{
		free(kangs);
		free(dists);
		return -1;
	}
	int res = 0;
	for (int i = 0; i < KangCnt; i++)
	{
		EcPoint Pnt, p;
		memcpy(Pnt.x.data, &kangs[i * 4 + 0], 32);
		memcpy(Pnt.y.data, &kangs[i * 4 + 0 + PartStride / 8], 32);

		EcInt dist;
		dist.Set(0);
		memcpy(dist.data, &dists[i * 4], 24);
		bool neg = false;
		if (dist.data[2] >> 63)
		{
			neg = true;
			memset(((u8*)dist.data) + 24, 0xFF, 16);
			dist.Neg();
		}
		p = ec.MultiplyG_Fast(dist);
		if (neg)
			p.y.NegModP();
		if (i < KangCnt / 3)
			p = p;
		else
			p = ec.AddPoints(PntWild, p);
		if (!p.IsEqual(Pnt))
			res++;
	}
	free(kangs);
	free(dists);
	return res;
}

#endif

extern u32 gTotalErrors;

//executes in separate thread
void RCGpuKang::Execute()
{
	if (!CheckHip(hipSetDevice(DeviceIndex), DeviceIndex, "hipSetDevice"))
	{
		gTotalErrors++;
		return;
	}

	if (!Start())
	{
		gTotalErrors++;
		Release();
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
	hipError_t err;
	while (!StopFlag)
	{
		u64 t1 = GetTickCount64();
		err = hipMemsetAsync(Kparams.DPs_out, 0, 4, Stream);
		if (!CheckHip(err, DeviceIndex, "hipMemsetAsync(DPs_out)"))
			break;
		err = hipMemsetAsync(Kparams.DPTable, 0, KangCnt * sizeof(u32), Stream);
		if (!CheckHip(err, DeviceIndex, "hipMemsetAsync(DPTable)"))
			break;
		err = hipMemsetAsync(Kparams.LoopedKangs, 0, 8, Stream);
		if (!CheckHip(err, DeviceIndex, "hipMemsetAsync(LoopedKangs)"))
			break;

		if (ProfilingEnabled &&
			!CheckHip(hipEventRecord(ProfileStart, Stream), DeviceIndex, "hipEventRecord(KernelA start)"))
		{
			gTotalErrors++;
			break;
		}
		if (!CheckHip(LaunchKernelA(Kparams, Stream), DeviceIndex, "LaunchKernelA") ||
			(ProfilingEnabled && !CheckHip(hipEventRecord(ProfileAfterA, Stream), DeviceIndex,
				"hipEventRecord(KernelA stop)")) ||
			!CheckHip(LaunchKernelB(Kparams, Stream), DeviceIndex, "LaunchKernelB") ||
			(ProfilingEnabled && !CheckHip(hipEventRecord(ProfileAfterB, Stream), DeviceIndex,
				"hipEventRecord(KernelB stop)")) ||
			!CheckHip(LaunchKernelC(Kparams, Stream), DeviceIndex, "LaunchKernelC") ||
			(ProfilingEnabled && !CheckHip(hipEventRecord(ProfileAfterC, Stream), DeviceIndex,
				"hipEventRecord(KernelC stop)")))
		{
			gTotalErrors++;
			break;
		}
		int cnt;
		err = hipMemcpyAsync(&cnt, Kparams.DPs_out, 4, hipMemcpyDeviceToHost, Stream);
		if (!CheckHip(err, DeviceIndex, "hipMemcpyAsync(DP count)") ||
			!CheckHip(hipStreamSynchronize(Stream), DeviceIndex, "hipStreamSynchronize(iteration)"))
		{
			gTotalErrors++;
			break;
		}
		if (ProfilingEnabled)
		{
			float kernel_a_ms = 0.0f;
			float kernel_b_ms = 0.0f;
			float kernel_c_ms = 0.0f;
			if (!CheckHip(hipEventElapsedTime(&kernel_a_ms, ProfileStart, ProfileAfterA),
				DeviceIndex, "hipEventElapsedTime(KernelA)") ||
				!CheckHip(hipEventElapsedTime(&kernel_b_ms, ProfileAfterA, ProfileAfterB),
					DeviceIndex, "hipEventElapsedTime(KernelB)") ||
				!CheckHip(hipEventElapsedTime(&kernel_c_ms, ProfileAfterB, ProfileAfterC),
					DeviceIndex, "hipEventElapsedTime(KernelC)"))
			{
				gTotalErrors++;
				break;
			}
			KernelAMilliseconds.push_back(kernel_a_ms);
			KernelBMilliseconds.push_back(kernel_b_ms);
			KernelCMilliseconds.push_back(kernel_c_ms);
		}

		if (cnt >= MAX_DP_CNT)
		{
			cnt = MAX_DP_CNT;
			printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", DeviceIndex);
		}
		u64 pnt_cnt = (u64)KangCnt * Kparams.iter_cnt;

		if (cnt)
		{
			err = hipMemcpyAsync(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE,
				hipMemcpyDeviceToHost, Stream);
			if (!CheckHip(err, DeviceIndex, "hipMemcpyAsync(DPs)") ||
				!CheckHip(hipStreamSynchronize(Stream), DeviceIndex, "hipStreamSynchronize(DPs)"))
			{
				gTotalErrors++;
				break;
			}
			AddPointsToList(DPs_out, cnt, KangCnt, pnt_cnt, JumperInd);
		}

		//dbg
		if (!CheckHip(hipMemcpyAsync(dbg, Kparams.dbg_buf, 1024, hipMemcpyDeviceToHost, Stream),
			DeviceIndex, "hipMemcpyAsync(dbg)"))
			break;

		u32 lcnt;
		if (!CheckHip(hipMemcpyAsync(&lcnt, Kparams.LoopedKangs, 4, hipMemcpyDeviceToHost, Stream),
			DeviceIndex, "hipMemcpyAsync(loop count)") ||
			!CheckHip(hipStreamSynchronize(Stream), DeviceIndex, "hipStreamSynchronize(debug copies)"))
			break;
		//printf("GPU %d, Looped: %d\r\n", DeviceIndex, lcnt);

		DoRestartKangs();

		u64 t2 = GetTickCount64();
		u64 tm = t2 - t1;
		if (!tm)
			tm = 1;
		int cur_speed = (int)(pnt_cnt / (tm * 1000));
		if (ProfilingEnabled)
			EndToEndMKeys.push_back(static_cast<float>(pnt_cnt) / (static_cast<float>(tm) * 1000.0f));
		//printf("GPU %d kernel time %d ms, speed %d MH\r\n", DeviceIndex, (int)tm, cur_speed);

		SpeedStats[cur_stats_ind] = cur_speed;
		cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
		if ((iter % 300) == 0)
		{
			int corr_cnt = Dbg_CheckKangs();
			if (corr_cnt)
			{
				printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", DeviceIndex, corr_cnt);
				gTotalErrors++;
			}
			else
				printf("DBG: GPU %d, ALL KANGS OK!\r\n", DeviceIndex);
		}
		iter++;
#endif


	}

	if (ProfilingEnabled)
		PrintProfileSummary();
	Release();
}

void RCGpuKang::PrintProfileSummary() const
{
	const ProfileStats kernel_a = Summarize(KernelAMilliseconds);
	const ProfileStats kernel_b = Summarize(KernelBMilliseconds);
	const ProfileStats kernel_c = Summarize(KernelCMilliseconds);
	const ProfileStats end_to_end = Summarize(EndToEndMKeys);

	printf("GPU %d profile: KernelGen %.3f ms; KernelA %.3f ms; KernelB %.3f ms; "
		"KernelC %.3f ms; end-to-end %.3f MKeys/s (%zu samples)\n",
		DeviceIndex, KernelGenMilliseconds, kernel_a.median, kernel_b.median,
		kernel_c.median, end_to_end.median, KernelAMilliseconds.size());
	printf("RCK_PROFILE_JSON={\"schema\":1,\"device\":%d,\"blocks\":%u,"
		"\"threads\":%u,\"groups\":%u,\"iterations\":%u,\"kangaroos\":%u,"
		"\"kernel_a_table_mode\":%u,\"kernel_a_lds_bytes\":%u,"
		"\"state_layout\":%u,"
		"\"sample_count\":%zu,\"kernel_gen_ms\":%.6f,"
		"\"kernel_a\":{\"median_ms\":%.6f,\"mad_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f},"
		"\"kernel_b\":{\"median_ms\":%.6f,\"mad_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f},"
		"\"kernel_c\":{\"median_ms\":%.6f,\"mad_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f},"
		"\"end_to_end\":{\"median_mkeys_per_second\":%.6f,"
		"\"mad_mkeys_per_second\":%.6f,\"min_mkeys_per_second\":%.6f,"
		"\"max_mkeys_per_second\":%.6f}}\n",
		DeviceIndex, Kparams.BlockCnt, Kparams.BlockSize, Kparams.GroupCnt,
		Kparams.iter_cnt, Kparams.KangCnt, KernelATableMode, KernelALdsBytes,
		StateLayout, KernelAMilliseconds.size(), KernelGenMilliseconds,
		kernel_a.median, kernel_a.mad, kernel_a.minimum, kernel_a.maximum,
		kernel_b.median, kernel_b.mad, kernel_b.minimum, kernel_b.maximum,
		kernel_c.median, kernel_c.mad, kernel_c.minimum, kernel_c.maximum,
		end_to_end.median, end_to_end.mad, end_to_end.minimum, end_to_end.maximum);
}

void RCGpuKang::ToRestartKangaroo(int KangInd)
{
	cr.Enter();
	lsToRestart.push_back(KangInd);
	cr.Leave();
}

int RCGpuKang::GetStatsSpeed()
{
	int res = SpeedStats[0];
	for (int i = 1; i < STATS_WND_SIZE; i++)
		res += SpeedStats[i];
	return res / STATS_WND_SIZE;
}
