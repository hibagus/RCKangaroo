// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

#include "rckangaroo/ec.hpp"
#include "rckangaroo/config.hpp"
#include "rckangaroo/gpu/kernel_params.hpp"
#include "rckangaroo/host_pipeline.hpp"

#define STATS_WND_SIZE	16

struct EcJMP
{
	EcPoint p;
	EcInt dist;
};

//96bytes size
struct TPointPriv
{
	u64 x[4];
	u64 y[4];
	u64 priv[4];
};

class RCGpuKang
{
private:
	using DistinguishedPointConsumer = void (*)(
		const u32* data, u32 count, u32 kangaroo_count,
		u64 operation_count, int worker_index);

	struct OutputSlot
	{
		u32* Device = nullptr;
		u32* Host = nullptr;
		hipEvent_t ComputeDone = nullptr;
		hipEvent_t TransferDone = nullptr;
		hipEvent_t ProfileStart = nullptr;
		hipEvent_t ProfileAfterA = nullptr;
		hipEvent_t ProfileAfterB = nullptr;
		u64 OperationCount = 0;
	};

	std::atomic<bool> StopFlag{false};
	EcPoint PntToSolve;
	int Range; //in bits
	int DP; //in bits
	Ec ec;

	CriticalSection cr;
	std::vector<int> lsToRestart; //list of kangs to restart
	void DoRestartKangs();

	TKparams Kparams{};
	std::array<OutputSlot, rckangaroo::DoubleBufferedOutputRing::slot_count> OutputSlots;
	rckangaroo::DoubleBufferedOutputRing OutputRing;
	std::thread ConsumerThread;
	DistinguishedPointConsumer PointConsumer = nullptr;

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	TPointPriv* RndPnts = nullptr;
	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;

	EcPoint PntWild;

	int cur_stats_ind;
	std::array<std::atomic<int>, STATS_WND_SIZE> SpeedStats{};
	u64 LastCompletionTick = 0;
	bool ProfilingEnabled = false;
	float KernelGenMilliseconds = 0.0f;
	std::vector<float> KernelAMilliseconds;
	std::vector<float> KernelBMilliseconds;
	std::vector<float> KernelCMilliseconds;
	std::vector<float> EndToEndMKeys;
	hipEvent_t ProfileGenStart = nullptr;
	hipEvent_t ProfileGenStop = nullptr;

	void GenerateRndDistances();
	bool Start();
	void Release();
	bool AllocateOutputRing();
	bool LaunchIteration(std::size_t slot);
	bool FinishIteration(std::size_t slot);
	void ConsumeOutputRing();
	void StopOutputConsumer(bool abort);
	void PrintProfileSummary() const;
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif

	hipStream_t Stream = nullptr;
	hipStream_t TransferStream = nullptr;
public:
	~RCGpuKang();

	int DeviceIndex = -1;
	int mpCnt;
	int KangCnt;
	int JumperInd;
	int PointGroupCnt = PNT_GROUP_CNT;
	int KernelStepCnt = STEP_CNT;
	u32 KernelATableMode = RCK_KERNEL_A_TABLE_MODE;
	u32 KernelALdsBytes = KERNEL_A_LDS_BYTES;
	u32 StateLayout = RCK_STATE_LAYOUT;
	bool Failed = false;
	int WorkerCpu = -1;
	int ConsumerCpu = -1;
	u64 InitializationSeed = 0;

	void ApplyArchitectureTuning(const char* architecture);
	int CalcKangCnt();
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();
	void ToRestartKangaroo(int KangInd);
	void SetDistinguishedPointConsumer(DistinguishedPointConsumer consumer)
	{
		PointConsumer = consumer;
	}

	u32 dbg[256];

	int GetStatsSpeed();
};
