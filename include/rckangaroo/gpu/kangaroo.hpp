// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include <hip/hip_runtime.h>

#include <vector>

#include "rckangaroo/ec.hpp"
#include "rckangaroo/config.hpp"
#include "rckangaroo/gpu/kernel_params.hpp"

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
	bool StopFlag;
	EcPoint PntToSolve;
	int Range; //in bits
	int DP; //in bits
	Ec ec;

	CriticalSection cr;
	std::vector<int> lsToRestart; //list of kangs to restart
	void DoRestartKangs();

	u32* DPs_out = nullptr;
	TKparams Kparams{};

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	TPointPriv* RndPnts = nullptr;
	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;

	EcPoint PntWild;

	int cur_stats_ind;
	int SpeedStats[STATS_WND_SIZE];
	bool ProfilingEnabled = false;
	float KernelGenMilliseconds = 0.0f;
	std::vector<float> KernelAMilliseconds;
	std::vector<float> KernelBMilliseconds;
	std::vector<float> KernelCMilliseconds;
	std::vector<float> EndToEndMKeys;
	hipEvent_t ProfileStart = nullptr;
	hipEvent_t ProfileAfterA = nullptr;
	hipEvent_t ProfileAfterB = nullptr;
	hipEvent_t ProfileAfterC = nullptr;

	void GenerateRndDistances();
	bool Start();
	void Release();
	void PrintProfileSummary() const;
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif

	hipStream_t Stream = nullptr;
public:
	int DeviceIndex = -1;
	int mpCnt;
	int KangCnt;
	int JumperInd;
	bool Failed;

	int CalcKangCnt();
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();
	void ToRestartKangaroo(int KangInd);

	u32 dbg[256];

	int GetStatsSpeed();
};
