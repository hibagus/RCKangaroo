// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include "Ec.h"
#ifdef __HIP_PLATFORM_AMD__
#include "cdna/cuda_compat.h"
#else
#include "CallCubin.h"
#endif

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

	//Checkpoint support. PauseFlag asks Execute() to park at its next batch
	//boundary, which is the only point where the device is quiescent: the
	//blocking DPs_out copy has already synchronised the previous batch and the
	//next one has not launched. QuiescedFlag reports that it is parked, or that
	//the worker never started / already finished, so the main thread never waits
	//on a thread that is not coming.
	volatile bool PauseFlag = false;
	volatile bool QuiescedFlag = true;	//not running yet, so nothing to wait for
	volatile bool DevMemValid = false;	//device allocations are live and readable
	char CkptFileName[1024] = {0};
	u64 CkptKangOffset = 0;
	bool CkptResume = false;
	bool RestoreKangsFromCkpt();
	//guards the device allocations against being freed by the worker thread
	//while the main thread is reading them for a checkpoint
	CriticalSection ckpt_cr;
	EcPoint PntToSolve;
	int Range; //in bits
	int DP; //in bits
	Ec ec;

	CriticalSection cr;
	std::vector<int> lsToRestart; //list of kangs to restart
	void DoRestartKangs();

	u32* DPs_out;
	TKparams Kparams;

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	TPointPriv* RndPnts;
	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;

	EcPoint PntWild;

	int cur_stats_ind;
	int SpeedStats[STATS_WND_SIZE];

	u64 Inv_DataSize;

	void GenerateRndDistances();
	bool GenerateStartPoints();
	bool Start();
	void Release();
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif

#ifndef __HIP_PLATFORM_AMD__
	// Prebuilt NVIDIA SASS "turbo" kernels, loaded through the CUDA driver API.
	// No AMD counterpart: the CDNA build always uses the compiled HIP kernels.
	TCubinCall cc;
	void Asm_CallGpuKernelAB();
#endif
public:
	u64 persistingL2CacheMaxSize;
	int CudaIndex; //gpu index in cuda
	int mpCnt;
	int KangCnt;
	int JumperInd;
	bool Failed;

	bool Is5xxx;
	int sm_inv_cnt; //number of SMs used for inverse calculation

	int CalcKangCnt();
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();

	//Checkpoint interface, all called from the main thread.
	void SetResumeSource(const char* fn, u64 offset);
	void RequestPause();
	void ReleasePause();
	bool IsQuiesced();
	bool IsStateReadable();
	bool SaveKangsToStream(FILE* fp);

	void Execute();
	void ToRestartKangaroo(int KangInd);

	u32 dbg[256];

	int GetStatsSpeed();
};
