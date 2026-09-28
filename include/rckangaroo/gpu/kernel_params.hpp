#pragma once

#include "rckangaroo/types.hpp"

struct TKparams
{
    u64* L2;
    u32* Jumps12;
    u32* DPTable;
    u32* Reserved1;
    u64* JumpsList;
    u64* LastPnts;
    u32* dbg_buf;
    u32* L1S2;
    u64* Reserved2;

    u32 iter_cnt;
    u32 BlockCnt;
    u32 StopThr;
    u32 dp_mask;

    u32* DPs_out;
    u64* LoopTable;
    u32* LoopedKangs;
    u64* dists;
    u64* JmpDists12;
    u32 KangCnt;
    u64* Jumps1;
    u64* Jumps2;
    u64* Jumps3;
    u32 BlockSize;
    u32 GroupCnt;
    u64 DP;
    u32 IsGenMode;
    u32 KernelA_LDS_Size;
    u32 KernelB_LDS_Size;
    u32 KernelC_LDS_Size;
};
