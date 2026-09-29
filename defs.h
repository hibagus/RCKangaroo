// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once 

#pragma warning(disable : 4996)

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;



// CPX compute partitioning on MI300X presents each of the 8 XCDs as a separate
// logical device, so a node of 8 physical GPUs enumerates as 64. The CUDA build
// keeps 32.
#ifdef __HIP_PLATFORM_AMD__
#define MAX_GPU_CNT			64
#else
#define MAX_GPU_CNT			32
#endif

//must be divisible by MD_LEN
#define STEP_CNT			1000

#define JMP_CNT				512

#define BLOCK_SIZE			256	

#ifdef __HIP_PLATFORM_AMD__
// Kangaroos per thread, i.e. the batched-inversion group size.
//
// Swept on MI300X (docs/CDNA_PHASE2_DESIGN.md): 8 -> 7871, 12 -> 8363,
// 24 -> 9325, 32 -> 9444 MKeys/s. Larger is better because one InvModP is
// amortised over the whole group and because more groups give the thread more
// independent work to interleave.
//
// 32 is the ceiling: L1S2 carries one bit per group in a u32.
#define PNT_GROUP_CNT		32
#else
#define PNT_GROUP_CNT		24
#endif

#ifdef __HIP_PLATFORM_AMD__
// How many u64 each entry of KernelA's LDS copy of the jmp1 table occupies.
// CDNA3's 64 KB per CU only affords the x-coordinates at 3 workgroups/CU;
// CDNA4's 160 KB affords x and y both, which removes a global read per point
// addition. Shared with the host so it can size the dynamic LDS request.
//
// Must be even - every access is a 16-byte-aligned ds_read_b128 - and it also
// picks which LDS bank an entry starts in, so it doubles as a bank-conflict
// knob. Override at build time to sweep it.
#define JMP1_LDS_STRIDE_CDNA3	4
#ifndef JMP1_LDS_STRIDE_CDNA4
#define JMP1_LDS_STRIDE_CDNA4	8
#endif
#endif

// kang type
#define TAME				0  // Tame kangs
#define WILD				1  // Wild kangs 

#define GPU_DP_SIZE			48
#define MAX_DP_CNT			(256 * 1024)

#define JMP_MASK			(JMP_CNT-1)
#define JMP_MASK_ADV		(2048 - 1) //including INV_FLAG and JMP2_FLAG

#define DPTABLE_MAX_CNT		16

#define MAX_CNT_LIST		(512 * 1024)

#define DP_FLAG				0x0800
#define INV_FLAG			0x0200
#define JMP2_FLAG			0x0400

#define MD_LEN				10

//#define DEBUG_MODE

//gpu kernel parameters
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
	///////////////////////////////////////////
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
	bool IsGenMode; //tames generation mode
	u32 KernelA_LDS_Size;
	u32 KernelB_LDS_Size;
	u32 KernelC_LDS_Size;
};

