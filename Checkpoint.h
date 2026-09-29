// Checkpoint / resume support for RCKangaroo.
//
// Not part of RetiredCoder's original software; added for the CDNA port so a
// multi-day run survives Ctrl+C, a job-scheduler SIGTERM, or a crash.
//
// WHAT A CHECKPOINT CONTAINS
//
//   1. the problem identity (pubkey, start offset, range, DP) so a checkpoint
//      can never be resumed onto a different search,
//   2. the DP database, in the same on-disk format the -tames option uses,
//   3. every kangaroo's current point and accumulated distance, per GPU.
//
// WHAT IT DELIBERATELY OMITS
//
// The loop-detection state - LoopTable, LastPnts and L1S2 - is not saved, and
// is zeroed on resume. That is the state a cold start begins in: Start() clears
// LoopTable and L1S2 and leaves LastPnts uninitialised, so the kernel already
// has to tolerate it. A kangaroo that was mid-loop-escape re-detects the loop
// within MD_LEN steps. Saving it would cost MD_LEN*(2*32+8) = 648 bytes per
// kangaroo - about 4.8 GB per MI300X - to avoid a few thousand wasted jumps.
//
// COST OF THE OMISSION WE DO MAKE
//
// Resuming without the kangaroo state (checkpoint from an older layout, or a
// GPU that failed to quiesce) costs one DP interval per kangaroo, i.e. about
// KangCnt * 2^DP operations, because every kangaroo restarts from a fresh
// random distance and its walk since the last DP is discarded. With the state
// restored the loss is only the work done since the last checkpoint.

#pragma once

#include "defs.h"
#include "Ec.h"

#include <stdio.h>

#define CKPT_MAGIC          "RCKANGAROO-CKPT"   // 15 chars + NUL = 16 bytes
#define CKPT_VERSION        1

// Set on a GPU descriptor whose kangaroo block is present in the file. A GPU
// that could not be quiesced in time is written with this clear, and is
// regenerated from random distances on resume.
#define CKPT_GPU_HAS_STATE  0x00000001

#ifdef _WIN32
    #define CKPT_FSEEK64    _fseeki64
    #define CKPT_FTELL64    _ftelli64
#else
    #define CKPT_FSEEK64    fseeko
    #define CKPT_FTELL64    ftello
#endif

#pragma pack(push, 1)

struct TCkptHeader
{
    char    Magic[16];
    u32     Version;
    u32     Flags;              // reserved, 0

    u32     Range;
    u32     DP;
    u32     GpuCnt;

    // Compile-time layout constants. The kangaroo blocks are flat dumps of
    // device memory whose indexing depends on all of these, and the TAME/WILD
    // split depends on each GPU's KangCnt, so a mismatch makes the saved state
    // unusable rather than merely suboptimal.
    u32     BlockSize;
    u32     GroupCnt;
    u32     JmpCnt;
    u32     StepCnt;
    u32     MdLen;

    u64     TotalOps;           // PntTotalOps at the moment of the snapshot
    u64     ElapsedMs;          // wall time already spent on this point
    u64     KangTotal;
    u64     DbOffset;           // file offset of the DP database section
    u64     DbRecCnt;           // informational, for the resume message

    u8      PntToSolve[64];     // the offset-adjusted point the kangaroos walk on
    u8      PubKey[64];         // as given on the command line, for diagnostics
    u8      Start[40];          // gStart.data

    u64     Reserved[8];
};

struct TCkptGpuDesc
{
    u32     CudaIndex;
    u32     KangCnt;
    u32     DescFlags;          // CKPT_GPU_HAS_STATE
    u32     Reserved;
    u64     Offset;             // file offset of this GPU's 96*KangCnt block
    u64     Bytes;              // 96 * KangCnt
};

#pragma pack(pop)

// Configuration, from the command line.
void        Ckpt_Configure(const char* fn, int interval_min);
bool        Ckpt_IsEnabled();
const char* Ckpt_FileName();
int         Ckpt_IntervalSec();
bool        Ckpt_FileExists();
void        Ckpt_Disable();

// Reads the header and the DP database, validates the checkpoint against the
// current run, and hands each GPU the file offset of its kangaroo block. Call
// after every GPU's Prepare() has run and before the worker threads start.
// Returns false and leaves the run untouched if the file cannot be used.
bool Ckpt_Load(EcPoint& PntToSolve, int Range, int DP, u64* pTotalOps, u64* pElapsedMs);

// Quiesces every GPU, drains the pending DPs into the database, writes the file
// atomically, and lets the GPUs run again. Safe to call from the main thread
// while the workers are running; that is the only supported caller.
bool Ckpt_Save(u64 elapsed_ms);
