#pragma once

#include "defs.h"

#include <cstddef>
#include <type_traits>

static_assert(sizeof(void*) == 8, "TKparams requires a 64-bit host ABI");
static_assert(std::is_standard_layout_v<TKparams>, "TKparams must remain standard-layout");
static_assert(std::is_same_v<decltype(TKparams::IsGenMode), u32>,
              "TKparams::IsGenMode must remain fixed-width");
static_assert(alignof(TKparams) == 8, "Unexpected TKparams alignment");
static_assert(sizeof(TKparams) == 192, "Unexpected TKparams size");

static_assert(offsetof(TKparams, L2) == 0);
static_assert(offsetof(TKparams, Jumps12) == 8);
static_assert(offsetof(TKparams, DPTable) == 16);
static_assert(offsetof(TKparams, Reserved1) == 24);
static_assert(offsetof(TKparams, JumpsList) == 32);
static_assert(offsetof(TKparams, LastPnts) == 40);
static_assert(offsetof(TKparams, dbg_buf) == 48);
static_assert(offsetof(TKparams, L1S2) == 56);
static_assert(offsetof(TKparams, Reserved2) == 64);
static_assert(offsetof(TKparams, iter_cnt) == 72);
static_assert(offsetof(TKparams, BlockCnt) == 76);
static_assert(offsetof(TKparams, StopThr) == 80);
static_assert(offsetof(TKparams, dp_mask) == 84);
static_assert(offsetof(TKparams, DPs_out) == 88);
static_assert(offsetof(TKparams, LoopTable) == 96);
static_assert(offsetof(TKparams, LoopedKangs) == 104);
static_assert(offsetof(TKparams, dists) == 112);
static_assert(offsetof(TKparams, JmpDists12) == 120);
static_assert(offsetof(TKparams, KangCnt) == 128);
static_assert(offsetof(TKparams, Jumps1) == 136);
static_assert(offsetof(TKparams, Jumps2) == 144);
static_assert(offsetof(TKparams, Jumps3) == 152);
static_assert(offsetof(TKparams, BlockSize) == 160);
static_assert(offsetof(TKparams, GroupCnt) == 164);
static_assert(offsetof(TKparams, DP) == 168);
static_assert(offsetof(TKparams, IsGenMode) == 176);
static_assert(offsetof(TKparams, KernelA_LDS_Size) == 180);
static_assert(offsetof(TKparams, KernelB_LDS_Size) == 184);
static_assert(offsetof(TKparams, KernelC_LDS_Size) == 188);
