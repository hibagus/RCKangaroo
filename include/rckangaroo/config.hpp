#pragma once

#define MAX_GPU_CNT 32

// Must be divisible by MD_LEN.
#define STEP_CNT 1000
#define JMP_CNT 512
#define BLOCK_SIZE 256
#define PNT_GROUP_CNT 24

#define TAME 0
#define WILD 1

#define GPU_DP_SIZE 48
#define MAX_DP_CNT (256 * 1024)
#define JMP_MASK (JMP_CNT - 1)
#define JMP_MASK_ADV (2048 - 1)
#define DPTABLE_MAX_CNT 16
#define MAX_CNT_LIST (512 * 1024)

#define DP_FLAG 0x0800
#define INV_FLAG 0x0200
#define JMP2_FLAG 0x0400

#define MD_LEN 10

// #define DEBUG_MODE
