#pragma once

#include <cstdint>
#include <string_view>

#include "rckangaroo/config.hpp"

namespace rckangaroo::hip
{
struct ArchitectureTuning
{
	unsigned point_group_count;
	unsigned step_count;
	unsigned kernel_a_table_mode;
	unsigned kernel_a_lds_bytes;
	unsigned state_layout;
};

inline constexpr unsigned KernelATableLdsBytes(unsigned mode)
{
	return mode == 1 ? 0 :
		(mode == 2 ? 16 : 8) * JMP_CNT * sizeof(std::uint64_t);
}

inline constexpr ArchitectureTuning SelectArchitectureTuning(
	std::string_view architecture)
{
	if (architecture.starts_with("gfx950"))
	{
		constexpr unsigned mode = RCK_USE_ARCH_TABLE_DEFAULTS
			? RCK_GFX950_KERNEL_A_TABLE_MODE : RCK_KERNEL_A_TABLE_MODE;
		constexpr unsigned layout = RCK_USE_ARCH_STATE_LAYOUT_DEFAULTS
			? RCK_GFX950_STATE_LAYOUT : RCK_STATE_LAYOUT;
		return {RCK_GFX950_POINT_GROUP_COUNT, RCK_GFX950_STEP_COUNT,
			mode, KernelATableLdsBytes(mode), layout};
	}
	if (architecture.starts_with("gfx942"))
	{
		constexpr unsigned mode = RCK_USE_ARCH_TABLE_DEFAULTS
			? RCK_GFX942_KERNEL_A_TABLE_MODE : RCK_KERNEL_A_TABLE_MODE;
		constexpr unsigned layout = RCK_USE_ARCH_STATE_LAYOUT_DEFAULTS
			? RCK_GFX942_STATE_LAYOUT : RCK_STATE_LAYOUT;
		return {RCK_GFX942_POINT_GROUP_COUNT, RCK_GFX942_STEP_COUNT,
			mode, KernelATableLdsBytes(mode), layout};
	}
	return {PNT_GROUP_CNT, STEP_CNT, RCK_KERNEL_A_TABLE_MODE,
		KernelATableLdsBytes(RCK_KERNEL_A_TABLE_MODE), RCK_STATE_LAYOUT};
}

static_assert(SelectArchitectureTuning("gfx942:sramecc+").point_group_count ==
	RCK_GFX942_POINT_GROUP_COUNT);
static_assert(SelectArchitectureTuning("gfx950:xnack-").point_group_count ==
	RCK_GFX950_POINT_GROUP_COUNT);
} // namespace rckangaroo::hip
