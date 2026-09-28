#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace rckangaroo {

struct RuntimeOptions {
    std::uint64_t seed = 0;
    std::uint64_t benchmark_iterations = 0;
    std::uint64_t duration_seconds = 0;
    std::uint64_t kernel_steps = 0;
    std::uint64_t point_groups = 0;
    bool seed_specified = false;
};

enum class RuntimeOptionParseResult {
    not_runtime_option,
    success,
    error,
};

RuntimeOptionParseResult ParseRuntimeOption(
    std::string_view argument,
    int argc,
    char* const argv[],
    int& next_index,
    RuntimeOptions& options,
    std::string& error);

std::uint64_t DeriveSeed(std::uint64_t base_seed,
                         std::uint64_t iteration,
                         std::uint64_t stream);

} // namespace rckangaroo
