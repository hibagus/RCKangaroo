#include "rckangaroo/runtime_options.hpp"

#include <charconv>
#include <system_error>

namespace rckangaroo {
namespace {

bool IsOption(std::string_view argument, std::string_view name)
{
    return argument == name ||
           (argument.size() + 1 == name.size() && name.starts_with("--") &&
            argument == name.substr(1));
}

bool ParseUnsigned(std::string_view text, std::uint64_t& value)
{
    if (text.empty()) {
        return false;
    }

    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result = std::from_chars(begin, end, value, 10);
    return result.ec == std::errc{} && result.ptr == end;
}

RuntimeOptionParseResult ParseValue(
    std::string_view option_name,
    int argc,
    char* const argv[],
    int& next_index,
    std::uint64_t& destination,
    bool allow_zero,
    std::string& error)
{
    if (next_index >= argc) {
        error = "missing value after " + std::string(option_name) + " option";
        return RuntimeOptionParseResult::error;
    }

    const std::string_view value_text(argv[next_index]);
    std::uint64_t value = 0;
    if (!ParseUnsigned(value_text, value) || (!allow_zero && value == 0)) {
        error = "invalid value for " + std::string(option_name) + " option: " +
                std::string(value_text);
        return RuntimeOptionParseResult::error;
    }

    destination = value;
    ++next_index;
    return RuntimeOptionParseResult::success;
}

} // namespace

RuntimeOptionParseResult ParseRuntimeOption(
    std::string_view argument,
    int argc,
    char* const argv[],
    int& next_index,
    RuntimeOptions& options,
    std::string& error)
{
    error.clear();
    if (IsOption(argument, "--seed")) {
        const RuntimeOptionParseResult result =
            ParseValue("--seed", argc, argv, next_index, options.seed, true, error);
        if (result == RuntimeOptionParseResult::success) {
            options.seed_specified = true;
        }
        return result;
    }
    if (IsOption(argument, "--iterations")) {
        return ParseValue("--iterations", argc, argv, next_index,
                          options.benchmark_iterations, false, error);
    }
    if (IsOption(argument, "--duration")) {
        return ParseValue("--duration", argc, argv, next_index,
                          options.duration_seconds, false, error);
    }
    return RuntimeOptionParseResult::not_runtime_option;
}

std::uint64_t DeriveSeed(std::uint64_t base_seed,
                         std::uint64_t iteration,
                         std::uint64_t stream)
{
    constexpr std::uint64_t golden_ratio = 0x9E3779B97F4A7C15ULL;
    std::uint64_t value =
        base_seed + golden_ratio * (iteration * 2U + stream + 1U);
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

} // namespace rckangaroo
