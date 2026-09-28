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

bool AppendGpuIndex(std::string_view text,
                    std::vector<int>& gpu_indices,
                    std::string& error)
{
    std::uint64_t value = 0;
    if (!ParseUnsigned(text, value) || value >= 32) {
        error = "GPU index must be between 0 and 31: " + std::string(text);
        return false;
    }
    const int index = static_cast<int>(value);
    for (const int existing : gpu_indices) {
        if (existing == index) {
            error = "duplicate GPU index: " + std::to_string(index);
            return false;
        }
    }
    gpu_indices.push_back(index);
    return true;
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

bool ParseGpuList(std::string_view text,
                  bool legacy_compact,
                  std::vector<int>& gpu_indices,
                  std::string& error)
{
    gpu_indices.clear();
    error.clear();
    if (text.empty()) {
        error = "GPU list is empty";
        return false;
    }

    if (legacy_compact && text.find(',') == std::string_view::npos &&
        text.size() > 1) {
        for (const char character : text) {
            if (character < '0' || character > '9' ||
                !AppendGpuIndex(std::string_view(&character, 1), gpu_indices,
                                error)) {
                if (error.empty()) {
                    error = "invalid compact GPU list: " + std::string(text);
                }
                return false;
            }
        }
        return true;
    }

    std::size_t offset = 0;
    while (offset < text.size()) {
        const std::size_t comma = text.find(',', offset);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        if (!AppendGpuIndex(text.substr(offset, end - offset), gpu_indices, error)) {
            return false;
        }
        if (comma == std::string_view::npos) {
            break;
        }
        offset = comma + 1;
        if (offset == text.size()) {
            error = "GPU list has an empty trailing token";
            return false;
        }
    }
    return true;
}

RuntimeOptionParseResult ParseRuntimeOption(
    std::string_view argument,
    int argc,
    char* const argv[],
    int& next_index,
    RuntimeOptions& options,
    std::string& error)
{
    error.clear();
    if (argument == "--gpu" || argument == "-gpu") {
        if (next_index >= argc) {
            error = "missing value after --gpu option";
            return RuntimeOptionParseResult::error;
        }
        if (!ParseGpuList(argv[next_index], argument == "-gpu",
                          options.gpu_indices, error)) {
            return RuntimeOptionParseResult::error;
        }
        options.gpu_selection_specified = true;
        ++next_index;
        return RuntimeOptionParseResult::success;
    }
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
    if (IsOption(argument, "--kernel-steps")) {
        const RuntimeOptionParseResult result =
            ParseValue("--kernel-steps", argc, argv, next_index,
                       options.kernel_steps, false, error);
        if (result == RuntimeOptionParseResult::success &&
            options.kernel_steps > 65535) {
            error = "--kernel-steps must be between 1 and 65535";
            return RuntimeOptionParseResult::error;
        }
        return result;
    }
    if (IsOption(argument, "--point-groups")) {
        const RuntimeOptionParseResult result =
            ParseValue("--point-groups", argc, argv, next_index,
                       options.point_groups, false, error);
        if (result == RuntimeOptionParseResult::success &&
            (options.point_groups < 2 || options.point_groups > 32 ||
             (options.point_groups & 1U) != 0)) {
            error = "--point-groups must be an even value between 2 and 32";
            return RuntimeOptionParseResult::error;
        }
        return result;
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
