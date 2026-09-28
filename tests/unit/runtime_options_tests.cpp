#include "rckangaroo/runtime_options.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Expect(bool condition, const char* label)
{
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << label << '\n';
    }
}

rckangaroo::RuntimeOptionParseResult Parse(
    int argc,
    char* argv[],
    rckangaroo::RuntimeOptions& options,
    std::string& error,
    int& next_index)
{
    next_index = 2;
    return rckangaroo::ParseRuntimeOption(argv[1], argc, argv, next_index,
                                          options, error);
}

void TestSuccessfulOptions()
{
    rckangaroo::RuntimeOptions options;
    std::string error;
    int next_index = 0;

    char program[] = "rckangaroo";
    char seed_option[] = "--seed";
    char zero[] = "0";
    char* seed_argv[] = {program, seed_option, zero};
    Expect(Parse(3, seed_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::success,
           "parse --seed");
    Expect(options.seed_specified && options.seed == 0,
           "zero is a valid explicit seed");
    Expect(next_index == 3 && error.empty(), "seed consumes one value");

    char iterations_option[] = "-iterations";
    char iterations[] = "25";
    char* iterations_argv[] = {program, iterations_option, iterations};
    Expect(Parse(3, iterations_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::success,
           "parse legacy-style -iterations alias");
    Expect(options.benchmark_iterations == 25, "store benchmark iterations");

    char duration_option[] = "--duration";
    char duration[] = "3600";
    char* duration_argv[] = {program, duration_option, duration};
    Expect(Parse(3, duration_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::success,
           "parse --duration");
    Expect(options.duration_seconds == 3600, "store duration seconds");

    char groups_option[] = "--point-groups";
    char groups[] = "32";
    char* groups_argv[] = {program, groups_option, groups};
    Expect(Parse(3, groups_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::success,
           "parse --point-groups");
    Expect(options.point_groups == 32, "store point groups");

    char steps_option[] = "--kernel-steps";
    char steps[] = "2048";
    char* steps_argv[] = {program, steps_option, steps};
    Expect(Parse(3, steps_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::success,
           "parse --kernel-steps");
    Expect(options.kernel_steps == 2048, "store kernel steps");

    char gpu_option[] = "--gpu";
    char gpu_list[] = "0,3,12";
    char* gpu_argv[] = {program, gpu_option, gpu_list};
    Expect(Parse(3, gpu_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::success,
           "parse modern GPU list");
    Expect(options.gpu_selection_specified &&
               options.gpu_indices == std::vector<int>({0, 3, 12}),
           "store modern GPU list");

    char legacy_gpu_option[] = "-gpu";
    char compact_gpu_list[] = "035";
    char* compact_gpu_argv[] = {program, legacy_gpu_option, compact_gpu_list};
    Expect(Parse(3, compact_gpu_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::success,
           "parse compact legacy GPU list");
    Expect(options.gpu_indices == std::vector<int>({0, 3, 5}),
           "store compact legacy GPU list");

    char maximum[] = "18446744073709551615";
    char* maximum_argv[] = {program, seed_option, maximum};
    Expect(Parse(3, maximum_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::success,
           "parse maximum uint64 seed");
    Expect(options.seed == std::numeric_limits<std::uint64_t>::max(),
           "store maximum uint64 seed");
}

void TestRejectedOptions()
{
    rckangaroo::RuntimeOptions options;
    std::string error;
    int next_index = 0;

    char program[] = "rckangaroo";
    char unknown_option[] = "--unknown";
    char value[] = "1";
    char* unknown_argv[] = {program, unknown_option, value};
    Expect(Parse(3, unknown_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::not_runtime_option,
           "leave unknown options to the main parser");
    Expect(next_index == 2, "unknown option does not consume a value");

    char seed_option[] = "--seed";
    char* missing_argv[] = {program, seed_option};
    Expect(Parse(2, missing_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject missing seed");
    Expect(error == "missing value after --seed option", "missing-value error text");

    char duration_option[] = "--duration";
    char zero[] = "0";
    char* zero_argv[] = {program, duration_option, zero};
    Expect(Parse(3, zero_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject zero duration");

    char iterations_option[] = "--iterations";
    char negative[] = "-1";
    char* negative_argv[] = {program, iterations_option, negative};
    Expect(Parse(3, negative_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject negative iterations");

    char trailing[] = "10seconds";
    char* trailing_argv[] = {program, duration_option, trailing};
    Expect(Parse(3, trailing_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject trailing duration characters");

    char groups_option[] = "--point-groups";
    char odd_groups[] = "31";
    char* odd_groups_argv[] = {program, groups_option, odd_groups};
    Expect(Parse(3, odd_groups_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject odd point-group count");
    char too_many_groups[] = "34";
    char* too_many_groups_argv[] = {program, groups_option, too_many_groups};
    Expect(Parse(3, too_many_groups_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject point-group count above 32");

    char steps_option[] = "--kernel-steps";
    char too_many_steps[] = "65536";
    char* too_many_steps_argv[] = {program, steps_option, too_many_steps};
    Expect(Parse(3, too_many_steps_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject excessive kernel-step count");

    char overflow[] = "18446744073709551616";
    char* overflow_argv[] = {program, seed_option, overflow};
    Expect(Parse(3, overflow_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject uint64 overflow");

    char gpu_option[] = "--gpu";
    char duplicate_gpus[] = "0,3,3";
    char* duplicate_gpu_argv[] = {program, gpu_option, duplicate_gpus};
    Expect(Parse(3, duplicate_gpu_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject duplicate GPU index");

    char empty_gpu_token[] = "0,,3";
    char* empty_gpu_argv[] = {program, gpu_option, empty_gpu_token};
    Expect(Parse(3, empty_gpu_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject empty GPU-list token");

    char out_of_range_gpu[] = "32";
    char* out_of_range_gpu_argv[] = {program, gpu_option, out_of_range_gpu};
    Expect(Parse(3, out_of_range_gpu_argv, options, error, next_index) ==
               rckangaroo::RuntimeOptionParseResult::error,
           "reject GPU index above implementation limit");
}

void TestDerivedSeeds()
{
    const std::uint64_t first = rckangaroo::DeriveSeed(42, 0, 0);
    Expect(first == rckangaroo::DeriveSeed(42, 0, 0),
           "derived seed is deterministic");
    Expect(first != rckangaroo::DeriveSeed(42, 1, 0),
           "iteration selects a distinct seed");
    Expect(first != rckangaroo::DeriveSeed(42, 0, 1),
           "stream selects a distinct seed");
}

} // namespace

int main()
{
    TestSuccessfulOptions();
    TestRejectedOptions();
    TestDerivedSeeds();

    if (failures != 0) {
        std::cerr << failures << " runtime-option regression test(s) failed\n";
        return 1;
    }

    std::cout << "All runtime-option regression tests passed\n";
    return 0;
}
