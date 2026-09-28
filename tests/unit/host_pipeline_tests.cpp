#include "rckangaroo/host_pipeline.hpp"
#include "rckangaroo/host_topology.hpp"
#include "rckangaroo/utils.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
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

void TestOutputCountClamp()
{
    const rckangaroo::DistinguishedPointCount below =
        rckangaroo::ClampDistinguishedPointCount(17, 32);
    Expect(below.accepted == 17 && below.dropped == 0,
           "DP count below capacity is unchanged");

    const rckangaroo::DistinguishedPointCount overflow =
        rckangaroo::ClampDistinguishedPointCount(45, 32);
    Expect(overflow.accepted == 32 && overflow.dropped == 13,
           "DP overflow is clamped and counted");
}

void TestOutputRingLifecycle()
{
    rckangaroo::DoubleBufferedOutputRing ring;
    ring.Reset();
    Expect(ring.Acquire(0), "producer acquires a free output slot");
    Expect(ring.Publish(0), "producer publishes a completed output slot");
    Expect(ring.ReadyCount() == 1, "published slot is visible to consumer");
    const std::optional<std::size_t> slot = ring.Consume();
    Expect(slot && *slot == 0, "consumer receives published slot");
    Expect(ring.Release(0), "consumer releases output slot");
    ring.Close();
    Expect(!ring.Consume(), "closed and drained ring stops consumer");
    Expect(!ring.Acquire(1), "closed ring rejects new producer work");
}

void TestOutputRingBackpressure()
{
    using namespace std::chrono_literals;
    rckangaroo::DoubleBufferedOutputRing ring;
    ring.Reset();
    Expect(ring.Acquire(0) && ring.Publish(0),
           "prepare occupied slot for backpressure test");
    const std::optional<std::size_t> slot = ring.Consume();
    Expect(slot && *slot == 0, "consumer owns slot during backpressure test");

    std::future<bool> acquire = std::async(std::launch::async, [&ring] {
        return ring.Acquire(0);
    });
    Expect(acquire.wait_for(20ms) == std::future_status::timeout,
           "producer waits instead of overwriting consumer-owned slot");
    Expect(ring.Release(0), "release wakes blocked producer");
    Expect(acquire.wait_for(1s) == std::future_status::ready && acquire.get(),
           "blocked producer resumes after release");
    ring.Abort();
}

void TestOutputRingFailure()
{
    rckangaroo::DoubleBufferedOutputRing ring;
    ring.Reset();
    Expect(!ring.Publish(1), "invalid state transition is rejected");
    Expect(ring.IsAborted(), "invalid state transition aborts ring");
    Expect(!ring.Consume(), "aborted ring wakes and stops consumer");

    ring.Reset();
    ring.Abort();
    Expect(!ring.Acquire(0), "aborted ring wakes and stops producer");
}

void TestCpuLists()
{
    std::vector<unsigned> cpus;
    std::string error;
    Expect(rckangaroo::ParseCpuList("0-2,5,7-8", cpus, error),
           "parse Linux CPU list");
    Expect(cpus == std::vector<unsigned>({0, 1, 2, 5, 7, 8}),
           "expand and order Linux CPU list");
    Expect(rckangaroo::SelectTopologyCpu(cpus, 1, false) == 2,
           "select second GPU worker CPU");
    Expect(rckangaroo::SelectTopologyCpu(cpus, 1, true) == 5,
           "select second GPU consumer CPU");
    Expect(!rckangaroo::ParseCpuList("4-2", cpus, error),
           "reject descending CPU range");
    Expect(!rckangaroo::ParseCpuList("0,", cpus, error),
           "reject trailing CPU token");
}

using TestRecord = std::array<u8, 35>;

TestRecord MakeRecord(std::uint32_t index)
{
    TestRecord record{};
    record[0] = static_cast<u8>(index % 8U);
    record[1] = static_cast<u8>((index / 8U) % 8U);
    record[2] = static_cast<u8>((index / 64U) % 8U);
    for (std::size_t byte = 3; byte < record.size(); ++byte) {
        record[byte] = static_cast<u8>((index * 37U + byte * 13U) & 0xffU);
    }
    return record;
}

void TestConcurrentDatabaseShards()
{
    constexpr std::uint32_t record_count = 512;
    auto database = std::make_unique<TFastBase>();
    std::atomic<int> insert_errors{0};
    std::vector<std::thread> threads;
    for (std::uint32_t worker = 0; worker < 8; ++worker) {
        threads.emplace_back([&, worker] {
            for (std::uint32_t index = worker; index < record_count; index += 8) {
                TestRecord record = MakeRecord(index);
                std::array<u8, 32> existing{};
                if (database->FindOrAddDataBlockConcurrent(
                        record.data(), existing.data()) !=
                    TFastBase::InsertResult::inserted) {
                    insert_errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    Expect(insert_errors.load(std::memory_order_relaxed) == 0,
           "concurrent unique inserts succeed");
    Expect(database->GetBlockCnt() == record_count,
           "concurrent insert count is exact");

    threads.clear();
    std::atomic<int> lookup_errors{0};
    for (std::uint32_t worker = 0; worker < 8; ++worker) {
        threads.emplace_back([&, worker] {
            for (std::uint32_t index = worker; index < record_count; index += 8) {
                TestRecord record = MakeRecord(index);
                std::array<u8, 32> existing{};
                if (database->FindOrAddDataBlockConcurrent(
                        record.data(), existing.data()) !=
                        TFastBase::InsertResult::found ||
                    !std::equal(existing.begin(), existing.end(), record.begin() + 3)) {
                    lookup_errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    Expect(lookup_errors.load(std::memory_order_relaxed) == 0,
           "concurrent duplicate lookups return stable records");
    Expect(database->GetBlockCnt() == record_count,
           "duplicate lookups do not change count");
}

} // namespace

int main()
{
    TestOutputCountClamp();
    TestOutputRingLifecycle();
    TestOutputRingBackpressure();
    TestOutputRingFailure();
    TestCpuLists();
    TestConcurrentDatabaseShards();

    if (failures != 0) {
        std::cerr << failures << " host-pipeline regression test(s) failed\n";
        return 1;
    }
    std::cout << "All host-pipeline regression tests passed\n";
    return 0;
}
