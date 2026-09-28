#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>

namespace rckangaroo {

struct DistinguishedPointCount {
    std::uint32_t accepted = 0;
    std::uint64_t dropped = 0;
};

DistinguishedPointCount ClampDistinguishedPointCount(
    std::uint64_t produced,
    std::uint32_t capacity);

class DoubleBufferedOutputRing {
public:
    static constexpr std::size_t slot_count = 2;

    DoubleBufferedOutputRing() = default;
    DoubleBufferedOutputRing(const DoubleBufferedOutputRing&) = delete;
    DoubleBufferedOutputRing& operator=(const DoubleBufferedOutputRing&) = delete;

    void Reset();
    bool Acquire(std::size_t slot);
    bool Publish(std::size_t slot);
    std::optional<std::size_t> Consume();
    bool Release(std::size_t slot);
    void Close();
    void Abort();

    bool IsAborted() const;
    std::size_t ReadyCount() const;

private:
    enum class SlotState {
        free,
        producer,
        ready,
        consumer,
    };

    bool FailLocked();

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::array<SlotState, slot_count> states_{};
    std::deque<std::size_t> ready_;
    bool closed_ = false;
    bool aborted_ = false;
};

} // namespace rckangaroo
