#include "rckangaroo/host_pipeline.hpp"

#include <algorithm>

namespace rckangaroo {

DistinguishedPointCount ClampDistinguishedPointCount(
    std::uint64_t produced,
    std::uint32_t capacity)
{
    DistinguishedPointCount result;
    result.accepted = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(produced, capacity));
    result.dropped = produced - result.accepted;
    return result;
}

void DoubleBufferedOutputRing::Reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    states_.fill(SlotState::free);
    ready_.clear();
    closed_ = false;
    aborted_ = false;
    condition_.notify_all();
}

bool DoubleBufferedOutputRing::Acquire(std::size_t slot)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (slot >= slot_count) {
        return FailLocked();
    }
    condition_.wait(lock, [&] {
        return states_[slot] == SlotState::free || closed_ || aborted_;
    });
    if (closed_ || aborted_) {
        return false;
    }
    states_[slot] = SlotState::producer;
    return true;
}

bool DoubleBufferedOutputRing::Publish(std::size_t slot)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot >= slot_count || states_[slot] != SlotState::producer || aborted_) {
        return FailLocked();
    }
    states_[slot] = SlotState::ready;
    ready_.push_back(slot);
    condition_.notify_all();
    return true;
}

std::optional<std::size_t> DoubleBufferedOutputRing::Consume()
{
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] {
        return !ready_.empty() || closed_ || aborted_;
    });
    if (aborted_ || ready_.empty()) {
        return std::nullopt;
    }

    const std::size_t slot = ready_.front();
    ready_.pop_front();
    if (states_[slot] != SlotState::ready) {
        FailLocked();
        return std::nullopt;
    }
    states_[slot] = SlotState::consumer;
    return slot;
}

bool DoubleBufferedOutputRing::Release(std::size_t slot)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot >= slot_count || states_[slot] != SlotState::consumer) {
        return FailLocked();
    }
    states_[slot] = SlotState::free;
    condition_.notify_all();
    return true;
}

void DoubleBufferedOutputRing::Close()
{
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    condition_.notify_all();
}

void DoubleBufferedOutputRing::Abort()
{
    std::lock_guard<std::mutex> lock(mutex_);
    aborted_ = true;
    ready_.clear();
    condition_.notify_all();
}

bool DoubleBufferedOutputRing::IsAborted() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return aborted_;
}

std::size_t DoubleBufferedOutputRing::ReadyCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return ready_.size();
}

bool DoubleBufferedOutputRing::FailLocked()
{
    aborted_ = true;
    ready_.clear();
    condition_.notify_all();
    return false;
}

} // namespace rckangaroo
