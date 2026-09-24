#pragma once
// API-level admission control (PRD §12 queue caps, TRD §48): at most `max_active`
// generations run; at most `max_queue` wait for a slot. A full queue is rejected at once
// (429); a queued request that waits past its timeout is rejected (503).

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace halo::api {

class Admission {
public:
    enum class Result { Admitted, QueueFull, Timeout, Stopped };

    Admission(std::size_t max_active, std::size_t max_queue) : max_active_(max_active), max_queue_(max_queue) {}

    [[nodiscard]] Result acquire(std::chrono::milliseconds timeout);
    void release();
    /// Wakes every waiter with Stopped; later acquire() calls return Stopped.
    void shutdown();

    [[nodiscard]] std::size_t active() const;
    [[nodiscard]] std::size_t queued() const;

private:
    const std::size_t max_active_;
    const std::size_t max_queue_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::size_t active_ = 0;
    std::size_t queued_ = 0;
    bool stopped_ = false;
};

/// RAII slot; releases on destruction.
class AdmissionSlot {
public:
    explicit AdmissionSlot(Admission& a) noexcept : a_(&a) {}
    AdmissionSlot(const AdmissionSlot&) = delete;
    AdmissionSlot& operator=(const AdmissionSlot&) = delete;
    ~AdmissionSlot() { a_->release(); }

private:
    Admission* a_;
};

}  // namespace halo::api
