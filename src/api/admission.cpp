#include "admission.h"

namespace halo::api {

Admission::Result Admission::acquire(std::chrono::milliseconds timeout) {
    std::unique_lock lk(mu_);
    if (stopped_) return Result::Stopped;
    if (active_ < max_active_ && queued_ == 0) {
        ++active_;
        return Result::Admitted;
    }
    if (queued_ >= max_queue_) return Result::QueueFull;
    ++queued_;
    const bool ok = cv_.wait_for(lk, timeout, [&] { return stopped_ || active_ < max_active_; });
    --queued_;
    if (stopped_) {
        cv_.notify_all();
        return Result::Stopped;
    }
    if (!ok) return Result::Timeout;
    ++active_;
    return Result::Admitted;
}

void Admission::release() {
    {
        std::lock_guard lk(mu_);
        if (active_ > 0) --active_;
    }
    cv_.notify_one();
}

void Admission::shutdown() {
    {
        std::lock_guard lk(mu_);
        stopped_ = true;
    }
    cv_.notify_all();
}

std::size_t Admission::active() const {
    std::lock_guard lk(mu_);
    return active_;
}

std::size_t Admission::queued() const {
    std::lock_guard lk(mu_);
    return queued_;
}

}  // namespace halo::api
