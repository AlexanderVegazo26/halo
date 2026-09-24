#include "halo/backends/cpu/thread_pool.h"

#include <algorithm>

#include "halo/core/error.h"

namespace halo::cpu {

namespace {
// > 0 while this thread is executing a range of some pool (a worker, or a caller running
// range 0). Nested parallel_for then runs inline instead of re-entering a pool.
thread_local int tls_pool_depth = 0;

struct DepthGuard {
    DepthGuard() noexcept { ++tls_pool_depth; }
    ~DepthGuard() { --tls_pool_depth; }
    DepthGuard(const DepthGuard&) = delete;
    DepthGuard& operator=(const DepthGuard&) = delete;
};
}  // namespace

ThreadPool::ThreadPool(std::size_t n_threads) : n_threads_(n_threads) {
    HALO_CHECK(n_threads > 0, ErrorCode::Config, "ThreadPool: n_threads must be > 0");
    workers_.reserve(n_threads - 1);
    for (std::size_t i = 1; i < n_threads; ++i) {
        workers_.emplace_back([this, i](const std::stop_token& st) { worker_loop(st, i); });
    }
}

ThreadPool::~ThreadPool() {
    for (auto& w : workers_) w.request_stop();
    work_cv_.notify_all();
    workers_.clear();  // joins
}

std::size_t ThreadPool::default_threads() noexcept {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : static_cast<std::size_t>(hc);
}

void ThreadPool::execute_part(const Job& job, std::size_t part) noexcept {
    const auto [b, e] = partition_range(job.n, job.parts, part);
    try {
        DepthGuard depth;
        job.thunk(job.ctx, b, e);
    } catch (...) {
        errors_[part] = std::current_exception();
    }
}

void ThreadPool::run(std::size_t n, Thunk thunk, void* ctx) {
    if (n == 0) return;
    if (tls_pool_depth > 0 || n_threads_ == 1 || n == 1) {
        DepthGuard depth;
        thunk(ctx, 0, n);
        return;
    }
    std::lock_guard dispatch(dispatch_mu_);
    Job job{thunk, ctx, n, std::min(n_threads_, n)};
    errors_.assign(job.parts, nullptr);
    {
        std::lock_guard lk(mu_);
        job_ = job;
        pending_ = job.parts - 1;
        ++generation_;
    }
    work_cv_.notify_all();
    execute_part(job, 0);
    {
        std::unique_lock lk(mu_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
    }
    for (const auto& e : errors_) {
        if (e) std::rethrow_exception(e);
    }
}

void ThreadPool::worker_loop(const std::stop_token& stop, std::size_t index) {
    std::uint64_t seen = 0;
    while (true) {
        Job job;
        {
            std::unique_lock lk(mu_);
            if (!work_cv_.wait(lk, stop, [&] { return generation_ != seen; })) return;
            seen = generation_;
            job = job_;
        }
        if (index >= job.parts) continue;  // not needed for this job
        execute_part(job, index);
        std::lock_guard lk(mu_);
        if (--pending_ == 0) done_cv_.notify_one();
    }
}

}  // namespace halo::cpu
