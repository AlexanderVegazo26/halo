#pragma once
// HALO CPU reference backend — fixed fork-join thread pool (WS-D).
//
// Design contract (relied on by every CPU operator):
//   * Threads are created once, in the constructor (std::jthread); no per-call creation.
//   * parallel_for(n, fn) splits [0, n) into min(size(), n) contiguous ranges with a
//     *static* formula (partition_range) and runs range 0 on the calling thread.
//   * Operators only ever parallelize over independent output elements, and each element
//     is computed by the same sequential code whichever range it lands in. Therefore the
//     results of every operator are bit-identical for any thread count (tested).
//   * Nested parallel_for (from inside a running range, on any pool) runs inline on the
//     current thread: no deadlock, same results.
//   * Concurrent parallel_for calls from different external threads are serialized.
//   * An exception thrown by fn in any range is captured; after all ranges finish, the
//     exception from the lowest-indexed failing range is rethrown to the caller.
//
// Thread safety: all public member functions may be called from any thread.
// Lock order: dispatch_mu_ then mu_ (workers take only mu_). The dispatching caller holds
// dispatch_mu_ while its ranges run, so a range body must not block waiting for *another
// thread* that is itself calling parallel_for on the same pool (that would deadlock);
// same-thread nesting is safe (runs inline). Worker results are published to the caller
// through the mu_-guarded completion count (happens-before). Verified under TSan.

#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace halo::cpu {

class ThreadPool {
public:
    /// Creates a pool with `n_threads` participants in total: the calling thread plus
    /// `n_threads - 1` workers. Throws halo::Error(Config) when n_threads == 0.
    explicit ThreadPool(std::size_t n_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /// Number of participants (caller included).
    [[nodiscard]] std::size_t size() const noexcept { return n_threads_; }

    /// std::thread::hardware_concurrency(), or 1 when unknown.
    [[nodiscard]] static std::size_t default_threads() noexcept;

    /// Half-open range [first, second) of part `i` when [0, n) is split into `parts`
    /// contiguous parts. The first n % parts parts get one extra element.
    [[nodiscard]] static constexpr std::pair<std::size_t, std::size_t> partition_range(
        std::size_t n, std::size_t parts, std::size_t i) noexcept {
        const std::size_t base = n / parts;
        const std::size_t extra = n % parts;
        const std::size_t begin = i * base + (i < extra ? i : extra);
        return {begin, begin + base + (i < extra ? 1 : 0)};
    }

    /// Runs fn(begin, end) over a static partition of [0, n). Blocks until all ranges
    /// finish. n == 0 is a no-op.
    template <class Fn>
        requires std::invocable<Fn&, std::size_t, std::size_t>
    void parallel_for(std::size_t n, Fn&& fn) {
        using F = std::remove_reference_t<Fn>;
        const Thunk thunk = [](void* ctx, std::size_t b, std::size_t e) {
            (*static_cast<F*>(ctx))(b, e);
        };
        run(n, thunk, const_cast<void*>(static_cast<const void*>(std::addressof(fn))));
    }

private:
    using Thunk = void (*)(void*, std::size_t, std::size_t);
    struct Job {
        Thunk thunk = nullptr;
        void* ctx = nullptr;
        std::size_t n = 0;
        std::size_t parts = 0;
    };

    void run(std::size_t n, Thunk thunk, void* ctx);
    void execute_part(const Job& job, std::size_t part) noexcept;
    void worker_loop(const std::stop_token& stop, std::size_t index);

    std::size_t n_threads_;
    std::mutex dispatch_mu_;  // serializes external callers of run()
    std::mutex mu_;           // guards job_, generation_, pending_
    std::condition_variable_any work_cv_;
    std::condition_variable done_cv_;
    Job job_;
    std::uint64_t generation_ = 0;
    std::size_t pending_ = 0;
    std::vector<std::exception_ptr> errors_;  // one slot per part; written by that part only
    std::vector<std::jthread> workers_;       // declared last: stopped + joined first
};

/// Runs fn(begin, end) over [0, n) on `pool`, or inline as one range when pool is null.
template <class Fn>
    requires std::invocable<Fn&, std::size_t, std::size_t>
void parallel_for(ThreadPool* pool, std::size_t n, Fn&& fn) {
    if (n == 0) return;
    if (pool == nullptr || pool->size() == 1 || n == 1) {
        fn(std::size_t{0}, n);
        return;
    }
    pool->parallel_for(n, std::forward<Fn>(fn));
}

}  // namespace halo::cpu
