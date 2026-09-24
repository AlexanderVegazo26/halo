// ThreadPool contract tests (static partition, coverage, exceptions, nesting, lifetime).

#include <gtest/gtest.h>

#include <atomic>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "halo/backends/cpu/thread_pool.h"
#include "halo/core/error.h"

using halo::cpu::ThreadPool;

TEST(ThreadPool, RejectsZeroThreads) { EXPECT_THROW(ThreadPool(0), halo::Error); }

TEST(ThreadPool, PartitionIsContiguousCompleteAndBalanced) {
    for (std::size_t n : {0u, 1u, 5u, 16u, 17u, 1000u}) {
        for (std::size_t parts : {1u, 2u, 3u, 7u, 16u}) {
            std::size_t expect = 0;
            for (std::size_t i = 0; i < parts; ++i) {
                const auto [b, e] = ThreadPool::partition_range(n, parts, i);
                EXPECT_EQ(b, expect) << "n=" << n << " parts=" << parts << " i=" << i;
                EXPECT_LE(e - b, n / parts + 1);
                EXPECT_GE(e - b, n / parts);
                expect = e;
            }
            EXPECT_EQ(expect, n);
        }
    }
}

TEST(ThreadPool, EveryIndexVisitedExactlyOnce) {
    for (std::size_t threads : {1u, 2u, 3u, 7u, 16u, 40u}) {
        ThreadPool pool(threads);
        for (std::size_t n : {0u, 1u, 2u, 7u, 39u, 1000u}) {
            std::vector<std::atomic<int>> hits(n);
            pool.parallel_for(n, [&](std::size_t b, std::size_t e) {
                for (std::size_t i = b; i < e; ++i) hits[i].fetch_add(1);
            });
            for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(hits[i].load(), 1) << "threads=" << threads << " n=" << n;
        }
    }
}

TEST(ThreadPool, UsesMultipleThreadsAndStaticRanges) {
    ThreadPool pool(4);
    std::vector<std::thread::id> owner(4);
    std::vector<std::pair<std::size_t, std::size_t>> ranges(4);
    pool.parallel_for(4, [&](std::size_t b, std::size_t e) {
        ASSERT_EQ(e, b + 1);
        owner[b] = std::this_thread::get_id();
        ranges[b] = {b, e};
    });
    EXPECT_EQ(owner[0], std::this_thread::get_id());  // range 0 runs on the caller
    for (std::size_t i = 1; i < 4; ++i) EXPECT_NE(owner[i], owner[0]);
}

TEST(ThreadPool, ExceptionFromLowestFailingRangeIsRethrownAfterAllFinish) {
    ThreadPool pool(4);
    std::atomic<int> finished{0};
    try {
        pool.parallel_for(4, [&](std::size_t b, std::size_t) {
            if (b == 1) throw std::runtime_error("range1");
            if (b == 3) throw std::runtime_error("range3");
            finished.fetch_add(1);
        });
        FAIL() << "expected throw";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "range1");
    }
    EXPECT_EQ(finished.load(), 2);
    // Pool is still usable afterwards.
    std::atomic<int> sum{0};
    pool.parallel_for(10, [&](std::size_t b, std::size_t e) { sum.fetch_add(static_cast<int>(e - b)); });
    EXPECT_EQ(sum.load(), 10);
}

TEST(ThreadPool, NestedCallsRunInlineWithoutDeadlock) {
    ThreadPool pool(4);
    std::vector<std::atomic<int>> hits(8 * 8);
    pool.parallel_for(8, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) {
            pool.parallel_for(8, [&](std::size_t b2, std::size_t e2) {
                for (std::size_t j = b2; j < e2; ++j) hits[i * 8 + j].fetch_add(1);
            });
        }
    });
    for (auto& h : hits) EXPECT_EQ(h.load(), 1);
}

TEST(ThreadPool, ConcurrentExternalCallersAreSerializedCorrectly) {
    ThreadPool pool(3);
    std::atomic<long> total{0};
    {
        std::vector<std::jthread> callers;
        for (int c = 0; c < 4; ++c) {
            callers.emplace_back([&] {
                for (int r = 0; r < 50; ++r) {
                    pool.parallel_for(100, [&](std::size_t b, std::size_t e) {
                        long s = 0;
                        for (std::size_t i = b; i < e; ++i) s += static_cast<long>(i);
                        total.fetch_add(s);
                    });
                }
            });
        }
    }
    EXPECT_EQ(total.load(), 4L * 50L * (99L * 100L / 2L));
}

TEST(ThreadPool, ConstructDestroyRepeatedly) {
    for (int i = 0; i < 20; ++i) {
        ThreadPool pool(static_cast<std::size_t>(1 + i % 5));
        if (i % 2 == 0) {
            std::atomic<int> n{0};
            pool.parallel_for(5, [&](std::size_t b, std::size_t e) { n.fetch_add(static_cast<int>(e - b)); });
            EXPECT_EQ(n.load(), 5);
        }
    }
}

TEST(ThreadPool, NullPoolHelperRunsInline) {
    int calls = 0;
    halo::cpu::parallel_for(nullptr, 10, [&](std::size_t b, std::size_t e) {
        EXPECT_EQ(b, 0u);
        EXPECT_EQ(e, 10u);
        ++calls;
    });
    EXPECT_EQ(calls, 1);
    halo::cpu::parallel_for(nullptr, 0, [&](std::size_t, std::size_t) { ++calls; });
    EXPECT_EQ(calls, 1);
}
