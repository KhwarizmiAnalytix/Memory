/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#include <vector>

#include "common/pinned_buffer.h"
#include "fake_runtime.h"
#include "helper/pinned_memory_allocator.h"

using memory::cpu::pinned_memory_allocator;
namespace rt = fake_runtime;

class PinnedRuntime : public ::testing::Test
{
    void SetUp() override { rt::reset(); }
    void TearDown() override
    {
        EXPECT_EQ(rt::event_creates, rt::event_destroys);
        rt::reset();
    }
};

TEST_F(PinnedRuntime, ReusesAlignedSizeClassesAndAccountsBacking)
{
    pinned_memory_allocator pool;
    auto*                   first = pool.allocate(1000);
    EXPECT_EQ(0U, reinterpret_cast<std::uintptr_t>(first) % 64);
    EXPECT_EQ(1000U, pool.stats().bytes_requested);
    EXPECT_EQ(1024U, pool.stats().bytes_allocated);
    EXPECT_EQ(1087U, pool.stats().bytes_reserved);
    ASSERT_TRUE(pool.deallocate(first));
    EXPECT_FALSE(pool.deallocate(first));
    auto* second = pool.allocate(900);
    EXPECT_EQ(first, second);
    EXPECT_EQ(1U, pool.stats().cache_hits);
    EXPECT_EQ(1, rt::host_allocations);
    EXPECT_TRUE(pool.deallocate(second));
    pool.empty_cache();
    EXPECT_EQ(0U, pool.stats().bytes_reserved);
}

TEST_F(PinnedRuntime, DefaultStreamDefersReuseWithoutBlocking)
{
    pinned_memory_allocator pool;
    rt::ready[0] = false;
    auto* first  = pool.allocate(100);
    pool.record_stream(first, nullptr);
    EXPECT_TRUE(pool.deallocate(first));
    EXPECT_EQ(512U, pool.stats().bytes_pending);
    auto* second = pool.allocate(100);
    EXPECT_NE(first, second);
    EXPECT_EQ(0, rt::synchronizations);
    rt::ready[0] = true;
    pool.poll();
    EXPECT_EQ(0U, pool.stats().bytes_pending);
    auto* third = pool.allocate(100);
    EXPECT_EQ(first, third);
    pool.deallocate(second);
    pool.deallocate(third);
}

TEST_F(PinnedRuntime, InteriorPointersWaitForAllStreamsAndRecycleEvents)
{
    pinned_memory_allocator pool;
    auto*                   ptr = static_cast<char*>(pool.allocate(100));
    rt::ready[1] = rt::ready[2] = false;
    pool.record_stream(ptr + 20, rt::stream(1));
    pool.record_stream(ptr, rt::stream(2));
    pool.record_stream(ptr, rt::stream(2));
    EXPECT_EQ(2, rt::event_creates);
    pool.deallocate(ptr);
    rt::ready[1] = true;
    pool.poll();
    EXPECT_EQ(512U, pool.stats().bytes_pending);
    rt::ready[2] = true;
    pool.poll();
    auto* reused = pool.allocate(100);
    EXPECT_EQ(ptr, reused);
    pool.record_stream(reused, rt::stream(3));
    EXPECT_EQ(2, rt::event_creates);
    pool.deallocate(reused);
}

TEST_F(PinnedRuntime, ZeroCacheLimitStillProtectsPendingTransfers)
{
    pinned_memory_allocator pool(0, 0);
    auto*                   ptr = pool.allocate(100);
    rt::ready[1]                = false;
    pool.record_stream(ptr, rt::stream(1));
    pool.deallocate(ptr);
    EXPECT_EQ(0, rt::host_frees);
    pool.empty_cache();
    EXPECT_EQ(1, rt::synchronizations);
    EXPECT_EQ(1, rt::host_frees);
    EXPECT_EQ(0U, pool.stats().bytes_reserved);
}

TEST_F(PinnedRuntime, EmptyCachePreservesLiveAllocations)
{
    pinned_memory_allocator pool;
    auto*                   live  = pool.allocate(100);
    auto*                   freed = pool.allocate(1000);
    pool.deallocate(freed);
    pool.empty_cache();
    EXPECT_EQ(512U, pool.stats().bytes_allocated);
    EXPECT_EQ(575U, pool.stats().bytes_reserved);
    EXPECT_TRUE(pool.deallocate(live));
}

TEST_F(PinnedRuntime, FailedEventRecordingQuarantinesBacking)
{
    pinned_memory_allocator pool;
    auto*                   ptr = pool.allocate(100);
    pool.record_stream(ptr, rt::stream(1));
    rt::fail_record = true;
    EXPECT_FALSE(pool.deallocate(ptr));
    rt::fail_record = false;
    pool.empty_cache();
    EXPECT_EQ(512U, pool.stats().bytes_pending);
    EXPECT_EQ(1U, pool.stats().num_errors);
    EXPECT_EQ(0, rt::host_frees);
    auto* other = pool.allocate(100);
    EXPECT_NE(ptr, other);
    pool.deallocate(other);
}

TEST_F(PinnedRuntime, FailedEventQueryQuarantinesBacking)
{
    pinned_memory_allocator pool;
    auto*                   ptr = pool.allocate(100);
    pool.record_stream(ptr, rt::stream(1));
    rt::fail_query = true;
    EXPECT_FALSE(pool.deallocate(ptr));
    rt::fail_query = false;
    pool.empty_cache();
    EXPECT_EQ(0, rt::host_frees);
    EXPECT_EQ(512U, pool.stats().bytes_pending);
}

// Regression: deallocate(ptr)'s return value used to be `success &&
// stats_.num_errors == errors_before`, comparing the allocator's *global*
// error counter before/after -- not just this block's own outcome. Since
// deallocate() calls process_pending(false), which sweeps every block still
// waiting in the pending list (not only the one just deallocated), a
// completely unrelated older pending block failing its event query during
// that sweep would make this call report failure for a pointer that itself
// deallocated cleanly. `a` (below) is left deliberately not-yet-complete so
// it is still in the pending list when `b` is deallocated; `b` itself never
// records any stream, so its own path can never fail.
TEST_F(PinnedRuntime, DeallocateReturnValueIsNotPolluted)
{
    pinned_memory_allocator pool;

    auto* a = pool.allocate(100);
    pool.record_stream(a, rt::stream(1));
    rt::ready[1] = false;
    EXPECT_TRUE(pool.deallocate(a));  // pending, not failed -- just not complete yet

    auto* b = pool.allocate(100);  // a is still pending (not cached), so this is a fresh block
    EXPECT_NE(a, b);

    rt::fail_query = true;
    // b's own deallocate() has nothing to fail on, but process_pending(false)
    // inside it also revisits `a`, whose query now fails.
    EXPECT_TRUE(pool.deallocate(b));
    rt::fail_query = false;

    EXPECT_EQ(1U, pool.stats().num_errors);  // a's failure was still recorded
    pool.empty_cache();
}

TEST_F(PinnedRuntime, FailedDriverFreeIsNotRetried)
{
    pinned_memory_allocator pool(0, 0);
    auto*                   ptr = pool.allocate(100);
    rt::fail_host_free          = true;
    EXPECT_FALSE(pool.deallocate(ptr));
    rt::fail_host_free = false;
    pool.empty_cache();
    EXPECT_EQ(0, rt::host_frees);
    EXPECT_EQ(512U, pool.stats().bytes_pending);
}

TEST_F(PinnedRuntime, CopyValidatesBoundsAndRegistersBeforeSubmitting)
{
    pinned_memory_allocator pool;
    auto*                   host = static_cast<char*>(pool.allocate(100));
    char                    device[100]{};
    host[10] = 42;
    pool.copy_to_device_async(device, host + 10, 90, rt::stream(1));
    EXPECT_EQ(42, device[0]);
    pool.copy_from_device_async(host, device, 90, rt::stream(2));
    EXPECT_EQ(42, host[0]);
    EXPECT_THROW(pool.copy_to_device_async(device, host + 10, 91), std::invalid_argument);
    EXPECT_THROW(pool.copy_from_device_async(nullptr, device, 1), std::invalid_argument);
    EXPECT_THROW(pool.record_stream(host + 100, nullptr), std::invalid_argument);
    rt::fail_event_create = true;
    EXPECT_THROW(pool.copy_to_device_async(device, host, 100, rt::stream(3)), std::runtime_error);
    EXPECT_EQ(2, rt::copies);
    pool.deallocate(host);
}

TEST_F(PinnedRuntime, OomFlushesCacheRetriesAndReportsFailure)
{
    pinned_memory_allocator pool;
    auto*                   cached = pool.allocate(100);
    pool.deallocate(cached);
    rt::fail_allocations = 1;
    auto* larger         = pool.allocate(1000);
    EXPECT_EQ(1, rt::host_frees);
    pool.deallocate(larger);
    rt::fail_allocations = 2;
    EXPECT_THROW(pool.allocate(2000), std::bad_alloc);
    EXPECT_EQ(1U, pool.stats().num_ooms);
    EXPECT_EQ(0U, pool.stats().bytes_reserved);
}

TEST_F(PinnedRuntime, OverflowForeignPointersAndDeviceRestoration)
{
    pinned_memory_allocator pool(1);
    EXPECT_THROW(pool.allocate(std::numeric_limits<std::size_t>::max()), std::length_error);
    int foreign;
    EXPECT_FALSE(pool.deallocate(&foreign));
    EXPECT_THROW(pool.record_stream(&foreign, nullptr), std::invalid_argument);
    auto* ptr = pool.allocate(100);
    EXPECT_EQ(0, rt::current_device);
    EXPECT_TRUE(pool.deallocate(ptr));
    EXPECT_EQ(0, rt::current_device);
}

TEST_F(PinnedRuntime, DestructionWaitsForLiveAndPendingRecordedUses)
{
    {
        pinned_memory_allocator pool;
        auto*                   live    = pool.allocate(100);
        auto*                   pending = pool.allocate(100);
        rt::ready[1] = rt::ready[2] = false;
        pool.record_stream(live, rt::stream(1));
        pool.record_stream(pending, rt::stream(2));
        pool.deallocate(pending);
    }
    EXPECT_EQ(2, rt::host_frees);
    EXPECT_EQ(2, rt::synchronizations);
}

TEST_F(PinnedRuntime, TypedBufferMovesAndDefersDestruction)
{
    auto& pool = memory::cpu::pinned_allocator_for_device();
    {
        memory::pinned_buffer<int> first(16);
        int                        device[16]{};
        first.data()[0] = 123;
        rt::ready[0]    = false;
        first.copy_to_device_async(device);
        EXPECT_EQ(123, device[0]);
        memory::pinned_buffer<int> moved(std::move(first));
        EXPECT_EQ(nullptr, first.data());
        EXPECT_EQ(16U, moved.size());
        memory::pinned_buffer<int> assigned(8);
        assigned = std::move(moved);
        EXPECT_EQ(nullptr, moved.data());
        EXPECT_EQ(123, assigned.data()[0]);
    }
    EXPECT_EQ(512U, pool.stats().bytes_pending);
    pool.empty_cache();
    EXPECT_EQ(0U, pool.stats().bytes_reserved);
}

TEST_F(PinnedRuntime, ConcurrentAllocationAndCrossThreadFree)
{
    pinned_memory_allocator  pool;
    std::vector<void*>       pointers(16);
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < pointers.size(); ++i)
        threads.emplace_back([&, i] { pointers[i] = pool.allocate(100 + i); });
    for (auto& thread : threads)
        thread.join();
    EXPECT_EQ(16U * 512, pool.stats().bytes_allocated);
    for (auto* ptr : pointers)
        EXPECT_TRUE(pool.deallocate(ptr));
    EXPECT_EQ(0U, pool.stats().bytes_allocated);
    pool.empty_cache();
    EXPECT_EQ(0U, pool.stats().bytes_reserved);
}
