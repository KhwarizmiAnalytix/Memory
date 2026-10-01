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

// Fault-injection regression coverage for gpu/cuda_caching_allocator.cpp,
// compiled against Testing/CudaCachingAllocator/fake_runtime.h instead of a
// real CUDA/HIP driver. See Docs/memory_runtime_implementation_plan.md Appendix A and
// this suite's CMakeLists.txt for why this builds under the HIP labels.

#include <gtest/gtest.h>

#include <stdexcept>
#include <thread>

#include "fake_runtime.h"
#include "gpu/cuda_caching_allocator.h"

using memory::gpu::cuda_caching_allocator;
namespace rt = fake_runtime;

namespace
{
constexpr size_t kSegmentSize = 20 * 1024 * 1024;  // exact kLargeBuffer multiple
}

class CudaCachingAllocatorRuntime : public ::testing::Test
{
    void SetUp() override { rt::reset(); }
    void TearDown() override
    {
        EXPECT_EQ(rt::event_creates, rt::event_destroys);
        rt::reset();
    }
};

// Regression for the P0 fix landed 2026-09-28 (commit e34ea89): a partial
// event-insertion failure (first stream's cudaEventRecord succeeds, second
// stream's fails) must quarantine the block, never free it back to the pool
// -- the first stream's real in-flight work was never proven complete.
TEST_F(CudaCachingAllocatorRuntime, PartialEventFailureOnSecondStreamQuarantinesBlock)
{
    cuda_caching_allocator allocator(0);
    void*                  ptr = allocator.allocate(4096);
    ASSERT_NE(nullptr, ptr);
    allocator.record_stream(ptr, rt::stream(1));
    allocator.record_stream(ptr, rt::stream(2));

    rt::fail_event_record_at_call = 2;  // stream(1) succeeds, stream(2) fails
    EXPECT_THROW(allocator.deallocate(ptr, 4096), std::runtime_error);

    // The quarantined block must never be handed back out: a same-size
    // allocate() must return a different pointer.
    void* other = allocator.allocate(4096);
    ASSERT_NE(nullptr, other);
    EXPECT_NE(ptr, other);
}

// Regression for the same fix's other branch: a single stream whose only
// event fails (event_count==0, streams non-empty in the catch block) must
// also quarantine, not take the "no uses were ever registered" free path.
TEST_F(CudaCachingAllocatorRuntime, SingleStreamEventFailureQuarantinesNotFrees)
{
    cuda_caching_allocator allocator(0);
    void*                  ptr = allocator.allocate(4096);
    ASSERT_NE(nullptr, ptr);
    allocator.record_stream(ptr, rt::stream(1));

    rt::fail_event_record_at_call = 1;  // the only cudaEventRecord call fails
    EXPECT_THROW(allocator.deallocate(ptr, 4096), std::runtime_error);

    void* other = allocator.allocate(4096);
    ASSERT_NE(nullptr, other);
    EXPECT_NE(ptr, other);
}

// Order 2, item 1: the OOM-chain retry (after release_cached_blocks_locked())
// used to call alloc_segment_unlocked() again without rechecking the budget.
// alloc_segment_unlocked drops the lock around its own cudaMalloc call, so a
// concurrent set_memory_fraction() landing during that window can shrink
// headroom between this function's earlier checks and the retry. Pause the
// first cudaMalloc call mid-flight, shrink the budget from another thread,
// then release: the fixed code must recheck and reject rather than
// committing a second, now-over-budget driver allocation.
TEST_F(CudaCachingAllocatorRuntime, RetryAfterDriverFailureRechecksShrunkenBudget)
{
    rt::device_total_bytes = 100 * kSegmentSize;
    cuda_caching_allocator allocator(0);
    allocator.set_memory_fraction(2.0 * static_cast<double>(kSegmentSize) / rt::device_total_bytes);

    rt::malloc_pause_on_call = 1;  // pause the 1st cudaMalloc call
    rt::fail_malloc_calls    = 1;  // ...which then fails once released

    std::thread worker(
        [&] { EXPECT_THROW(allocator.allocate(kSegmentSize), std::bad_alloc); });
    rt::wait_for_malloc_pause();

    // Shrink the budget while the worker is paused mid-driver-call for its
    // first attempt -- the only real concurrent-mutation window given the
    // allocator's lock discipline (release_cached_blocks_locked holds the
    // lock throughout; only alloc_segment_unlocked's own cudaMalloc drops it).
    allocator.set_memory_fraction(
        0.5 * static_cast<double>(kSegmentSize) / rt::device_total_bytes);
    rt::release_malloc_pause();
    worker.join();

    EXPECT_EQ(1, rt::malloc_calls) << "the recheck must reject before spending a 2nd cudaMalloc "
                                       "once it sees the shrunken budget";
}

// Order 2, item 2: if malloc_segment's device_guard throws (its cudaGetDevice
// call fails) while the allocator lock is dropped for the driver call, the
// pending-budget reservation for that request must still be rolled back.
// Leaking it would make every subsequent allocate() see stale phantom
// headroom consumption and wrongly reject requests that should fit.
TEST_F(CudaCachingAllocatorRuntime, SegmentAllocDeviceGuardThrowRollsBackPendingBudget)
{
    rt::device_total_bytes = 100 * kSegmentSize;
    cuda_caching_allocator allocator(0);
    // Room for exactly one segment: set_memory_fraction's own device_guard
    // (inside query_device_total_memory) consumes one cudaGetDevice call here,
    // before the injected failure below is armed.
    allocator.set_memory_fraction(1.0 * static_cast<double>(kSegmentSize) / rt::device_total_bytes);

    rt::fail_get_device_calls = 1;  // fails the next cudaGetDevice call only:
                                     // malloc_segment's device_guard, inside
                                     // the retry-free first allocate() attempt.
    EXPECT_THROW(allocator.allocate(kSegmentSize), std::runtime_error);

    // If pending_reserved_bytes_ leaked from the failed attempt above, this
    // retry's budget check sees (stale pending + fresh alloc_size) > budget
    // and wrongly throws bad_alloc; the fix must have rolled it back to 0.
    void* ptr = nullptr;
    EXPECT_NO_THROW(ptr = allocator.allocate(kSegmentSize));
    ASSERT_NE(nullptr, ptr);
    allocator.deallocate(ptr, kSegmentSize);
}
