/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * P3.1 gate tests: lock-free per-device registry + memory::gpu::shutdown().
 *
 * Behavioral contracts verified here:
 *  - caching_allocator_for_device() returns the same address on every call for
 *    a given device index (singleton identity, §6.3).
 *  - Out-of-range device indices throw std::out_of_range.
 *  - shutdown() is safe to call before any device is initialized (no-op).
 *  - shutdown() is safe to call after initialization (flushes cache, allocator
 *    remains valid and returns the same address afterwards).
 *  - kMaxDevices == 16 (documented limit, §6.3).
 *
 * These tests do not require GPU hardware: the allocator constructors only
 * initialize metadata; driver calls happen at allocate() time.
 */

#include <gtest/gtest.h>

#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL

#include <atomic>
#include <stdexcept>

#include "gpu/caching_allocator.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/cuda_caching_allocator.h"
#include "gpu/gpu_runtime.h"
#endif

using namespace memory::gpu;

// ---------------------------------------------------------------------------
// kMaxDevices
// ---------------------------------------------------------------------------

TEST(Phase3Registry, MaxDevicesIs16)
{
    EXPECT_EQ(kMaxDevices, 16);
}

// ---------------------------------------------------------------------------
// Out-of-range rejection (no hardware needed)
// ---------------------------------------------------------------------------

TEST(Phase3Registry, NegativeDeviceIndexThrows)
{
    EXPECT_THROW(caching_allocator_for_device(-1), std::out_of_range);
}

TEST(Phase3Registry, DeviceIndexAtLimitThrows)
{
    EXPECT_THROW(caching_allocator_for_device(kMaxDevices), std::out_of_range);
}

TEST(Phase3Registry, DeviceIndexAboveLimitThrows)
{
    EXPECT_THROW(caching_allocator_for_device(kMaxDevices + 5), std::out_of_range);
}

// ---------------------------------------------------------------------------
// Singleton identity (warm-path: 0 registry locks after first call)
// ---------------------------------------------------------------------------

TEST(Phase3Registry, SameAddressOnRepeatedCalls)
{
    // Both calls must return the same object (P3.1 guarantee: one allocator per
    // device index, no mutex on the warm path).
    caching_allocator& a1 = caching_allocator_for_device(0);
    caching_allocator& a2 = caching_allocator_for_device(0);
    EXPECT_EQ(&a1, &a2);
}

TEST(Phase3Registry, ManyCallsReturnSameAddress)
{
    caching_allocator* first = &caching_allocator_for_device(0);
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_EQ(&caching_allocator_for_device(0), first);
    }
}

// ---------------------------------------------------------------------------
// shutdown() safety
// ---------------------------------------------------------------------------

TEST(Phase3Registry, ShutdownAfterInitDoesNotCrash)
{
    // Ensure device 0 is initialized before shutting down.
    (void)caching_allocator_for_device(0);
    EXPECT_NO_THROW(memory::gpu::shutdown());
}

TEST(Phase3Registry, AllocatorValidAfterShutdown)
{
    // shutdown() calls empty_cache() but does not destroy the allocator.
    // caching_allocator_for_device must still return the same singleton.
    caching_allocator* before = &caching_allocator_for_device(0);
    memory::gpu::shutdown();
    caching_allocator* after = &caching_allocator_for_device(0);
    EXPECT_EQ(before, after);
}

TEST(Phase3Registry, RepeatedShutdownsAreIdempotent)
{
    (void)caching_allocator_for_device(0);
    EXPECT_NO_THROW(memory::gpu::shutdown());
    EXPECT_NO_THROW(memory::gpu::shutdown());
    EXPECT_NO_THROW(memory::gpu::shutdown());
}

// ---------------------------------------------------------------------------
// P3.5: O(1) lock-free basic stats
// ---------------------------------------------------------------------------

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
namespace
{
bool gpu_available_p3()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
}  // namespace
#endif

TEST(Phase3Stats, ZeroBeforeAnyAllocation)
{
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
    if (!gpu_available_p3()) GTEST_SKIP() << "No GPU device";
    // A freshly-constructed local allocator must expose zero for all four
    // counters before any allocation (singleton may be non-zero from other tests).
    cuda_caching_allocator a(0);
    EXPECT_EQ(a.bytes_allocated_now(),      0U);
    EXPECT_EQ(a.peak_bytes_allocated_now(), 0U);
    EXPECT_EQ(a.bytes_reserved_now(),       0U);
    EXPECT_EQ(a.peak_bytes_reserved_now(),  0U);
#else
    GTEST_SKIP() << "CUDA/HIP not available";
#endif
}

TEST(Phase3Stats, LockFreeStatsConsistentWithStats)
{
    caching_allocator& a  = caching_allocator_for_device(0);
    const auto         cs = a.stats();
    // Lock-free reads must agree with the full stats() snapshot at idle.
    EXPECT_EQ(a.bytes_allocated_now(),
              cs.bytes_allocated.load(std::memory_order_relaxed));
    EXPECT_EQ(a.bytes_reserved_now(),
              cs.bytes_reserved.load(std::memory_order_relaxed));
}

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

// ---------------------------------------------------------------------------
// P3.2: inline_stream_set + block_freelist (hardware required)
// ---------------------------------------------------------------------------

namespace
{
bool gpu_available()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
}  // namespace

TEST(Phase3Freelist, AllocFreeRoundTripManyTimes)
{
    if (!gpu_available()) GTEST_SKIP() << "No GPU device";
    // 200 small alloc/free cycles exercise the block split path and the
    // block_freelist acquire/release loop (warm path must not crash or leak).
    cuda_caching_allocator a(0);
    constexpr size_t       kSmall = 512;
    for (int i = 0; i < 200; ++i)
    {
        void* p = a.allocate(kSmall);
        ASSERT_NE(p, nullptr);
        a.deallocate(p, kSmall);
    }
    // All memory must be back in the cache (none still allocated).
    EXPECT_EQ(a.bytes_allocated_now(), 0U);
}

TEST(Phase3Freelist, SplitBlockReuseDoesNotCrash)
{
    if (!gpu_available()) GTEST_SKIP() << "No GPU device";
    // Allocate two differently-sized blocks from the same segment so the
    // allocator must split; free them and reallocate to verify the freelist
    // recycles correctly.
    cuda_caching_allocator a(0);
    void* p1 = a.allocate(512);
    void* p2 = a.allocate(1024);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    a.deallocate(p1, 512);
    a.deallocate(p2, 1024);
    // Reallocate same sizes: should hit cache, not driver.
    const size_t misses_before = a.stats().cache_misses.load(std::memory_order_relaxed);
    void*        p3            = a.allocate(512);
    void*        p4            = a.allocate(1024);
    EXPECT_EQ(a.stats().cache_misses.load(std::memory_order_relaxed), misses_before);
    a.deallocate(p3, 512);
    a.deallocate(p4, 1024);
}

TEST(Phase3Freelist, InlineStreamSetUpToFourStreams)
{
    if (!gpu_available()) GTEST_SKIP() << "No GPU device";
    // Record ≤4 cross-stream uses (inline path, no overflow heap alloc).
    cuda_caching_allocator a(0);
    void*        p = a.allocate(512);
    cudaStream_t s[4]{};
    for (auto& st : s)
        (void)cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    for (auto st : s)
        a.record_stream(p, st);
    a.deallocate(p, 512);
    // Sync all streams so the block's events complete and the cache is reclaimable.
    for (auto st : s)
        (void)cudaStreamSynchronize(st);
    a.empty_cache();
    EXPECT_EQ(a.bytes_reserved_now(), 0U);
    for (auto st : s)
        (void)cudaStreamDestroy(st);
}

TEST(Phase3Freelist, InlineStreamSetOverflowFiveStreams)
{
    if (!gpu_available()) GTEST_SKIP() << "No GPU device";
    // Record 5 cross-stream uses: the 5th spills to the overflow std::set.
    cuda_caching_allocator a(0);
    void*        p = a.allocate(512);
    cudaStream_t s[5]{};
    for (auto& st : s)
        (void)cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    for (auto st : s)
        a.record_stream(p, st);
    a.deallocate(p, 512);
    for (auto st : s)
        (void)cudaStreamSynchronize(st);
    a.empty_cache();
    EXPECT_EQ(a.bytes_reserved_now(), 0U);
    for (auto st : s)
        (void)cudaStreamDestroy(st);
}

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
