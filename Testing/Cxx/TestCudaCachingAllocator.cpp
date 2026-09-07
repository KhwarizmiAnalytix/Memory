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

#include "MemoryTest.h"
#include "common/memory_macros.h"
#include "logger/logger.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "allocator.h"
#include "common/data_ptr.h"
#include "gpu/cuda_caching_allocator.h"
#include "gpu/gpu_runtime.h"

#if MEMORY_HAS_PROFILER
#include "gpu/caching_allocator_profiler_report.h"
#endif

using namespace memory;
using namespace memory::gpu;

namespace
{
bool cuda_device_available()
{
    int               device_count = 0;
    const cudaError_t err          = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

// cudaStreamAddCallback/cudaLaunchHostFunc target: blocks the stream it is
// enqueued on until `*static_cast<std::atomic<bool>*>(user_data)` is set,
// so a test can deterministically keep a stream "busy" without a real
// kernel (this .cpp is compiled by the host compiler, not nvcc).
void CUDART_CB block_stream_until_released(void* user_data)
{
    auto* release_flag = static_cast<std::atomic<bool>*>(user_data);
    while (!release_flag->load(std::memory_order_acquire))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
}  // namespace

class CudaCachingAllocator : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!cuda_device_available())
        {
            GTEST_SKIP() << "No GPU device available";
        }
    }
};

class CudaCachingAllocatorTemplate : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!cuda_device_available())
        {
            GTEST_SKIP() << "No GPU device available";
        }
    }
};

/**
 * @brief Test basic CUDA caching allocator construction and destruction
 */
MEMORYTEST_F(CudaCachingAllocator, constructs_with_valid_parameters)
{
    // Test basic construction
    cuda_caching_allocator allocator(0, 64 * 1024ULL);  // 64MB cache

    // Verify device index
    EXPECT_EQ(0, allocator.device());

    // Verify cache size
    EXPECT_EQ(64 * 1024ULL, allocator.max_cached_bytes());

    LOGGING_LOG_INFO("CUDA caching allocator construction test passed");
}

/**
 * @brief Test CUDA caching allocator constructor variations
 */
MEMORYTEST_F(CudaCachingAllocator, constructor_variations)
{
    // Test default constructor (if available)
    try
    {
        cuda_caching_allocator allocator1(0);  // Default cache size
        EXPECT_EQ(0, allocator1.device());
        EXPECT_GT(allocator1.max_cached_bytes(), 0);
    }
    catch (...)
    {
        // Default constructor may not be available, that's okay
    }

    // Test constructor with different cache sizes
    std::vector<size_t> cache_sizes = {
        1024 * 1024,       // 1MB
        16 * 1024 * 1024,  // 16MB
        64 * 1024 * 1024,  // 64MB
        256 * 1024 * 1024  // 256MB
    };

    for (size_t cache_size : cache_sizes)
    {
        cuda_caching_allocator allocator(0, cache_size);
        EXPECT_EQ(0, allocator.device());
        EXPECT_EQ(cache_size, allocator.max_cached_bytes());
    }

    LOGGING_LOG_INFO("CUDA caching allocator constructor variations test passed");
}

/**
 * @brief Test basic allocation and deallocation functionality
 */
MEMORYTEST_F(CudaCachingAllocator, allocates_and_deallocates_memory)
{
    cuda_caching_allocator allocator(0, 32 * 1024ULL);  // 32MB cache

    // Test basic allocation
    void* ptr1 = allocator.allocate(1024);
    EXPECT_NE(nullptr, ptr1);

    // Test deallocation
    allocator.deallocate(ptr1, 1024);

    // Test multiple allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator.allocate(512 * (i + 1));
        EXPECT_NE(nullptr, ptr);
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (size_t i = 0; i < ptrs.size(); ++i)
    {
        allocator.deallocate(ptrs[i], 512 * (i + 1));
    }

    LOGGING_LOG_INFO("CUDA caching allocator allocation/deallocation test passed");
}

/**
 * @brief Test cache management functionality
 */
MEMORYTEST_F(CudaCachingAllocator, manages_cache_correctly)
{
    cuda_caching_allocator allocator(0, 16ULL * 1024 * 1024);  // 16 MiB cache

    // Allocate and deallocate to populate cache
    void* ptr1 = allocator.allocate(1024);
    EXPECT_NE(nullptr, ptr1);
    allocator.deallocate(ptr1, 1024);

    // Get initial stats
    auto stats_before = allocator.stats();

    // Clear cache
    allocator.empty_cache();

    // Verify cache was cleared
    auto stats_after = allocator.stats();
    EXPECT_GT(stats_before.bytes_cached.load(), stats_after.bytes_cached.load());
    EXPECT_EQ(0, stats_after.bytes_cached.load());

    LOGGING_LOG_INFO("CUDA caching allocator cache management test passed");
}

/**
 * @brief Test cache size limits and configuration
 */
MEMORYTEST_F(CudaCachingAllocator, respects_cache_size_limits)
{
    cuda_caching_allocator allocator(0, 8 * 1024ULL);  // 8MB cache

    // Test setting new cache size
    allocator.set_max_cached_bytes(16 * 1024ULL);
    EXPECT_EQ(16 * 1024ULL, allocator.max_cached_bytes());

    // Test disabling cache
    allocator.set_max_cached_bytes(0);
    EXPECT_EQ(0, allocator.max_cached_bytes());

    LOGGING_LOG_INFO("CUDA caching allocator cache size limits test passed");
}

/**
 * @brief Test statistics collection and reporting
 */
MEMORYTEST_F(CudaCachingAllocator, provides_accurate_statistics)
{
    cuda_caching_allocator allocator(0, 32 * 1024ULL);  // 32MB cache

    // Get initial stats
    auto initial_stats = allocator.stats();

    // Perform some allocations
    void* ptr1 = allocator.allocate(2048);
    void* ptr2 = allocator.allocate(4096);

    auto after_alloc_stats = allocator.stats();
    EXPECT_GT(after_alloc_stats.bytes_allocated.load(), initial_stats.bytes_allocated.load());

    // Deallocate
    allocator.deallocate(ptr1, 2048);
    allocator.deallocate(ptr2, 4096);

    auto after_dealloc_stats = allocator.stats();
    EXPECT_EQ(0, after_dealloc_stats.bytes_allocated.load());

    LOGGING_LOG_INFO("CUDA caching allocator statistics test passed");
}

/**
 * @brief Test move semantics and resource transfer
 */
MEMORYTEST_F(CudaCachingAllocator, supports_move_semantics)
{
    // Create allocator
    cuda_caching_allocator allocator1(0, 16 * 1024ULL);

    // Allocate some memory
    void* ptr = allocator1.allocate(1024);
    EXPECT_NE(nullptr, ptr);

    // Move construct
    cuda_caching_allocator allocator2 = std::move(allocator1);
    EXPECT_EQ(0, allocator2.device());

    // Original allocator should be in moved-from state
    // Moved-to allocator should work
    allocator2.deallocate(ptr, 1024);

    LOGGING_LOG_INFO("CUDA caching allocator move semantics test passed");
}

/**
 * @brief Test error handling for invalid operations
 */
MEMORYTEST_F(CudaCachingAllocator, handles_errors_gracefully)
{
    cuda_caching_allocator allocator(0, 16 * 1024ULL);

    // Test zero-size allocation
    MEMORY_UNUSED void* ptr_zero = allocator.allocate(0);
    // Should handle gracefully (implementation-defined behavior)

    // Test null pointer deallocation
    // Should not crash
    allocator.deallocate(nullptr, 1024);

    LOGGING_LOG_INFO("CUDA caching allocator error handling test passed");
}

/**
 * @brief Test template allocator construction and basic operations
 */
MEMORYTEST_F(CudaCachingAllocatorTemplate, constructs_with_different_types)
{
    // Test template allocator for different types
    cuda_caching_allocator_template<float, 256>  float_allocator(0, 32 * 1024ULL);
    cuda_caching_allocator_template<double, 256> double_allocator(0, 32 * 1024ULL);
    cuda_caching_allocator_template<int, 128>    int_allocator(0, 16 * 1024ULL);

    // Verify device indices
    EXPECT_EQ(0, float_allocator.device());
    EXPECT_EQ(0, double_allocator.device());
    EXPECT_EQ(0, int_allocator.device());

    LOGGING_LOG_INFO("CUDA caching allocator template construction test passed");
}

/**
 * @brief Test template allocator type-safe allocation
 */
MEMORYTEST_F(CudaCachingAllocatorTemplate, allocates_typed_memory_safely)
{
    cuda_caching_allocator_template<float, 256> allocator(0, 16 * 1024ULL);

    // Test typed allocation
    float* ptr1 = allocator.allocate(100);
    EXPECT_NE(nullptr, ptr1);

    // Test deallocation
    allocator.deallocate(ptr1, 100);

    // Test larger allocation
    float* ptr2 = allocator.allocate(10000);
    EXPECT_NE(nullptr, ptr2);
    allocator.deallocate(ptr2, 10000);

    LOGGING_LOG_INFO("CUDA caching allocator template typed allocation test passed");
}

/**
 * @brief Test template allocator alignment requirements
 */
MEMORYTEST_F(CudaCachingAllocatorTemplate, respects_alignment_requirements)
{
    cuda_caching_allocator_template<double, 512> allocator(0, 16 * 1024ULL);

    // Allocate memory and check alignment
    double* ptr = allocator.allocate(50);
    EXPECT_NE(nullptr, ptr);

    // Check alignment (should be aligned to 512 bytes)
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(0, addr % 512);

    allocator.deallocate(ptr, 50);

    LOGGING_LOG_INFO("CUDA caching allocator template alignment test passed");
}

/**
 * @brief Test template allocator statistics and cache operations
 */
MEMORYTEST_F(CudaCachingAllocatorTemplate, provides_statistics_and_cache_control)
{
    cuda_caching_allocator_template<int, 256> allocator(0, 8 * 1024ULL);

    // Get initial stats
    auto initial_stats = allocator.stats();

    // Perform allocations
    int* ptr1 = allocator.allocate(1000);
    int* ptr2 = allocator.allocate(2000);

    // Check stats updated
    auto after_stats = allocator.stats();
    EXPECT_GT(after_stats.bytes_allocated.load(), initial_stats.bytes_allocated.load());

    // Deallocate
    allocator.deallocate(ptr1, 1000);
    allocator.deallocate(ptr2, 2000);
    EXPECT_EQ(0, allocator.stats().bytes_allocated.load());

    // Test cache clearing
    allocator.empty_cache();

    LOGGING_LOG_INFO("CUDA caching allocator template statistics test passed");
}

// ============================================================================
// PyTorch parity behavior (rounding, segmentation, split/merge, per-stream
// pools, OOM cache-flush retry chain, recordStream)
// ============================================================================

/**
 * @brief Requests are rounded to 512-byte multiples before block sizing
 */
MEMORYTEST_F(CudaCachingAllocator, rounds_requests_to_512_byte_multiples)
{
    cuda_caching_allocator allocator(0);

    void* ptr1 = allocator.allocate(1);  // rounds to 512
    void* ptr2 = allocator.allocate(512);
    ASSERT_NE(nullptr, ptr1);
    ASSERT_NE(nullptr, ptr2);

    auto stats = allocator.stats();
    // Both allocations are 512-byte blocks inside one 2 MiB small segment
    EXPECT_EQ(1024, stats.bytes_allocated.load());
    EXPECT_EQ(2 * 1024 * 1024, stats.bytes_reserved.load());

    allocator.deallocate(ptr1, 1);
    allocator.deallocate(ptr2, 512);
    EXPECT_EQ(0, allocator.stats().bytes_allocated.load());

    LOGGING_LOG_INFO("CUDA caching allocator size rounding test passed");
}

/**
 * @brief Small requests are packed into shared 2 MiB segments (one cudaMalloc)
 */
MEMORYTEST_F(CudaCachingAllocator, packs_small_allocations_into_one_segment)
{
    cuda_caching_allocator allocator(0);

    void* ptr1 = allocator.allocate(1024);
    void* ptr2 = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr1);
    ASSERT_NE(nullptr, ptr2);

    auto stats = allocator.stats();
    EXPECT_EQ(1, stats.driver_allocations.load());  // single segment cudaMalloc
    EXPECT_EQ(2 * 1024 * 1024, stats.bytes_reserved.load());
    EXPECT_EQ(2048, stats.bytes_allocated.load());

    allocator.deallocate(ptr1, 1024);
    allocator.deallocate(ptr2, 1024);

    LOGGING_LOG_INFO("CUDA caching allocator small segment packing test passed");
}

/**
 * @brief Freed blocks are reused from the cache without new driver calls
 */
MEMORYTEST_F(CudaCachingAllocator, reuses_cached_blocks)
{
    cuda_caching_allocator allocator(0);

    void* ptr1 = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr1);
    allocator.deallocate(ptr1, 1024);

    void* ptr2 = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr2);
    // Merge on free restores the full free segment, so the same address comes back
    EXPECT_EQ(ptr1, ptr2);

    auto stats = allocator.stats();
    EXPECT_EQ(1, stats.driver_allocations.load());
    EXPECT_EQ(1, stats.cache_hits.load());
    EXPECT_EQ(1, stats.cache_misses.load());

    allocator.deallocate(ptr2, 1024);

    LOGGING_LOG_INFO("CUDA caching allocator cached block reuse test passed");
}

/**
 * @brief Oversized cached blocks are split and the remainder returned to the pool
 */
MEMORYTEST_F(CudaCachingAllocator, splits_oversized_cached_blocks)
{
    cuda_caching_allocator allocator(0);

    void* big = allocator.allocate(4 * 1024 * 1024);  // 4 MiB block from a 20 MiB segment
    ASSERT_NE(nullptr, big);
    allocator.deallocate(big, 4 * 1024 * 1024);

    // The free-block search never crosses the small/large pool boundary
    // (kSmallSize == 1 MiB, see caching_allocator_config.h), so the second
    // request must stay in the same (large) pool as "big" -- i.e. > 1 MiB --
    // for this to exercise a same-segment split instead of a fresh cudaMalloc.
    // A 2 MiB request should split the cached 20 MiB segment, leaving an
    // 18 MiB remainder.
    void* small = allocator.allocate(2 * 1024 * 1024);
    ASSERT_NE(nullptr, small);
    EXPECT_EQ(big, small);

    auto stats = allocator.stats();
    EXPECT_EQ(1, stats.driver_allocations.load());  // no new cudaMalloc for the split
    EXPECT_EQ(2 * 1024 * 1024, stats.bytes_allocated.load());
    EXPECT_EQ(18 * 1024 * 1024, stats.inactive_split_bytes.load());

    allocator.deallocate(small, 2 * 1024 * 1024);
    // Merge restores the whole 20 MiB block; nothing split remains
    EXPECT_EQ(0, allocator.stats().inactive_split_bytes.load());

    LOGGING_LOG_INFO("CUDA caching allocator block split test passed");
}

/**
 * @brief Free blocks are scoped to their allocation stream and never reused
 *        on a different stream
 */
MEMORYTEST_F(CudaCachingAllocator, never_reuses_blocks_across_streams)
{
    cuda_caching_allocator allocator(0);

    cudaStream_t stream_a = nullptr;
    cudaStream_t stream_b = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_a));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_b));

    void* ptr_a = allocator.allocate(1024, stream_a);
    ASSERT_NE(nullptr, ptr_a);
    allocator.deallocate(ptr_a, 1024, stream_a);

    auto stats = allocator.stats();
    EXPECT_EQ(1, stats.driver_allocations.load());

    // Same-size request on stream_b must miss: cached blocks are stream-scoped
    void* ptr_b = allocator.allocate(1024, stream_b);
    ASSERT_NE(nullptr, ptr_b);

    stats = allocator.stats();
    EXPECT_EQ(2, stats.driver_allocations.load());  // new segment for stream_b
    EXPECT_EQ(2, stats.cache_misses.load());
    EXPECT_EQ(0, stats.cache_hits.load());

    allocator.deallocate(ptr_b, 1024, stream_b);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream_a));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream_b));

    LOGGING_LOG_INFO("CUDA caching allocator per-stream pool test passed");
}

/**
 * @brief record_stream defers reuse of a cross-stream allocation until the
 *        recorded stream's pending work completes
 */
MEMORYTEST_F(CudaCachingAllocator, defers_reuse_until_recorded_stream_completes)
{
    cuda_caching_allocator allocator(0);

    cudaStream_t alloc_stream = nullptr;
    cudaStream_t use_stream   = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&alloc_stream));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&use_stream));

    // Keep use_stream genuinely busy so the event the allocator records on it
    // (via record_stream/deallocate below) cannot complete until released --
    // without this, use_stream is idle and the event resolves immediately,
    // making the deferred-reuse behavior this test targets unobservable.
    std::atomic<bool> release_use_stream{false};
    ASSERT_EQ(
        cudaSuccess,
        cudaLaunchHostFunc(use_stream, block_stream_until_released, &release_use_stream));
    // WDDM batches queued stream work rather than submitting it immediately;
    // querying the stream forces the driver to flush the batch so the host
    // function actually starts running instead of sitting unsubmitted.
    (void)cudaStreamQuery(use_stream);

    // Pick a request whose segment_size_for() result equals the request
    // itself, so alloc_found_block_locked's should_split() sees a zero
    // remainder and caches nothing extra. 10 MiB hits the >= kMinLargeAlloc
    // branch of segment_size_for(), which rounds up to a 2 MiB multiple --
    // 10 MiB is already one, so segment_size_for(10 MiB) == 10 MiB exactly.
    // (A smaller request, e.g. 1024 bytes or even one exact small/large
    // buffer size below this threshold, rounds up to a bigger segment and
    // should_split() then immediately caches the leftover remainder as its
    // own free block -- independent of the deferred block below -- so a
    // second same-size request would reuse that leftover space instead of
    // exercising the deferral this test targets.)
    size_t const segment_size = 10 * 1024 * 1024;

    void* ptr = allocator.allocate(segment_size, alloc_stream);
    ASSERT_NE(nullptr, ptr);

    allocator.record_stream(ptr, use_stream);
    allocator.deallocate(ptr, segment_size, alloc_stream);

    // The block is withheld pending the use_stream event; a same-stream request
    // must not see it (and there is nothing else cached), so a new segment appears
    void* ptr2 = allocator.allocate(segment_size, alloc_stream);
    ASSERT_NE(nullptr, ptr2);
    EXPECT_EQ(2, allocator.stats().driver_allocations.load());
    allocator.deallocate(ptr2, segment_size, alloc_stream);

    // Release use_stream and wait for its pending work (and the allocator's
    // recorded event) to actually complete before expecting reclaim.
    release_use_stream.store(true, std::memory_order_release);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(use_stream));
    void* ptr3 = allocator.allocate(segment_size, alloc_stream);
    ASSERT_NE(nullptr, ptr3);
    EXPECT_EQ(ptr, ptr3);

    allocator.deallocate(ptr3, segment_size, alloc_stream);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(alloc_stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(use_stream));

    LOGGING_LOG_INFO("CUDA caching allocator record_stream deferral test passed");
}

/**
 * @brief empty_cache releases cached segments so they can be re-cudaMalloc'd
 */
MEMORYTEST_F(CudaCachingAllocator, empty_cache_releases_cached_segments)
{
    cuda_caching_allocator allocator(0);

    void* ptr = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr);
    allocator.deallocate(ptr, 1024);

    auto stats = allocator.stats();
    EXPECT_EQ(1, stats.driver_allocations.load());
    EXPECT_GT(stats.bytes_cached.load(), 0);
    EXPECT_GT(stats.bytes_reserved.load(), 0);

    allocator.empty_cache();

    stats = allocator.stats();
    EXPECT_EQ(0, stats.bytes_cached.load());
    EXPECT_EQ(0, stats.bytes_reserved.load());
    EXPECT_EQ(1, stats.driver_frees.load());

    // The released segment is gone, so the next allocation hits the driver again
    void* ptr2 = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr2);
    EXPECT_EQ(2, allocator.stats().driver_allocations.load());
    allocator.deallocate(ptr2, 1024);

    LOGGING_LOG_INFO("CUDA caching allocator empty_cache release test passed");
}

/**
 * @brief A cache cap trims cached segments on deallocate, keeping in-use
 *        allocations fully functional
 */
MEMORYTEST_F(CudaCachingAllocator, cache_cap_trims_on_deallocate)
{
    // max_cached_bytes is an "at most" cap (bytes_cached_ == cap does not trim,
    // matching the class's own "Maximum bytes to cache" contract), so the cap
    // must be strictly below the 2 MiB segment for this test to trigger a trim.
    cuda_caching_allocator allocator(0, 2 * 1024 * 1024 - 1);

    void* ptr = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr);
    allocator.deallocate(ptr, 1024);

    // Cap is below the 2 MiB segment, so the cached segment is trimmed immediately
    auto stats = allocator.stats();
    EXPECT_EQ(0, stats.bytes_cached.load());
    EXPECT_EQ(0, stats.bytes_reserved.load());
    EXPECT_EQ(1, stats.driver_frees.load());
    EXPECT_EQ(1, stats.cache_evictions.load());

    LOGGING_LOG_INFO("CUDA caching allocator cache cap trim test passed");
}

/**
 * @brief Requests larger than 10 MiB are rounded up to 2 MiB multiples and
 *        allocated as right-sized segments
 */
MEMORYTEST_F(CudaCachingAllocator, rounds_huge_allocations_to_2_mib_multiples)
{
    cuda_caching_allocator allocator(0);

    size_t const request = 11 * 1024 * 1024 + 1;  // just over 11 MiB -> 12 MiB segment
    void*        ptr     = allocator.allocate(request);
    ASSERT_NE(nullptr, ptr);

    auto stats = allocator.stats();
    EXPECT_EQ(1, stats.driver_allocations.load());
    EXPECT_EQ(12 * 1024 * 1024, stats.bytes_reserved.load());
    // 12 MiB - 11 MiB - 512 B remainder is below the 1 MiB split threshold,
    // so the whole segment backs the allocation
    EXPECT_EQ(12 * 1024 * 1024, stats.bytes_allocated.load());

    allocator.deallocate(ptr, request);
    EXPECT_EQ(0, allocator.stats().inactive_split_bytes.load());

    LOGGING_LOG_INFO("CUDA caching allocator huge allocation rounding test passed");
}

/**
 * @brief Free-memory callbacks run between the first cache miss and the
 *        cudaMalloc fallback; a callback that frees memory makes the cache
 *        retriable (upstream trigger_free_memory_callbacks)
 */
MEMORYTEST_F(CudaCachingAllocator, free_memory_callbacks_run_before_driver_fallback)
{
    cuda_caching_allocator allocator(0);

    void* big = allocator.allocate(4 * 1024 * 1024);  // 4 MiB block, 16 MiB remainder cached
    ASSERT_NE(nullptr, big);
    EXPECT_EQ(1, allocator.stats().driver_allocations.load());

    bool invoked = false;
    allocator.add_free_memory_callback(
        [&]()
        {
            invoked = true;
            // Runs under the allocator lock (recursive): free the block so the
            // retry can serve the request from the cache.
            allocator.deallocate(big, 4 * 1024 * 1024);
            return true;
        });

    // 17 MiB request: the cached 16 MiB remainder is too small, so the callback
    // must run; freeing `big` merges the segment back to 20 MiB and the retry hits.
    void* ptr = allocator.allocate(17 * 1024 * 1024);
    ASSERT_NE(nullptr, ptr);
    EXPECT_TRUE(invoked);
    EXPECT_EQ(ptr, big);
    EXPECT_EQ(1, allocator.stats().driver_allocations.load());  // no cudaMalloc happened

    allocator.deallocate(ptr, 17 * 1024 * 1024);

    LOGGING_LOG_INFO("CUDA caching allocator free-memory callback test passed");
}

/**
 * @brief Equal-size free segments recycle oldest-first (upstream
 *        registration_counter FIFO tie-break)
 */
MEMORYTEST_F(CudaCachingAllocator, recycles_equal_size_segments_fifo)
{
    cuda_caching_allocator allocator(0);

    // 17 MiB rounds to an 18 MiB segment whose 1 MiB remainder is below the
    // split threshold, so each request consumes a whole new segment.
    void* ptr_a = allocator.allocate(17 * 1024 * 1024);
    void* ptr_b = allocator.allocate(17 * 1024 * 1024);
    ASSERT_NE(nullptr, ptr_a);
    ASSERT_NE(nullptr, ptr_b);
    ASSERT_NE(ptr_a, ptr_b);
    EXPECT_EQ(2, allocator.stats().driver_allocations.load());

    allocator.deallocate(ptr_a, 17 * 1024 * 1024);  // freed first, oldest segment
    allocator.deallocate(ptr_b, 17 * 1024 * 1024);

    void* ptr_c = allocator.allocate(17 * 1024 * 1024);
    ASSERT_NE(nullptr, ptr_c);
    EXPECT_EQ(ptr_a, ptr_c);  // oldest segment recycles first
    void* ptr_d = allocator.allocate(17 * 1024 * 1024);
    ASSERT_NE(nullptr, ptr_d);
    EXPECT_EQ(ptr_b, ptr_d);
    EXPECT_EQ(2, allocator.stats().driver_allocations.load());  // no new segments

    allocator.deallocate(ptr_c, 17 * 1024 * 1024);
    allocator.deallocate(ptr_d, 17 * 1024 * 1024);

    LOGGING_LOG_INFO("CUDA caching allocator FIFO recycling test passed");
}

/**
 * @brief Every cache release pass (empty_cache / OOM flush) is counted in
 *        num_sync_all_streams (upstream DeviceStats parity)
 */
MEMORYTEST_F(CudaCachingAllocator, counts_sync_all_streams_on_cache_release)
{
    cuda_caching_allocator allocator(0);

    EXPECT_EQ(0, allocator.stats().num_sync_all_streams.load());
    allocator.empty_cache();
    allocator.empty_cache();
    EXPECT_EQ(2, allocator.stats().num_sync_all_streams.load());

    LOGGING_LOG_INFO("CUDA caching allocator sync-all-streams counter test passed");
}

MEMORYTEST_F(CudaCachingAllocator, data_ptr_move_assign_returns_block_to_cache)
{
    using ptr_t          = data_ptr<float>;
    auto&      cache     = caching_allocator_for_device(0);
    auto const allocated = cache.stats().bytes_allocated.load();

    {
        ptr_t first(1024, device_enum::CUDA);
        ptr_t second(1024, device_enum::CUDA);
        EXPECT_EQ(0, first.device_index());
        EXPECT_EQ(device_enum::CUDA, first.device());
        first = std::move(second);
    }

    EXPECT_EQ(allocated, cache.stats().bytes_allocated.load());
    LOGGING_LOG_INFO("data_ptr move-assign returns GPU block to cache");
}

MEMORYTEST_F(CudaCachingAllocator, data_ptr_copy_assign_returns_block_to_cache)
{
    using ptr_t          = data_ptr<float>;
    auto&      cache     = caching_allocator_for_device(0);
    auto const allocated = cache.stats().bytes_allocated.load();

    {
        ptr_t first(256, device_enum::CUDA);
        ptr_t second(256, device_enum::CUDA);
        first = second;
    }

    EXPECT_EQ(allocated, cache.stats().bytes_allocated.load());
    LOGGING_LOG_INFO("data_ptr copy-assign returns GPU block to cache");
}

MEMORYTEST_F(CudaCachingAllocator, data_ptr_uses_allocation_stream_pool)
{
    using ptr_t           = data_ptr<float>;
    cudaStream_t stream_a = nullptr;
    cudaStream_t stream_b = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_a));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_b));

    auto&      cache   = caching_allocator_for_device(0);
    auto const drivers = cache.stats().driver_allocations.load();

    {
        ptr_t first(256, device_enum::CUDA, 0, stream_a);
        EXPECT_EQ(stream_a, first.stream());
    }
    {
        ptr_t second(256, device_enum::CUDA, 0, stream_a);
        EXPECT_EQ(stream_a, second.stream());
    }
    EXPECT_EQ(drivers + 1, cache.stats().driver_allocations.load());

    {
        ptr_t other(256, device_enum::CUDA, 0, stream_b);
        EXPECT_EQ(stream_b, other.stream());
    }
    EXPECT_EQ(drivers + 2, cache.stats().driver_allocations.load());

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream_a));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream_b));
    LOGGING_LOG_INFO("data_ptr allocates and frees on the given CUDA stream");
}

#if MEMORY_HAS_PROFILER

/**
 * @brief report_caching_allocator_delta (gpu/caching_allocator_profiler_report.h)
 * is a test helper that diffs two unified_cache_stats snapshots into one
 * report_caching_allocator_event. Production allocate/deallocate report the
 * known block size from Impl instead. No Kineto/ITT session is active in this
 * test binary, so report_memory_usage() no-ops internally; this only verifies
 * the helper is safe to call for both a live pointer and the deallocate(nullptr)
 * case, with the real CUDA device_type (1).
 */
MEMORYTEST_F(CudaCachingAllocator, report_caching_allocator_delta_does_not_crash)
{
    unified_cache_stats before;
    unified_cache_stats after;
    before.bytes_allocated = 100;
    after.bytes_allocated  = 356;
    after.bytes_reserved   = 2048;

    int dummy_block = 0;
    EXPECT_NO_THROW({
        report_caching_allocator_delta(
            &dummy_block, before, after, /*device_index=*/0, /*device_type=*/1);
    });
    EXPECT_NO_THROW({
        report_caching_allocator_delta(
            nullptr, before, after, /*device_index=*/0, /*device_type=*/1);
    });

    LOGGING_LOG_INFO("CUDA caching allocator profiler-report helper test passed");
}

#endif  // MEMORY_HAS_PROFILER

MEMORYTEST_F(CudaCachingAllocator, process_wide_api_matches_pytorch)
{
    using alloc_t = allocator<float>;
    alloc_t::empty_cache(0);

    size_t const before_alloc = alloc_t::memory_allocated(0);
    float*       ptr          = alloc_t::allocate(1024, device_enum::CUDA);
    ASSERT_NE(nullptr, ptr);

    EXPECT_GT(alloc_t::memory_allocated(0), before_alloc);
    EXPECT_GE(alloc_t::max_memory_allocated(0), alloc_t::memory_allocated(0));
    EXPECT_GT(alloc_t::memory_reserved(0), 0U);
    EXPECT_GT(gpu::device_total_memory(0), 0U);

    alloc_t::reset_peak_memory_stats(0);
    EXPECT_EQ(alloc_t::max_memory_allocated(0), alloc_t::memory_allocated(0));

    EXPECT_THROW(alloc_t::set_memory_fraction(0.0, 0), std::invalid_argument);
    alloc_t::set_memory_fraction(1.0, 0);

    alloc_t::free(ptr, device_enum::CUDA);
    alloc_t::empty_cache(0);
    LOGGING_LOG_INFO("process-wide GPU cache API test passed");
}

MEMORYTEST_F(CudaCachingAllocator, expandable_segments_default_off)
{
    cuda_caching_allocator allocator(0);
    EXPECT_FALSE(allocator.expandable_segments());

    void* ptr = allocator.allocate(1024);
    ASSERT_NE(nullptr, ptr);
    allocator.deallocate(ptr, 1024);

    allocator.set_expandable_segments(true);
    EXPECT_TRUE(allocator.expandable_segments());
    void* ptr_vm = allocator.allocate(2048);
    ASSERT_NE(nullptr, ptr_vm);
    allocator.deallocate(ptr_vm, 2048);
    allocator.set_expandable_segments(false);
    EXPECT_FALSE(allocator.expandable_segments());
}

MEMORYTEST_F(CudaCachingAllocator, same_size_alloc_free_churn)
{
    cuda_caching_allocator allocator(0);
    constexpr size_t       kBytes      = 4096;
    constexpr int          kIterations = 100000;
    for (int i = 0; i < kIterations; ++i)
    {
        void* ptr = allocator.allocate(kBytes);
        ASSERT_NE(nullptr, ptr);
        allocator.deallocate(ptr, kBytes);
    }
    EXPECT_EQ(0U, allocator.stats().bytes_allocated.load());
}

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP
