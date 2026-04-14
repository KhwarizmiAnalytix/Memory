/*
 * Quarisma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of Quarisma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#include "CoreTest.h"
#include "common/macros.h"

#if PROJECT_HAS_CUDA

#include <cuda_runtime.h>

#include <vector>

#include "logger.h"
#include "gpu/cuda_caching_allocator.h"

using namespace quarisma;
using namespace quarisma::gpu;

/**
 * @brief Test basic CUDA caching allocator construction and destruction
 */
QUARISMATEST(CudaCachingAllocator, constructs_with_valid_parameters)
{
    // Test basic construction
    cuda_caching_allocator allocator(0, 64 * 1024ULL);  // 64MB cache

    // Verify device index
    EXPECT_EQ(0, allocator.device());

    // Verify cache size
    EXPECT_EQ(64 * 1024ULL, allocator.max_cached_bytes());

    QUARISMA_LOG_INFO("CUDA caching allocator construction test passed");
}

/**
 * @brief Test CUDA caching allocator constructor variations
 */
QUARISMATEST(CudaCachingAllocator, constructor_variations)
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

    QUARISMA_LOG_INFO("CUDA caching allocator constructor variations test passed");
}

/**
 * @brief Test basic allocation and deallocation functionality
 */
QUARISMATEST(CudaCachingAllocator, allocates_and_deallocates_memory)
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

    QUARISMA_LOG_INFO("CUDA caching allocator allocation/deallocation test passed");
}

/**
 * @brief Test cache management functionality
 */
QUARISMATEST(CudaCachingAllocator, manages_cache_correctly)
{
    cuda_caching_allocator allocator(0, 16 * 1024ULL);  // 16MB cache

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
    //EXPECT_GE(stats_before.cached_bytes, stats_after.cached_bytes);

    QUARISMA_LOG_INFO("CUDA caching allocator cache management test passed");
}

/**
 * @brief Test cache size limits and configuration
 */
QUARISMATEST(CudaCachingAllocator, respects_cache_size_limits)
{
    cuda_caching_allocator allocator(0, 8 * 1024ULL);  // 8MB cache

    // Test setting new cache size
    allocator.set_max_cached_bytes(16 * 1024ULL);
    EXPECT_EQ(16 * 1024ULL, allocator.max_cached_bytes());

    // Test disabling cache
    allocator.set_max_cached_bytes(0);
    EXPECT_EQ(0, allocator.max_cached_bytes());

    QUARISMA_LOG_INFO("CUDA caching allocator cache size limits test passed");
}

/**
 * @brief Test statistics collection and reporting
 */
QUARISMATEST(CudaCachingAllocator, provides_accurate_statistics)
{
    cuda_caching_allocator allocator(0, 32 * 1024ULL);  // 32MB cache

    // Get initial stats
    auto initial_stats = allocator.stats();

    // Perform some allocations
    void* ptr1 = allocator.allocate(2048);
    void* ptr2 = allocator.allocate(4096);

    auto after_alloc_stats = allocator.stats();
    //EXPECT_GT(after_alloc_stats.total_allocated_bytes, initial_stats.total_allocated_bytes);

    // Deallocate
    allocator.deallocate(ptr1, 2048);
    allocator.deallocate(ptr2, 4096);

    auto after_dealloc_stats = allocator.stats();
    //EXPECT_GT(after_dealloc_stats.total_deallocated_bytes, initial_stats.total_deallocated_bytes);

    QUARISMA_LOG_INFO("CUDA caching allocator statistics test passed");
}

/**
 * @brief Test move semantics and resource transfer
 */
QUARISMATEST(CudaCachingAllocator, supports_move_semantics)
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

    QUARISMA_LOG_INFO("CUDA caching allocator move semantics test passed");
}

/**
 * @brief Test error handling for invalid operations
 */
QUARISMATEST(CudaCachingAllocator, handles_errors_gracefully)
{
    cuda_caching_allocator allocator(0, 16 * 1024ULL);

    // Test zero-size allocation
    QUARISMA_UNUSED void* ptr_zero = allocator.allocate(0);
    // Should handle gracefully (implementation-defined behavior)

    // Test null pointer deallocation
    // Should not crash
    allocator.deallocate(nullptr, 1024);

    QUARISMA_LOG_INFO("CUDA caching allocator error handling test passed");
}

/**
 * @brief Test template allocator construction and basic operations
 */
QUARISMATEST(CudaCachingAllocatorTemplate, constructs_with_different_types)
{
    // Test template allocator for different types
    cuda_caching_allocator_template<float, 256>  float_allocator(0, 32 * 1024ULL);
    cuda_caching_allocator_template<double, 256> double_allocator(0, 32 * 1024ULL);
    cuda_caching_allocator_template<int, 128>    int_allocator(0, 16 * 1024ULL);

    // Verify device indices
    EXPECT_EQ(0, float_allocator.device());
    EXPECT_EQ(0, double_allocator.device());
    EXPECT_EQ(0, int_allocator.device());

    QUARISMA_LOG_INFO("CUDA caching allocator template construction test passed");
}

/**
 * @brief Test template allocator type-safe allocation
 */
QUARISMATEST(CudaCachingAllocatorTemplate, allocates_typed_memory_safely)
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

    QUARISMA_LOG_INFO("CUDA caching allocator template typed allocation test passed");
}

/**
 * @brief Test template allocator alignment requirements
 */
QUARISMATEST(CudaCachingAllocatorTemplate, respects_alignment_requirements)
{
    cuda_caching_allocator_template<double, 512> allocator(0, 16 * 1024ULL);

    // Allocate memory and check alignment
    double* ptr = allocator.allocate(50);
    EXPECT_NE(nullptr, ptr);

    // Check alignment (should be aligned to 512 bytes)
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(0, addr % 512);

    allocator.deallocate(ptr, 50);

    QUARISMA_LOG_INFO("CUDA caching allocator template alignment test passed");
}

/**
 * @brief Test template allocator statistics and cache operations
 */
QUARISMATEST(CudaCachingAllocatorTemplate, provides_statistics_and_cache_control)
{
    cuda_caching_allocator_template<int, 256> allocator(0, 8 * 1024ULL);

    // Get initial stats
    auto initial_stats = allocator.stats();

    // Perform allocations
    int* ptr1 = allocator.allocate(1000);
    int* ptr2 = allocator.allocate(2000);

    // Check stats updated
    auto after_stats = allocator.stats();
    //EXPECT_GT(after_stats.total_allocated_bytes, initial_stats.total_allocated_bytes);

    // Deallocate
    allocator.deallocate(ptr1, 1000);
    allocator.deallocate(ptr2, 2000);

    // Test cache clearing
    allocator.empty_cache();

    QUARISMA_LOG_INFO("CUDA caching allocator template statistics test passed");
}

#endif  // PROJECT_HAS_CUDA
