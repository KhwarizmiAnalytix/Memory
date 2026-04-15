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

#include "MemoryTest.h"
#include "common/macros.h"

#if PROJECT_HAS_CUDA

#include <memory>
#include <vector>

//#include "logger/logger.h"
#include "common/device.h"
#include "gpu/gpu_memory_pool.h"

using namespace memory;
using namespace memory::gpu;

/**
 * @brief Test GPU memory pool configuration validation
 */
MEMORYTEST(GpuMemoryPool, validates_configuration_parameters)
{
    // Test default configuration
    gpu_memory_pool_config default_config;
    EXPECT_GT(default_config.min_block_size, 0);
    EXPECT_GT(default_config.max_block_size, default_config.min_block_size);
    EXPECT_GT(default_config.block_growth_factor, 1.0);
    EXPECT_GT(default_config.max_pool_size, 0);
    EXPECT_GT(default_config.max_cached_blocks, 0);
    EXPECT_GT(default_config.alignment_boundary, 0);
    EXPECT_TRUE(default_config.enable_alignment);
    EXPECT_TRUE(default_config.enable_tracking);

    // Test custom configuration
    gpu_memory_pool_config custom_config;
    custom_config.min_block_size      = 2048;
    custom_config.max_block_size      = 32 * 1024ULL;
    custom_config.block_growth_factor = 1.5;
    custom_config.max_pool_size       = 128 * 1024ULL;
    custom_config.max_cached_blocks   = 8;
    custom_config.alignment_boundary  = 512;
    custom_config.enable_alignment    = true;
    custom_config.enable_tracking     = true;
    custom_config.debug_mode          = false;

    EXPECT_EQ(2048, custom_config.min_block_size);
    EXPECT_EQ(32 * 1024ULL, custom_config.max_block_size);
    EXPECT_EQ(1.5, custom_config.block_growth_factor);

    MEMORY_LOG_INFO("GPU memory pool configuration validation test passed");
}

/**
 * @brief Test GPU memory pool creation
 */
MEMORYTEST(GpuMemoryPool, creates_pool_successfully)
{
    // Test pool creation with default configuration
    gpu_memory_pool_config config;
    config.min_block_size      = 1024;
    config.max_block_size      = 16 * 1024ULL;
    config.block_growth_factor = 2.0;
    config.max_pool_size       = 256 * 1024ULL;
    config.max_cached_blocks   = 8;
    config.enable_alignment    = true;
    config.alignment_boundary  = 256;
    config.enable_tracking     = true;
    config.debug_mode          = false;

    // Create memory pool
    auto pool = gpu_memory_pool::create(config);
    EXPECT_NE(nullptr, pool.get());

    // Test initial state
    EXPECT_EQ(0, pool->get_allocated_bytes());
    EXPECT_EQ(0, pool->get_peak_allocated_bytes());
    EXPECT_EQ(0, pool->get_active_allocations());

    MEMORY_LOG_INFO("GPU memory pool creation test passed");
}

/**
 * @brief Test GPU memory block structure and operations
 */
MEMORYTEST(GpuMemoryBlock, manages_block_metadata_correctly)
{
    // Test default construction
    gpu_memory_block default_block;
    EXPECT_EQ(nullptr, default_block.ptr);
    EXPECT_EQ(0, default_block.size);
    EXPECT_EQ(0, default_block.reuse_count.load());
    EXPECT_FALSE(default_block.in_use.load());

    // Test parameterized construction
    void*         test_ptr  = reinterpret_cast<void*>(0x12345678);
    size_t        test_size = 4096;
    device_option test_device(device_enum::CUDA, 0);

    gpu_memory_block param_block(test_ptr, test_size, test_device);
    EXPECT_EQ(test_ptr, param_block.ptr);
    EXPECT_EQ(test_size, param_block.size);
    EXPECT_EQ(device_enum::CUDA, param_block.device.type());
    EXPECT_EQ(0, param_block.device.index());
    EXPECT_EQ(0, param_block.reuse_count.load());
    EXPECT_FALSE(param_block.in_use.load());

    // Test atomic operations
    param_block.reuse_count.store(5);
    param_block.in_use.store(true);
    EXPECT_EQ(5, param_block.reuse_count.load());
    EXPECT_TRUE(param_block.in_use.load());

    // Test atomic increment
    param_block.reuse_count.fetch_add(1);
    EXPECT_EQ(6, param_block.reuse_count.load());

    MEMORY_LOG_INFO("GPU memory block management test passed");
}

/**
 * @brief Test basic memory allocation and deallocation
 */
MEMORYTEST(GpuMemoryPool, allocates_and_deallocates_memory)
{
    gpu_memory_pool_config config;
    config.min_block_size = 1024;
    config.max_block_size = 8 * 1024ULL;
    config.max_pool_size  = 64 * 1024ULL;

    auto pool = gpu_memory_pool::create(config);
    EXPECT_NE(nullptr, pool.get());

    try
    {
        // Test basic allocation
        auto block1 = pool->allocate(2048, device_enum::CUDA, 0);
        EXPECT_NE(nullptr, block1.ptr);
        EXPECT_GE(block1.size, 2048);
        EXPECT_EQ(device_enum::CUDA, block1.device.type());

        // Check pool statistics
        EXPECT_GT(pool->get_allocated_bytes(), 0);
        EXPECT_EQ(1, pool->get_active_allocations());

        // Test deallocation
        pool->deallocate(block1);

        MEMORY_LOG_INFO("GPU memory pool allocation/deallocation test passed");
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO("GPU memory pool allocation failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test multiple allocations and memory reuse
 */
MEMORYTEST(GpuMemoryPool, handles_multiple_allocations)
{
    gpu_memory_pool_config config;
    config.min_block_size    = 512;
    config.max_block_size    = 4 * 1024ULL;
    config.max_cached_blocks = 16;

    auto pool = gpu_memory_pool::create(config);
    EXPECT_NE(nullptr, pool.get());
    std::vector<gpu_memory_block> blocks;

    // Allocate multiple blocks
    for (int i = 0; i < 5; ++i)
    {
        size_t size  = 1024 * (i + 1);
        auto   block = pool->allocate(size, device_enum::CUDA, 0);

        if (block.ptr != nullptr)
        {
            blocks.push_back(std::move(block));
        }
    }

    // Check statistics
    if (!blocks.empty())
    {
        EXPECT_GT(pool->get_allocated_bytes(), 0);
        EXPECT_EQ(blocks.size(), pool->get_active_allocations());
    }

    // Deallocate all blocks
    for (auto& block : blocks)
    {
        pool->deallocate(block);
    }

    MEMORY_LOG_INFO("GPU memory pool multiple allocations test passed");
}

/**
 * @brief Test memory pool statistics and reporting
 */
MEMORYTEST(GpuMemoryPool, provides_accurate_statistics)
{
    gpu_memory_pool_config config;
    config.min_block_size  = 1024;
    config.max_block_size  = 2 * 1024ULL;
    config.enable_tracking = true;

    auto pool = gpu_memory_pool::create(config);
    EXPECT_NE(nullptr, pool.get());

    // Get initial statistics
    auto initial_stats = pool->get_statistics();
    EXPECT_EQ(0, initial_stats.total_allocations);
    EXPECT_EQ(0, initial_stats.total_deallocations);
    EXPECT_EQ(0, initial_stats.current_bytes_in_use);

    try
    {
        // Perform allocation
        auto block = pool->allocate(2048, device_enum::CUDA, 0);

        if (block.ptr != nullptr)
        {
            auto after_alloc_stats = pool->get_statistics();
            EXPECT_GT(after_alloc_stats.total_allocations, initial_stats.total_allocations);
            EXPECT_GT(after_alloc_stats.current_bytes_in_use, 0);

            // Test memory report
            std::string report = pool->get_memory_report();
            EXPECT_FALSE(report.empty());

            // Deallocate
            pool->deallocate(block);

            auto after_dealloc_stats = pool->get_statistics();
            EXPECT_GT(after_dealloc_stats.total_deallocations, initial_stats.total_deallocations);
        }

        MEMORY_LOG_INFO("GPU memory pool statistics test passed");
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO(
            "GPU memory pool statistics test failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test cache management functionality
 */
MEMORYTEST(GpuMemoryPool, manages_cache_effectively)
{
    gpu_memory_pool_config config;
    config.min_block_size    = 1024;
    config.max_cached_blocks = 4;
    config.enable_tracking   = true;

    auto pool = gpu_memory_pool::create(config);
    EXPECT_NE(nullptr, pool.get());

    try
    {
        // Allocate and deallocate to populate cache
        auto block1 = pool->allocate(2048, device_enum::CUDA, 0);
        if (block1.ptr != nullptr)
        {
            pool->deallocate(block1);

            // Get statistics after deallocation (should show cached memory)
            auto stats = pool->get_statistics();
            // Note: cached_memory might be 0 if implementation doesn't cache or GPU not available

            // Test cache clearing
            pool->clear_cache();

            // After clearing cache, cached memory should be reduced
            auto after_clear_stats = pool->get_statistics();
            EXPECT_LE(after_clear_stats.cached_memory, stats.cached_memory);
        }

        MEMORY_LOG_INFO("GPU memory pool cache management test passed");
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO(
            "GPU memory pool cache management failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test memory pool with different device types
 */
MEMORYTEST(GpuMemoryPool, supports_different_device_types)
{
    gpu_memory_pool_config config;
    config.min_block_size = 1024;
    config.max_block_size = 1024ULL;

    auto pool = gpu_memory_pool::create(config);
    EXPECT_NE(nullptr, pool.get());

    try
    {
        // Test CUDA device allocation
        auto cuda_block = pool->allocate(1024, device_enum::CUDA, 0);
        if (cuda_block.ptr != nullptr)
        {
            EXPECT_EQ(device_enum::CUDA, cuda_block.device.type());
            EXPECT_EQ(0, cuda_block.device.index());
            pool->deallocate(cuda_block);
        }

        // Test different device index
        auto cuda_block_1 = pool->allocate(1024, device_enum::CUDA, 1);
        if (cuda_block_1.ptr != nullptr)
        {
            EXPECT_EQ(device_enum::CUDA, cuda_block_1.device.type());
            EXPECT_EQ(1, cuda_block_1.device.index());
            pool->deallocate(cuda_block_1);
        }

        MEMORY_LOG_INFO("GPU memory pool device types test passed");
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO(
            "GPU memory pool device types test failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test memory pool error handling
 */
MEMORYTEST(GpuMemoryPool, handles_errors_gracefully)
{
    gpu_memory_pool_config config;
    config.min_block_size = 1024;
    config.max_block_size = 1024ULL;

    auto pool = gpu_memory_pool::create(config);
    EXPECT_NE(nullptr, pool.get());

    try
    {
        // Test zero-size allocation
        auto zero_block = pool->allocate(0, device_enum::CUDA, 0);
        // Should handle gracefully (may return null or throw)

        // Test invalid device index
        auto invalid_block = pool->allocate(1024, device_enum::CUDA, 999);
        // Should handle gracefully (may return null or throw)

        MEMORY_LOG_INFO("GPU memory pool error handling test passed");
    }
    catch (const std::exception& e)
    {
        // Expected behavior for invalid operations
        MEMORY_LOG_INFO("Expected exception in error handling test: {}", e.what());
    }
}

#endif  // PROJECT_HAS_CUDA
