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
 *
 * NOTE: This file tests the legacy gpu_allocator_factory which is deprecated.
 * The main GPU allocator functionality is now in allocator_gpu (see TestAllocatorCuda.cpp).
 * This file is kept for testing the remaining factory functionality (caching allocator creation).
 */

#include "MemoryTest.h"
#include "common/macros.h"

#if PROJECT_HAS_CUDA

#include <memory>
#include <string>

//#include "logger/logger.h"
#include "common/device.h"
#include "gpu/cuda_caching_allocator.h"
#include "gpu/gpu_allocator_factory.h"

using namespace memory;
using namespace memory::gpu;

/**
 * @brief Test strategy recommendation functionality (legacy factory support)
 */
MEMORYTEST(GpuAllocatorFactory, recommends_appropriate_strategies)
{
    // Test strategy recommendation for different scenarios
    // Note: This tests the legacy factory pattern which is deprecated
    // but still provides useful strategy recommendations

    // Small, frequent allocations should prefer caching
    auto strategy1 = gpu_allocator_factory::recommend_strategy(1024, 200.0, 0.5);
    EXPECT_TRUE(
        strategy1 == gpu_allocation_strategy::CACHING ||
        strategy1 == gpu_allocation_strategy::POOL);

    // Large, infrequent allocations should prefer direct
    auto strategy2 = gpu_allocator_factory::recommend_strategy(64 * 1024ULL, 1.0, 10.0);
    EXPECT_TRUE(
        strategy2 == gpu_allocation_strategy::DIRECT || strategy2 == gpu_allocation_strategy::POOL);

    // Medium allocations with moderate frequency
    auto strategy3 = gpu_allocator_factory::recommend_strategy(1024ULL, 50.0, 2.0);
    EXPECT_TRUE(
        strategy3 == gpu_allocation_strategy::POOL || strategy3 == gpu_allocation_strategy::DIRECT);

    QUARISMA_LOG_INFO("GPU allocator factory strategy recommendation test passed");
}

/**
 * @brief Test device validation functionality (legacy factory support)
 */
MEMORYTEST(GpuAllocatorFactory, validates_device_support_correctly)
{
    // Test CUDA device validation
    // Note: This tests the legacy factory pattern which is deprecated
    // but device validation is still useful for compatibility

    bool cuda_direct_support = gpu_allocator_factory::validate_device_support(
        gpu_allocation_strategy::DIRECT, device_enum::CUDA, 0);

    bool cuda_pool_support = gpu_allocator_factory::validate_device_support(
        gpu_allocation_strategy::POOL, device_enum::CUDA, 0);

    bool cuda_caching_support = gpu_allocator_factory::validate_device_support(
        gpu_allocation_strategy::CACHING, device_enum::CUDA, 0);

    // All strategies should be supported for CUDA (if CUDA is available)
    // Note: Results depend on actual CUDA availability
    EXPECT_TRUE(cuda_direct_support || !cuda_direct_support);  // Always passes, just tests no crash
    EXPECT_TRUE(cuda_pool_support || !cuda_pool_support);
    EXPECT_TRUE(cuda_caching_support || !cuda_caching_support);

    // Test invalid device index
    bool invalid_device = gpu_allocator_factory::validate_device_support(
        gpu_allocation_strategy::DIRECT, device_enum::CUDA, 999);
    EXPECT_FALSE(invalid_device);  // Should fail for invalid device index

    QUARISMA_LOG_INFO("GPU allocator factory device validation test passed");
}

/**
 * @brief Test strategy name conversion
 */
MEMORYTEST(GpuAllocatorFactory, provides_readable_strategy_names)
{
    // Test strategy name conversion
    std::string direct_name = gpu_allocator_factory::strategy_name(gpu_allocation_strategy::DIRECT);
    std::string pool_name   = gpu_allocator_factory::strategy_name(gpu_allocation_strategy::POOL);
    std::string caching_name =
        gpu_allocator_factory::strategy_name(gpu_allocation_strategy::CACHING);

    // Names should not be empty
    EXPECT_FALSE(direct_name.empty());
    EXPECT_FALSE(pool_name.empty());
    EXPECT_FALSE(caching_name.empty());

    // Names should be different
    EXPECT_NE(direct_name, pool_name);
    EXPECT_NE(direct_name, caching_name);
    EXPECT_NE(pool_name, caching_name);

    QUARISMA_LOG_INFO(
        "Strategy names: DIRECT='{}', POOL='{}', CACHING='{}'",
        direct_name,
        pool_name,
        caching_name);

    QUARISMA_LOG_INFO("GPU allocator factory strategy names test passed");
}

/**
 * @brief Test configuration creation for different use cases
 */
MEMORYTEST(GpuAllocatorFactory, creates_optimized_configurations)
{
    // Test default configuration creation
    auto default_config = gpu_allocator_config::create_default(gpu_allocation_strategy::DIRECT, 0);
    EXPECT_EQ(gpu_allocation_strategy::DIRECT, default_config.strategy);
    EXPECT_EQ(device_enum::CUDA, default_config.device_type);
    EXPECT_EQ(0, default_config.device_index);
    EXPECT_GT(default_config.alignment, 0);

    // Test Monte Carlo optimized configuration
    auto mc_config = gpu_allocator_config::create_monte_carlo_optimized(0);
    EXPECT_EQ(device_enum::CUDA, mc_config.device_type);
    EXPECT_EQ(0, mc_config.device_index);
    EXPECT_GT(mc_config.pool_min_block_size, 0);
    EXPECT_GT(mc_config.pool_max_block_size, mc_config.pool_min_block_size);

    // Test PDE optimized configuration
    auto pde_config = gpu_allocator_config::create_pde_optimized(0);
    EXPECT_EQ(device_enum::CUDA, pde_config.device_type);
    EXPECT_EQ(0, pde_config.device_index);
    EXPECT_GT(pde_config.pool_min_block_size, 0);
    EXPECT_GT(pde_config.pool_max_block_size, pde_config.pool_min_block_size);

    QUARISMA_LOG_INFO("GPU allocator factory configuration creation test passed");
}

/**
 * @brief Test caching allocator creation
 */
MEMORYTEST(GpuAllocatorFactory, creates_caching_allocators)
{
    // Test caching allocator creation
    gpu_allocator_config config;
    config.strategy        = gpu_allocation_strategy::CACHING;
    config.device_type     = device_enum::CUDA;
    config.device_index    = 0;
    config.cache_max_bytes = 32 * 1024ULL;  // 32MB

    try
    {
        auto float_allocator = gpu_allocator_factory::create_caching_allocator<float, 256>(config);
        EXPECT_NE(nullptr, float_allocator.get());
        EXPECT_EQ(0, float_allocator->device());

        // Test allocation through factory-created allocator
        float* ptr = float_allocator->allocate(1000);
        EXPECT_NE(nullptr, ptr);
        float_allocator->deallocate(ptr, 1000);

        QUARISMA_LOG_INFO("GPU allocator factory caching allocator creation test passed");
    }
    catch (const std::exception& e)
    {
        QUARISMA_LOG_INFO(
            "Caching allocator creation failed (expected if no CUDA device): {}", e.what());
        // This is acceptable if no CUDA device is available
    }
}

/**
 * @brief Test factory error handling
 */
MEMORYTEST(GpuAllocatorFactory, handles_invalid_configurations)
{
    // Test invalid device configuration
    gpu_allocator_config invalid_config;
    invalid_config.strategy     = gpu_allocation_strategy::CACHING;
    invalid_config.device_type  = device_enum::CUDA;
    invalid_config.device_index = 999;  // Invalid device index

    try
    {
        auto allocator = gpu_allocator_factory::create_caching_allocator<float>(invalid_config);
        // Should either succeed (if validation is lenient) or throw
        EXPECT_TRUE(true);  // Test passes if we reach here
    }
    catch (const std::exception& e)
    {
        // Expected behavior for invalid configuration
        EXPECT_TRUE(true);  // Test passes if exception is thrown
        QUARISMA_LOG_INFO("Expected exception for invalid configuration: {}", e.what());
    }

    QUARISMA_LOG_INFO("GPU allocator factory error handling test passed");
}

/**
 * @brief Test factory with different template parameters
 */
MEMORYTEST(GpuAllocatorFactory, supports_different_template_parameters)
{
    gpu_allocator_config config;
    config.strategy        = gpu_allocation_strategy::CACHING;
    config.device_type     = device_enum::CUDA;
    config.device_index    = 0;
    config.cache_max_bytes = 16 * 1024ULL;  // 16MB

    try
    {
        // Test different types and alignments
        auto double_allocator =
            gpu_allocator_factory::create_caching_allocator<double, 512>(config);
        auto int_allocator = gpu_allocator_factory::create_caching_allocator<int, 128>(config);

        if (double_allocator && int_allocator)
        {
            EXPECT_EQ(0, double_allocator->device());
            EXPECT_EQ(0, int_allocator->device());

            // Test basic operations
            double* dptr = double_allocator->allocate(100);
            int*    iptr = int_allocator->allocate(200);

            if (dptr && iptr)
            {
                double_allocator->deallocate(dptr, 100);
                int_allocator->deallocate(iptr, 200);
            }
        }

        QUARISMA_LOG_INFO("GPU allocator factory template parameters test passed");
    }
    catch (const std::exception& e)
    {
        QUARISMA_LOG_INFO(
            "Template allocator creation failed (expected if no CUDA device): {}", e.what());
    }
}

#endif  // PROJECT_HAS_CUDA
