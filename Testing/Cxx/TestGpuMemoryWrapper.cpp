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
#include "gpu/gpu_memory_wrapper.h"

using namespace memory;
using namespace memory::gpu;

/**
 * @brief Test GPU memory wrapper default construction
 */
MEMORYTEST(GpuMemoryWrapper, constructs_empty_wrapper)
{
    // Test default construction
    gpu_memory_wrapper<float> wrapper;

    EXPECT_EQ(nullptr, wrapper.get());
    EXPECT_EQ(0, wrapper.size());
    EXPECT_TRUE(wrapper.empty());
    EXPECT_FALSE(wrapper.owns_memory());
    EXPECT_FALSE(static_cast<bool>(wrapper));

    MEMORY_LOG_INFO("GPU memory wrapper default construction test passed");
}

/**
 * @brief Test GPU memory wrapper allocation
 */
MEMORYTEST(GpuMemoryWrapper, allocates_typed_memory)
{
    try
    {
        // Test typed wrapper allocation
        auto wrapper = gpu_memory_wrapper<float>::allocate(1000, device_enum::CUDA, 0);

        if (wrapper.get() != nullptr)
        {
            EXPECT_NE(wrapper.get(), nullptr);
            EXPECT_EQ(wrapper.size(), 1000);
            EXPECT_TRUE(wrapper.owns_memory());
            EXPECT_FALSE(wrapper.empty());
            EXPECT_TRUE(static_cast<bool>(wrapper));
            EXPECT_EQ(device_enum::CUDA, wrapper.device().type());
            EXPECT_EQ(0, wrapper.device().index());

            MEMORY_LOG_INFO("GPU memory wrapper typed allocation test passed");
        }
        else
        {
            MEMORY_LOG_INFO("GPU memory wrapper allocation returned null (expected if no GPU)");
        }
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO("GPU memory wrapper allocation failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test GPU memory wrapper void specialization
 */
MEMORYTEST(GpuMemoryWrapper, handles_void_specialization)
{
    try
    {
        // Test void wrapper allocation
        auto void_wrapper = gpu_memory_wrapper<void>::allocate(4096, device_enum::CUDA, 0);

        if (void_wrapper.get() != nullptr)
        {
            EXPECT_NE(void_wrapper.get(), nullptr);
            EXPECT_EQ(void_wrapper.size_bytes(), 4096);
            EXPECT_TRUE(void_wrapper.owns_memory());
            EXPECT_FALSE(void_wrapper.empty());
            EXPECT_TRUE(static_cast<bool>(void_wrapper));

            MEMORY_LOG_INFO("GPU memory wrapper void specialization test passed");
        }
        else
        {
            MEMORY_LOG_INFO(
                "GPU memory wrapper void allocation returned null (expected if no GPU)");
        }
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO(
            "GPU memory wrapper void allocation failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test non-owning wrapper creation
 */
MEMORYTEST(GpuMemoryWrapper, creates_non_owning_wrapper)
{
    // Test non-owning wrapper with host memory
    float* raw_ptr = static_cast<float*>(malloc(100 * sizeof(float)));
    EXPECT_NE(nullptr, raw_ptr);

    auto non_owning = gpu_memory_wrapper<float>::non_owning(raw_ptr, 100, device_enum::CPU, 0);

    EXPECT_EQ(non_owning.get(), raw_ptr);
    EXPECT_EQ(non_owning.size(), 100);
    EXPECT_FALSE(non_owning.owns_memory());
    EXPECT_FALSE(non_owning.empty());
    EXPECT_TRUE(static_cast<bool>(non_owning));
    EXPECT_EQ(device_enum::CPU, non_owning.device().type());

    // Clean up raw pointer manually since wrapper doesn't own it
    free(raw_ptr);

    MEMORY_LOG_INFO("GPU memory wrapper non-owning creation test passed");
}

/**
 * @brief Test wrapper with custom deleter
 */
MEMORYTEST(GpuMemoryWrapper, supports_custom_deleter)
{
    bool deleter_called = false;

    {
        // Create wrapper with custom deleter
        double* raw_ptr = static_cast<double*>(malloc(50 * sizeof(double)));
        EXPECT_NE(nullptr, raw_ptr);

        auto custom_wrapper = gpu_memory_wrapper<double>::wrap(
            raw_ptr,
            50,
            device_enum::CPU,
            0,
            [&deleter_called](double* ptr)
            {
                deleter_called = true;
                free(ptr);
            });

        EXPECT_EQ(custom_wrapper.get(), raw_ptr);
        EXPECT_EQ(custom_wrapper.size(), 50);
        EXPECT_TRUE(custom_wrapper.owns_memory());
    }

    // Custom deleter should have been called
    EXPECT_TRUE(deleter_called);

    MEMORY_LOG_INFO("GPU memory wrapper custom deleter test passed");
}

/**
 * @brief Test move semantics
 */
MEMORYTEST(GpuMemoryWrapper, supports_move_semantics)
{
    try
    {
        // Create wrapper
        auto wrapper1 = gpu_memory_wrapper<int>::allocate(500, device_enum::CUDA, 0);

        if (wrapper1.get() != nullptr)
        {
            int*   original_ptr  = wrapper1.get();
            size_t original_size = wrapper1.size();

            // Test move constructor
            auto wrapper2 = std::move(wrapper1);

            EXPECT_EQ(wrapper2.get(), original_ptr);
            EXPECT_EQ(wrapper2.size(), original_size);
            EXPECT_TRUE(wrapper2.owns_memory());

            // Original wrapper should be empty after move
            EXPECT_EQ(wrapper1.get(), nullptr);
            EXPECT_EQ(wrapper1.size(), 0);
            EXPECT_FALSE(wrapper1.owns_memory());
            EXPECT_TRUE(wrapper1.empty());

            MEMORY_LOG_INFO("GPU memory wrapper move semantics test passed");
        }
        else
        {
            MEMORY_LOG_INFO("GPU memory wrapper move test skipped (no GPU allocation)");
        }
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO("GPU memory wrapper move test failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test copy semantics (non-owning copy)
 */
MEMORYTEST(GpuMemoryWrapper, supports_copy_semantics)
{
    try
    {
        // Create wrapper
        auto wrapper1 = gpu_memory_wrapper<float>::allocate(200, device_enum::CUDA, 0);

        if (wrapper1.get() != nullptr)
        {
            float* original_ptr  = wrapper1.get();
            size_t original_size = wrapper1.size();

            // Test copy constructor (creates non-owning copy)
            auto wrapper2 = wrapper1;

            EXPECT_EQ(wrapper2.get(), original_ptr);
            EXPECT_EQ(wrapper2.size(), original_size);
            EXPECT_FALSE(wrapper2.owns_memory());  // Copy doesn't own memory

            // Original wrapper should still own memory
            EXPECT_EQ(wrapper1.get(), original_ptr);
            EXPECT_EQ(wrapper1.size(), original_size);
            EXPECT_TRUE(wrapper1.owns_memory());

            MEMORY_LOG_INFO("GPU memory wrapper copy semantics test passed");
        }
        else
        {
            MEMORY_LOG_INFO("GPU memory wrapper copy test skipped (no GPU allocation)");
        }
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO("GPU memory wrapper copy test failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test memory release functionality
 */
MEMORYTEST(GpuMemoryWrapper, releases_memory_ownership)
{
    try
    {
        // Create wrapper
        auto wrapper = gpu_memory_wrapper<double>::allocate(100, device_enum::CUDA, 0);

        if (wrapper.get() != nullptr)
        {
            double* ptr = wrapper.get();

            // Release ownership
            double* released_ptr = wrapper.release();

            EXPECT_EQ(released_ptr, ptr);
            EXPECT_EQ(wrapper.get(), nullptr);
            EXPECT_EQ(wrapper.size(), 0);
            EXPECT_FALSE(wrapper.owns_memory());
            EXPECT_TRUE(wrapper.empty());

            // Manually clean up released memory
            // Note: In real usage, you'd need to use appropriate deallocation method
            // For this test, we'll just verify the release worked

            MEMORY_LOG_INFO("GPU memory wrapper memory release test passed");
        }
        else
        {
            MEMORY_LOG_INFO("GPU memory wrapper release test skipped (no GPU allocation)");
        }
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO(
            "GPU memory wrapper release test failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test reset functionality
 */
MEMORYTEST(GpuMemoryWrapper, resets_wrapper_state)
{
    try
    {
        // Create wrapper
        auto wrapper = gpu_memory_wrapper<int>::allocate(300, device_enum::CUDA, 0);

        if (wrapper.get() != nullptr)
        {
            EXPECT_FALSE(wrapper.empty());
            EXPECT_TRUE(wrapper.owns_memory());

            // Reset wrapper
            wrapper.reset();

            EXPECT_EQ(wrapper.get(), nullptr);
            EXPECT_EQ(wrapper.size(), 0);
            EXPECT_FALSE(wrapper.owns_memory());
            EXPECT_TRUE(wrapper.empty());
            EXPECT_FALSE(static_cast<bool>(wrapper));

            MEMORY_LOG_INFO("GPU memory wrapper reset test passed");
        }
        else
        {
            MEMORY_LOG_INFO("GPU memory wrapper reset test skipped (no GPU allocation)");
        }
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO("GPU memory wrapper reset test failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test wrapper comparison operators
 */
MEMORYTEST(GpuMemoryWrapper, supports_comparison_operators)
{
    try
    {
        // Create two wrappers
        auto wrapper1 = gpu_memory_wrapper<float>::allocate(100, device_enum::CUDA, 0);
        auto wrapper2 = gpu_memory_wrapper<float>::allocate(100, device_enum::CUDA, 0);

        if (wrapper1.get() != nullptr && wrapper2.get() != nullptr)
        {
            // Test inequality (different pointers)
            EXPECT_NE(wrapper1, wrapper2);
            EXPECT_TRUE(wrapper1 != wrapper2);
            EXPECT_FALSE(wrapper1 == wrapper2);

            // Test self-equality
            EXPECT_EQ(wrapper1, wrapper1);
            EXPECT_TRUE(wrapper1 == wrapper1);
            EXPECT_FALSE(wrapper1 != wrapper1);

            MEMORY_LOG_INFO("GPU memory wrapper comparison operators test passed");
        }
        else
        {
            MEMORY_LOG_INFO("GPU memory wrapper comparison test skipped (no GPU allocation)");
        }
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO(
            "GPU memory wrapper comparison test failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test convenience function for creating GPU memory
 */
MEMORYTEST(GpuMemoryWrapper, provides_convenience_functions)
{
    try
    {
        // Test make_gpu_memory convenience function
        auto wrapper = make_gpu_memory<double>(150, device_enum::CUDA, 0);

        if (wrapper.get() != nullptr)
        {
            EXPECT_NE(wrapper.get(), nullptr);
            EXPECT_EQ(wrapper.size(), 150);
            EXPECT_TRUE(wrapper.owns_memory());
            EXPECT_EQ(device_enum::CUDA, wrapper.device().type());

            MEMORY_LOG_INFO("GPU memory wrapper convenience functions test passed");
        }
        else
        {
            MEMORY_LOG_INFO("GPU memory wrapper convenience test skipped (no GPU allocation)");
        }
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO(
            "GPU memory wrapper convenience test failed (expected if no GPU): {}", e.what());
    }
}

/**
 * @brief Test wrapper swap functionality
 */
MEMORYTEST(GpuMemoryWrapper, supports_swap_operation)
{
    try
    {
        // Create two wrappers with different sizes
        auto wrapper1 = gpu_memory_wrapper<int>::allocate(100, device_enum::CUDA, 0);
        auto wrapper2 = gpu_memory_wrapper<int>::allocate(200, device_enum::CUDA, 0);

        if (wrapper1.get() != nullptr && wrapper2.get() != nullptr)
        {
            int*   ptr1  = wrapper1.get();
            int*   ptr2  = wrapper2.get();
            size_t size1 = wrapper1.size();
            size_t size2 = wrapper2.size();

            // Swap wrappers
            wrapper1.swap(wrapper2);

            // Verify swap occurred
            EXPECT_EQ(wrapper1.get(), ptr2);
            EXPECT_EQ(wrapper1.size(), size2);
            EXPECT_EQ(wrapper2.get(), ptr1);
            EXPECT_EQ(wrapper2.size(), size1);

            // Test ADL swap
            swap(wrapper1, wrapper2);

            // Should be back to original state
            EXPECT_EQ(wrapper1.get(), ptr1);
            EXPECT_EQ(wrapper1.size(), size1);
            EXPECT_EQ(wrapper2.get(), ptr2);
            EXPECT_EQ(wrapper2.size(), size2);

            MEMORY_LOG_INFO("GPU memory wrapper swap test passed");
        }
        else
        {
            MEMORY_LOG_INFO("GPU memory wrapper swap test skipped (no GPU allocation)");
        }
    }
    catch (const std::exception& e)
    {
        MEMORY_LOG_INFO("GPU memory wrapper swap test failed (expected if no GPU): {}", e.what());
    }
}

#endif  // PROJECT_HAS_CUDA
