/*
 * Quarisma: High-Performance Computational Library
 * Copyright 2025 Quarisma Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "MemoryTest.h"
#include "gpu/allocator_gpu.h"
#include "gpu/cuda_caching_allocator.h"
#include "helper/memory_allocator.h"

#if PROJECT_HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace memory;
using namespace memory::gpu;

namespace
{

/**
 * @brief Check if CUDA is available and return device count
 */
int get_cuda_device_count()
{
#if PROJECT_HAS_CUDA
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess)
    {
        return 0;
    }
    return device_count;
#else
    return 0;
#endif
}

/**
 * @brief RAII helper for CUDA memory validation
 */
class cuda_memory_validator
{
public:
    explicit cuda_memory_validator(void* ptr, size_t size) : ptr_(ptr), size_(size)
    {
#if PROJECT_HAS_CUDA
        if (ptr_ != nullptr)
        {
            // Verify pointer is valid GPU memory
            cudaPointerAttributes attrs;
            cudaError_t           error = cudaPointerGetAttributes(&attrs, ptr_);
            valid_ = (error == cudaSuccess && attrs.type == cudaMemoryTypeDevice);
        }
#endif
    }

    bool is_valid() const { return valid_; }

    bool test_memory_access() const
    {
#if PROJECT_HAS_CUDA
        if (!valid_ || ptr_ == nullptr || size_ == 0)
        {
            return false;
        }

        // Test memory by setting and reading back a pattern
        std::vector<uint8_t> test_pattern(std::min(size_, size_t(1024)), 0xAB);
        std::vector<uint8_t> read_back(test_pattern.size());

        cudaError_t error =
            cudaMemcpy(ptr_, test_pattern.data(), test_pattern.size(), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            return false;

        error = cudaMemcpy(read_back.data(), ptr_, test_pattern.size(), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
            return false;

        return test_pattern == read_back;
#else
        return false;
#endif
    }

private:
    void*  ptr_;
    size_t size_;
    bool   valid_ = false;
};

}  // anonymous namespace

/**
 * @brief Test basic CUDA allocator functionality
 */
MEMORYTEST(AllocatorCuda, BasicFunctionality)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    // Test GPU allocator with default options
    auto gpu_allocator = create_gpu_allocator(0, "Test-GPU");
    ASSERT_NE(gpu_allocator, nullptr);

    // Test basic allocation
    void* ptr1 = gpu_allocator->allocate_raw(256, 1024);
    EXPECT_NE(ptr1, nullptr);

    cuda_memory_validator validator1(ptr1, 1024);
    EXPECT_TRUE(validator1.is_valid());
    EXPECT_TRUE(validator1.test_memory_access());

    // Test deallocation
    gpu_allocator->deallocate_raw(ptr1);

    // Test GPU allocator with custom options
    allocator_gpu::Options options;
    options.enable_statistics = true;
    auto custom_allocator     = create_gpu_allocator(0, options, "Test-GPU-Custom");
    ASSERT_NE(custom_allocator, nullptr);

    void* ptr2 = custom_allocator->allocate_raw(256, 2048);
    EXPECT_NE(ptr2, nullptr);

    cuda_memory_validator validator2(ptr2, 2048);
    EXPECT_TRUE(validator2.is_valid());
    EXPECT_TRUE(validator2.test_memory_access());

    custom_allocator->deallocate_raw(ptr2);
}

/**
 * @brief Test allocator properties and metadata
 */
MEMORYTEST(AllocatorCuda, AllocatorProperties)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    auto allocator = create_gpu_allocator(0, "Properties-Test");

    // Test name
    EXPECT_EQ(allocator->Name(), "Properties-Test");

    // Test device ID
    EXPECT_EQ(allocator->device_id(), 0);

    // Test allocation method
    auto method = allocator->allocation_method();
    //EXPECT_TRUE(method == gpu_allocation_method::SYNC);

    // Test memory type
    EXPECT_EQ(allocator->GetMemoryType(), allocator_memory_enum::DEVICE);

    // Test size tracking (depends on backend)
    void* ptr = allocator->allocate_raw(64, 1024);
    ASSERT_NE(ptr, nullptr);

    if (allocator->tracks_allocation_sizes())
    {
        EXPECT_EQ(allocator->RequestedSize(ptr), 1024);
        EXPECT_GE(allocator->AllocatedSize(ptr), 1024);
        EXPECT_GT(allocator->AllocationId(ptr), 0);
    }

    allocator->deallocate_raw(ptr);
}

/**
 * @brief Test statistics collection
 */
MEMORYTEST(AllocatorCuda, Statistics)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    auto allocator = create_gpu_allocator(0, "Stats-Test");

    // Get initial stats
    auto initial_stats = allocator->GetStats();
    EXPECT_TRUE(initial_stats.has_value());

    // Perform some allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = allocator->allocate_raw(64, 1024 * (i + 1));
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // Check stats after allocations
    auto after_alloc_stats = allocator->GetStats();
    EXPECT_TRUE(after_alloc_stats.has_value());
    EXPECT_GT(after_alloc_stats->num_allocs, initial_stats->num_allocs);
    EXPECT_GT(after_alloc_stats->bytes_in_use, initial_stats->bytes_in_use);

    // Deallocate all
    for (void* ptr : ptrs)
    {
        allocator->deallocate_raw(ptr);
    }

    // Check stats after deallocation
    auto after_dealloc_stats = allocator->GetStats();
    EXPECT_TRUE(after_dealloc_stats.has_value());
    //EXPECT_EQ(after_dealloc_stats->bytes_in_use, initial_stats->bytes_in_use);

    // Test stats clearing
    EXPECT_TRUE(allocator->ClearStats());
}

/**
 * @brief Test multi-threaded allocation
 */
MEMORYTEST(AllocatorCuda, MultiThreaded)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    auto allocator = create_gpu_allocator(0, "MT-Test");

    const int                num_threads       = 4;
    const int                allocs_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int>         success_count{0};
    std::atomic<int>         failure_count{0};

    auto worker = [&]()
    {
        std::vector<void*>                    local_ptrs;
        std::mt19937                          rng(std::random_device{}());
        std::uniform_int_distribution<size_t> size_dist(64, 4096);

        for (int i = 0; i < allocs_per_thread; ++i)
        {
            size_t alloc_size = size_dist(rng);
            void*  ptr        = allocator->allocate_raw(64, alloc_size);

            if (ptr != nullptr)
            {
                local_ptrs.push_back(ptr);
                success_count.fetch_add(1);
            }
            else
            {
                failure_count.fetch_add(1);
            }
        }

        // Clean up
        for (void* ptr : local_ptrs)
        {
            allocator->deallocate_raw(ptr);
        }
    };

    // Launch threads
    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(worker);
    }

    // Wait for completion
    for (auto& thread : threads)
    {
        thread.join();
    }

    // Verify results
    EXPECT_GT(success_count.load(), 0);
    EXPECT_EQ(success_count.load() + failure_count.load(), num_threads * allocs_per_thread);

    // Most allocations should succeed
    double success_rate =
        static_cast<double>(success_count.load()) / (num_threads * allocs_per_thread);
    EXPECT_GT(success_rate, 0.8);  // At least 80% success rate
}

/**
 * @brief Test large allocation handling
 */
MEMORYTEST(AllocatorCuda, LargeAllocations)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    auto allocator = create_gpu_allocator(0, "Large-Test");

    // Test progressively larger allocations
    std::vector<size_t> sizes = {
        1024 * 1024,       // 1 MB
        16 * 1024 * 1024,  // 16 MB
        64 * 1024 * 1024,  // 64 MB
        128 * 1024 * 1024  // 128 MB
    };

    for (size_t size : sizes)
    {
        void* ptr = allocator->allocate_raw(256, size);
        if (ptr != nullptr)
        {
            cuda_memory_validator validator(ptr, size);
            EXPECT_TRUE(validator.is_valid());

            // Test memory access for smaller sizes only (to avoid long test times)
            if (size <= 16 * 1024 * 1024)
            {
                EXPECT_TRUE(validator.test_memory_access());
            }

            allocator->deallocate_raw(ptr);
        }
        else
        {
            // Large allocation failure is acceptable if device memory is limited
            QUARISMA_LOG_WARNING(
                "Failed to allocate {} bytes - device memory may be limited", size);
        }
    }
}

/**
 * @brief Test error handling and edge cases
 */
MEMORYTEST(AllocatorCuda, ErrorHandling)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    auto allocator = create_gpu_allocator(0, "Error-Test");

    // Test zero-size allocation
    void* ptr_zero = allocator->allocate_raw(64, 0);
    EXPECT_EQ(ptr_zero, nullptr);

    // Test null pointer deallocation
    allocator->deallocate_raw(nullptr);  // Should not crash

    // Test double deallocation (should be handled gracefully)
    void* ptr = allocator->allocate_raw(64, 1024);
    ASSERT_NE(ptr, nullptr);

    allocator->deallocate_raw(ptr);
    allocator->deallocate_raw(ptr);  // Double free - should be handled gracefully

    // Test very large allocation (should fail gracefully)
    //void* huge_ptr = allocator->allocate_raw(64, SIZE_MAX);
    //EXPECT_EQ(huge_ptr, nullptr);
}

/**
 * @brief Compare performance between BFC and Pool strategies
 */
MEMORYTEST(AllocatorCuda, StrategyComparison)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int    num_iterations = 1000;
    const size_t alloc_size     = 1024;

    // Test GPU allocator with default options
    auto gpu_allocator = create_gpu_allocator(0, "GPU-Perf");

    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i)
    {
        void* ptr = gpu_allocator->allocate_raw(64, alloc_size);
        if (ptr != nullptr)
        {
            gpu_allocator->deallocate_raw(ptr);
        }
    }
    auto end_gpu = std::chrono::high_resolution_clock::now();

    // Test GPU allocator with custom options
    allocator_gpu::Options options;
    options.enable_statistics = false;  // Disable stats for better performance
    auto custom_allocator     = create_gpu_allocator(0, options, "GPU-Custom-Perf");

    auto start_custom = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i)
    {
        void* ptr = custom_allocator->allocate_raw(64, alloc_size);
        if (ptr != nullptr)
        {
            custom_allocator->deallocate_raw(ptr);
        }
    }
    auto end_custom = std::chrono::high_resolution_clock::now();

    // Calculate and report timings
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
    auto custom_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end_custom - start_custom);

    std::cout << "GPU Default: " << gpu_duration.count() << " microseconds\n";
    std::cout << "GPU Custom: " << custom_duration.count() << " microseconds\n";

    // Both should complete successfully
    EXPECT_GT(gpu_duration.count(), 0);
    EXPECT_GT(custom_duration.count(), 0);
}

/**
 * @brief Test basic_gpu_allocator Free method directly
 */
MEMORYTEST(AllocatorCuda, basic_gpu_allocator_free_method)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    // Create basic GPU allocator directly
    basic_gpu_allocator sub_allocator(0);

    // Test allocation and deallocation
    size_t bytes_received = 0;
    void*  ptr            = sub_allocator.Alloc(256, 1024, &bytes_received);
    EXPECT_NE(ptr, nullptr);
    EXPECT_GE(bytes_received, 1024);

    // Test Free method directly
    sub_allocator.Free(ptr, bytes_received);

    // Test multiple allocations and frees
    std::vector<void*>  ptrs;
    std::vector<size_t> sizes;

    for (int i = 0; i < 10; ++i)
    {
        size_t size     = (i + 1) * 256;
        size_t received = 0;
        void*  test_ptr = sub_allocator.Alloc(256, size, &received);
        EXPECT_NE(test_ptr, nullptr);
        ptrs.push_back(test_ptr);
        sizes.push_back(received);
    }

    // Free all allocations
    for (size_t i = 0; i < ptrs.size(); ++i)
    {
        sub_allocator.Free(ptrs[i], sizes[i]);
    }
}

/**
 * @brief Test allocate_gpu_memory and deallocate_gpu_memory functions directly
 */
MEMORYTEST(AllocatorCuda, gpu_memory_allocation_functions)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    // Test gpu::memory_allocator::allocate function
    void* ptr1 = memory::gpu::memory_allocator::allocate(1024, 0);
    EXPECT_NE(ptr1, nullptr);

    // Test gpu::memory_allocator::free function
    memory::gpu::memory_allocator::free(ptr1, 1024, 0);

    // Test with different sizes
    std::vector<size_t> test_sizes = {256, 1024, 4096, 16384, 65536};
    std::vector<void*>  ptrs;

    for (size_t size : test_sizes)
    {
        void* ptr = memory::gpu::memory_allocator::allocate(size, 0);
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (size_t i = 0; i < ptrs.size(); ++i)
    {
        memory::gpu::memory_allocator::free(ptrs[i], test_sizes[i], 0);
    }

    // Test error case - invalid device
    void* invalid_ptr = memory::gpu::memory_allocator::allocate(1024, 999);
    EXPECT_EQ(invalid_ptr, nullptr);  // Should fail for invalid device
}

/**
 * @brief Test set_device_context function
 */
MEMORYTEST(AllocatorCuda, set_device_context_function)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    // Test setting device context
    bool result1 = memory::gpu::memory_allocator::set_device(0);
    EXPECT_TRUE(result1);

    // Test with invalid device
    bool result2 = memory::gpu::memory_allocator::set_device(999);
    EXPECT_FALSE(result2);  // Should fail for invalid device

    // Test setting context multiple times
    for (int i = 0; i < device_count && i < 3; ++i)
    {
        bool result = memory::gpu::memory_allocator::set_device(i);
        EXPECT_TRUE(result);
    }
}

/**
 * @brief Test ClearStats method
 */
MEMORYTEST(AllocatorCuda, clear_stats_method)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    auto allocator = create_gpu_allocator(0, "ClearStats-Test");

    // Perform some allocations to generate stats
    void* ptr1 = allocator->allocate_raw(256, 1024);
    void* ptr2 = allocator->allocate_raw(256, 2048);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);

    // Get stats before clearing
    auto stats_before = allocator->GetStats();
    EXPECT_TRUE(stats_before.has_value());

    // Clear stats
    allocator->ClearStats();

    // Get stats after clearing
    auto stats_after = allocator->GetStats();
    EXPECT_TRUE(stats_after.has_value());

    // Clean up
    allocator->deallocate_raw(ptr1);
    allocator->deallocate_raw(ptr2);
}

/**
 * @brief Test device_id method
 */
MEMORYTEST(AllocatorCuda, device_id_method)
{
    int device_count = get_cuda_device_count();
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    // Test device_id for different devices
    for (int i = 0; i < device_count && i < 3; ++i)
    {
        auto allocator = create_gpu_allocator(i, "DeviceID-Test-" + std::to_string(i));
        EXPECT_EQ(allocator->device_id(), i);
    }
}
