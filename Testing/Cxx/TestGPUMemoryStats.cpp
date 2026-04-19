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
#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>
//#include "logger/logger.h"
#include "allocator.h"
#include "common/data_ptr.h"
#include "common/device.h"
#include "cpu/allocator_device.h"
#include "gpu/cuda_caching_allocator.h"
#include "gpu/gpu_allocator_factory.h"
#include "gpu/gpu_allocator_tracking.h"
#include "gpu/gpu_device_manager.h"
#include "gpu/gpu_memory_alignment.h"
#include "gpu/gpu_memory_pool.h"
#include "gpu/gpu_memory_transfer.h"
#include "gpu/gpu_memory_wrapper.h"
#include "gpu/gpu_resource_tracker.h"
#include "profiler/unified_memory_stats.h"

using namespace memory;

namespace memory
{

/**
 * @brief Test suite for GPU memory statistics structures and functionality
 *
 * This test suite validates the unified memory statistics system for GPU allocators,
 * including CUDA-specific timing, resource tracking, and cache performance metrics.
 */
class test_gpu_memory_stats
{
public:
    /**
     * @brief Check if CUDA is available for testing
     */
    static bool is_cuda_available()
    {
        int         device_count = 0;
        cudaError_t error        = cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess || device_count == 0)
        {
            MEMORY_LOG_INFO("CUDA not available or no devices found. Skipping GPU tests.");
            return false;
        }

        // Try to set device 0
        error = cudaSetDevice(0);
        if (error != cudaSuccess)
        {
            MEMORY_LOG_INFO("Cannot set CUDA device 0. Skipping GPU tests.");
            return false;
        }

        return true;
    }

    /**
     * @brief Test GPU timing statistics with CUDA events
     */
    static void test_gpu_timing_stats()
    {
        MEMORY_LOG_INFO("Testing GPU timing statistics with CUDA events...");

        if (!is_cuda_available())
        {
            MEMORY_LOG_INFO("⚠ Skipping GPU timing tests - CUDA not available");
            return;
        }

        atomic_timing_stats gpu_stats;

        // Test initial state
        EXPECT_EQ(gpu_stats.total_allocations.load(), 0);
        EXPECT_EQ(gpu_stats.cuda_sync_time_us.load(), 0);
        EXPECT_EQ(gpu_stats.total_transfer_time_us.load(), 0);

        // Simulate GPU memory operations with timing
        void*  gpu_ptr    = nullptr;
        size_t alloc_size = 1024ULL;  // 1MB

        auto        start_time = std::chrono::steady_clock::now();
        cudaError_t error      = cudaMalloc(&gpu_ptr, alloc_size);
        auto        end_time   = std::chrono::steady_clock::now();

        if (error == cudaSuccess)
        {
            auto duration_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                    .count();

            // Update statistics
            gpu_stats.total_allocations.fetch_add(1);
            gpu_stats.total_alloc_time_us.fetch_add(duration_us);

            // Update min/max timing
            uint64_t current_min = gpu_stats.min_alloc_time_us.load();
            while (duration_us < current_min &&
                   !gpu_stats.min_alloc_time_us.compare_exchange_weak(current_min, duration_us))
            {
            }

            uint64_t current_max = gpu_stats.max_alloc_time_us.load();
            while (duration_us > current_max &&
                   !gpu_stats.max_alloc_time_us.compare_exchange_weak(current_max, duration_us))
            {
            }

            // Test memory transfer timing
            std::vector<float> host_data(alloc_size / sizeof(float), 1.0f);

            start_time = std::chrono::steady_clock::now();
            error      = cudaMemcpy(gpu_ptr, host_data.data(), alloc_size, cudaMemcpyHostToDevice);
            end_time   = std::chrono::steady_clock::now();

            if (error == cudaSuccess)
            {
                auto transfer_time_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                        .count();
                gpu_stats.total_transfer_time_us.fetch_add(transfer_time_us);

                MEMORY_LOG_INFO("  GPU allocation time: {} μs", duration_us);
                MEMORY_LOG_INFO("  Memory transfer time: {} μs", transfer_time_us);
            }

            // Test CUDA synchronization timing
            start_time = std::chrono::steady_clock::now();
            cudaDeviceSynchronize();
            end_time = std::chrono::steady_clock::now();

            auto sync_time_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                    .count();
            gpu_stats.cuda_sync_time_us.fetch_add(sync_time_us);

            // Clean up
            cudaFree(gpu_ptr);
            gpu_stats.total_deallocations.fetch_add(1);
        }

        // Verify statistics were updated
        EXPECT_GT(gpu_stats.total_allocations.load(), 0);

        if (gpu_stats.total_alloc_time_us.load() > 0)
        {
            double avg_time = gpu_stats.average_alloc_time_us();
            EXPECT_GT(avg_time, 0.0);
            MEMORY_LOG_INFO("  Average allocation time: {:.2f} μs", avg_time);
        }

        MEMORY_LOG_INFO("✓ GPU timing statistics test passed");
    }

    /**
     * @brief Test GPU resource statistics tracking
     */
    static void test_gpu_resource_stats()
    {
        MEMORY_LOG_INFO("Testing GPU resource statistics tracking...");

        if (!is_cuda_available())
        {
            MEMORY_LOG_INFO("⚠ Skipping GPU resource tests - CUDA not available");
            return;
        }

        unified_resource_stats gpu_resources;

        // Simulate multiple GPU allocations
        std::vector<void*>  gpu_ptrs;
        std::vector<size_t> alloc_sizes = {1024, 2048, 4096, 8192, 16384};  // Various sizes

        size_t total_allocated = 0;
        for (size_t size : alloc_sizes)
        {
            void*       ptr   = nullptr;
            cudaError_t error = cudaMalloc(&ptr, size);

            if (error == cudaSuccess)
            {
                gpu_ptrs.push_back(ptr);

                // Update statistics
                gpu_resources.num_allocs.fetch_add(1);
                gpu_resources.active_allocations.fetch_add(1);
                gpu_resources.bytes_in_use.fetch_add(size);
                gpu_resources.total_bytes_allocated.fetch_add(size);
                total_allocated += size;

                // Update peak usage
                int64_t current_peak  = gpu_resources.peak_bytes_in_use.load();
                int64_t current_usage = gpu_resources.bytes_in_use.load();
                while (current_usage > current_peak &&
                       !gpu_resources.peak_bytes_in_use.compare_exchange_weak(
                           current_peak, current_usage))
                {
                }

                // Update largest allocation
                int64_t current_largest = gpu_resources.largest_alloc_size.load();
                while (
                    static_cast<int64_t>(size) > current_largest &&
                    !gpu_resources.largest_alloc_size.compare_exchange_weak(current_largest, size))
                {
                }
            }
        }

        // Verify resource tracking
        EXPECT_EQ(gpu_resources.num_allocs.load(), gpu_ptrs.size());
        EXPECT_EQ(gpu_resources.active_allocations.load(), gpu_ptrs.size());
        EXPECT_EQ(gpu_resources.bytes_in_use.load(), static_cast<int64_t>(total_allocated));
        EXPECT_GE(gpu_resources.peak_bytes_in_use.load(), static_cast<int64_t>(total_allocated));
        EXPECT_EQ(gpu_resources.largest_alloc_size.load(), 16384);  // Largest size

        // Test derived metrics
        double avg_size     = gpu_resources.average_allocation_size();
        double expected_avg = static_cast<double>(total_allocated) / gpu_ptrs.size();
        EXPECT_LT(std::abs(avg_size - expected_avg), 1.0);

        double efficiency = gpu_resources.memory_efficiency();
        EXPECT_GE(efficiency, 0.0);
        EXPECT_LE(efficiency, 1.0);

        MEMORY_LOG_INFO("  Total allocated: {} bytes", total_allocated);
        MEMORY_LOG_INFO("  Active allocations: {}", gpu_resources.active_allocations.load());
        MEMORY_LOG_INFO("  Average allocation size: {:.1f} bytes", avg_size);
        MEMORY_LOG_INFO("  Memory efficiency: {:.1f}%", (efficiency * 100.0));

        // Clean up allocations
        for (void* ptr : gpu_ptrs)
        {
            cudaFree(ptr);
            gpu_resources.num_deallocs.fetch_add(1);
            gpu_resources.active_allocations.fetch_sub(1);
        }

        // Test debug string
        std::string debug_str = gpu_resources.debug_string();
        EXPECT_FALSE(debug_str.empty());
        MEMORY_LOG_INFO("  Debug string: {}", debug_str);

        MEMORY_LOG_INFO("✓ GPU resource statistics test passed");
    }

    /**
     * @brief Test CUDA caching allocator statistics
     */
    static void test_cuda_caching_stats()
    {
        MEMORY_LOG_INFO("Testing CUDA caching allocator statistics...");

        if (!is_cuda_available())
        {
            MEMORY_LOG_INFO("⚠ Skipping CUDA caching tests - CUDA not available");
            return;
        }

        unified_cache_stats cache_stats;

        // Simulate cache behavior
        cache_stats.cache_hits.store(150);
        cache_stats.cache_misses.store(50);
        cache_stats.cache_evictions.store(10);
        cache_stats.bytes_cached.store(2 * 1024ULL);       // 2MB cached
        cache_stats.peak_bytes_cached.store(3 * 1024ULL);  // 3MB peak
        cache_stats.cache_blocks.store(25);
        cache_stats.driver_allocations.store(60);  // Actual cudaMalloc calls
        cache_stats.driver_frees.store(50);        // Actual cudaFree calls

        // Test cache efficiency metrics
        double hit_rate = cache_stats.cache_hit_rate();
        EXPECT_NEAR(hit_rate, 0.75, 0.01);  // 150 / (150 + 50) = 0.75

        double efficiency_percent = cache_stats.cache_efficiency_percent();
        EXPECT_NEAR(efficiency_percent, 75.0, 0.1);  // 75%

        // Test driver call reduction
        double reduction          = cache_stats.driver_call_reduction();
        double expected_reduction = 200.0 / 110.0;  // (150+50) / (60+50)
        EXPECT_NEAR(reduction, expected_reduction, 0.01);

        MEMORY_LOG_INFO("  Cache hit rate: {:.1f}%", (hit_rate * 100.0));
        MEMORY_LOG_INFO("  Driver call reduction: {:.2f}x", reduction);
        MEMORY_LOG_INFO("  Bytes cached: {} KB", (cache_stats.bytes_cached.load() / 1024));
        MEMORY_LOG_INFO("  Cache blocks: {}", cache_stats.cache_blocks.load());

        // Test reset functionality
        cache_stats.reset();
        EXPECT_EQ(cache_stats.cache_hits.load(), 0);
        EXPECT_EQ(cache_stats.cache_misses.load(), 0);
        EXPECT_EQ(cache_stats.bytes_cached.load(), 0);
        EXPECT_EQ(cache_stats.cache_hit_rate(), 0.0);

        MEMORY_LOG_INFO("✓ CUDA caching statistics test passed");
    }

    /**
     * @brief Test comprehensive GPU memory statistics integration
     */
    static void test_comprehensive_gpu_stats()
    {
        MEMORY_LOG_INFO("Testing comprehensive GPU memory statistics...");

        if (!is_cuda_available())
        {
            MEMORY_LOG_INFO("⚠ Skipping comprehensive GPU tests - CUDA not available");
            return;
        }

        comprehensive_memory_stats gpu_stats("CUDA_Allocator");

        // Populate with realistic GPU statistics
        gpu_stats.resource_stats.num_allocs.store(500);
        gpu_stats.resource_stats.num_deallocs.store(450);
        gpu_stats.resource_stats.active_allocations.store(50);
        gpu_stats.resource_stats.bytes_in_use.store(64 * 1024ULL);        // 64MB
        gpu_stats.resource_stats.peak_bytes_in_use.store(128 * 1024ULL);  // 128MB

        gpu_stats.timing_stats.total_allocations.store(500);
        gpu_stats.timing_stats.total_alloc_time_us.store(250000);     // 250ms total
        gpu_stats.timing_stats.total_transfer_time_us.store(100000);  // 100ms transfers
        gpu_stats.timing_stats.cuda_sync_time_us.store(50000);        // 50ms sync

        gpu_stats.cache_stats.cache_hits.store(400);
        gpu_stats.cache_stats.cache_misses.store(100);
        gpu_stats.cache_stats.bytes_cached.store(32 * 1024ULL);  // 32MB cached

        // Test overall efficiency
        double efficiency = gpu_stats.overall_efficiency();
        EXPECT_GE(efficiency, 0.0);
        EXPECT_LE(efficiency, 1.0);

        // Test operations per second
        double ops_per_sec = gpu_stats.operations_per_second();
        EXPECT_GT(ops_per_sec, 0.0);

        // Generate and validate report
        std::string report = gpu_stats.generate_report();
        EXPECT_FALSE(report.empty());
        EXPECT_NE(report.find("CUDA_Allocator"), std::string::npos);
        EXPECT_NE(report.find("CUDA Sync Time"), std::string::npos);
        EXPECT_NE(report.find("Cache Performance"), std::string::npos);

        MEMORY_LOG_INFO("  Overall efficiency: {}%", (efficiency * 100.0));
        MEMORY_LOG_INFO("  GPU operations/sec: {}", ops_per_sec);

        // Test that report contains GPU-specific sections
        EXPECT_NE(report.find("Transfer Time"), std::string::npos);

        MEMORY_LOG_INFO("✓ Comprehensive GPU statistics test passed");
    }

    /**
     * @brief Test thread safety of GPU statistics under concurrent access
     */
    static void test_gpu_stats_thread_safety()
    {
        MEMORY_LOG_INFO("Testing GPU statistics thread safety...");

        if (!is_cuda_available())
        {
            MEMORY_LOG_INFO("⚠ Skipping GPU thread safety tests - CUDA not available");
            return;
        }

        atomic_timing_stats    shared_stats;
        unified_resource_stats shared_resources;

        const int num_threads           = 4;
        const int operations_per_thread = 100;

        std::vector<std::thread> threads;

        // Launch threads that concurrently update statistics
        for (int t = 0; t < num_threads; ++t)
        {
            threads.emplace_back(
                [&shared_stats, &shared_resources, operations_per_thread]()
                {
                    for (int i = 0; i < operations_per_thread; ++i)
                    {
                        // Update timing stats
                        shared_stats.total_allocations.fetch_add(1);
                        shared_stats.total_alloc_time_us.fetch_add(100 + i);

                        // Update resource stats
                        shared_resources.num_allocs.fetch_add(1);
                        shared_resources.bytes_in_use.fetch_add(1024);

                        // Simulate some work
                        std::this_thread::sleep_for(std::chrono::microseconds(1));

                        shared_resources.bytes_in_use.fetch_sub(1024);
                        shared_resources.num_deallocs.fetch_add(1);
                    }
                });
        }

        // Wait for all threads to complete
        for (auto& thread : threads)
        {
            thread.join();
        }

        // Verify final counts
        int expected_total = num_threads * operations_per_thread;
        EXPECT_EQ(shared_stats.total_allocations.load(), expected_total);
        EXPECT_EQ(shared_resources.num_allocs.load(), expected_total);
        EXPECT_EQ(shared_resources.num_deallocs.load(), expected_total);
        EXPECT_EQ(shared_resources.bytes_in_use.load(), 0);  // Should be balanced

        MEMORY_LOG_INFO("  Concurrent operations completed: {}", expected_total);
        MEMORY_LOG_INFO("  Final timing total: {} μs", shared_stats.total_alloc_time_us.load());

        MEMORY_LOG_INFO("✓ GPU statistics thread safety test passed");
    }
};

}  // namespace memory

// Test execution functions
void TestGPUTimingStats()
{
    test_gpu_memory_stats::test_gpu_timing_stats();
}

void TestGPUResourceStats()
{
    test_gpu_memory_stats::test_gpu_resource_stats();
}

void TestCUDACachingStats()
{
    test_gpu_memory_stats::test_cuda_caching_stats();
}

void TestComprehensiveGPUStats()
{
    test_gpu_memory_stats::test_comprehensive_gpu_stats();
}

void TestGPUStatsThreadSafety()
{
    test_gpu_memory_stats::test_gpu_stats_thread_safety();
}

#else  // !MEMORY_HAS_CUDA

// Stub implementations when CUDA is not available
void TestGPUTimingStats()
{
    MEMORY_LOG_INFO("⚠ CUDA not enabled - GPU timing stats tests skipped");
}

void TestGPUResourceStats()
{
    MEMORY_LOG_INFO("⚠ CUDA not enabled - GPU resource stats tests skipped");
}

void TestCUDACachingStats()
{
    MEMORY_LOG_INFO("⚠ CUDA not enabled - CUDA caching stats tests skipped");
}

void TestComprehensiveGPUStats()
{
    MEMORY_LOG_INFO("⚠ CUDA not enabled - comprehensive GPU stats tests skipped");
}

void TestGPUStatsThreadSafety()
{
    MEMORY_LOG_INFO("⚠ CUDA not enabled - GPU thread safety tests skipped");
}

#endif  // MEMORY_HAS_CUDA

// Main test function expected by the test framework
MEMORYTEST(TestGPUMemoryStats, test)
{
    MEMORY_LOG_INFO("Starting GPU Memory Statistics Tests...");

    TestGPUTimingStats();
    TestGPUResourceStats();
    TestCUDACachingStats();
    TestComprehensiveGPUStats();
    TestGPUStatsThreadSafety();

    MEMORY_LOG_INFO("All GPU Memory Statistics Tests completed successfully!");
    END_TEST();
}
