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

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "logger.h"
#include "common/device.h"
#include "gpu/gpu_allocator_tracking.h"
#include "gpu/gpu_device_manager.h"

using namespace quarisma;
using namespace quarisma::gpu;

/**
 * @brief Test basic GPU allocator tracking construction
 */
QUARISMATEST(GpuAllocatorTracking, constructs_with_valid_parameters)
{
    // Check if CUDA is available
    auto& device_manager = gpu_device_manager::instance();
    device_manager.initialize();
    auto runtime_info = device_manager.get_runtime_info();

    if (!runtime_info.cuda_available)
    {
        QUARISMA_LOG_INFO("CUDA not available, skipping GPU allocator tracking construction test");
        return;
    }

    // Test basic construction
    auto gpu_tracker = std::make_unique<gpu_allocator_tracking>(device_enum::CUDA, 0, true, false);
    EXPECT_NE(nullptr, gpu_tracker.get());

    // Test device info retrieval
    auto device_info = gpu_tracker->GetDeviceInfo();
    EXPECT_EQ(device_enum::CUDA, device_info.device_type);
    EXPECT_EQ(0, device_info.device_index);

    QUARISMA_LOG_INFO("GPU allocator tracking construction test passed");
}

/**
 * @brief Test basic allocation and deallocation tracking
 */
QUARISMATEST(GpuAllocatorTracking, tracks_allocations_and_deallocations)
{
    auto& device_manager = gpu_device_manager::instance();
    device_manager.initialize();
    auto runtime_info = device_manager.get_runtime_info();

    if (!runtime_info.cuda_available)
    {
        QUARISMA_LOG_INFO("CUDA not available, skipping allocation tracking test");
        return;
    }

    auto gpu_tracker = std::make_unique<gpu_allocator_tracking>(device_enum::CUDA, 0, true, false);

    // Test basic GPU allocation
    void* gpu_ptr = gpu_tracker->allocate_raw(1024, 256);
    if (gpu_ptr != nullptr)
    {
        // Test memory usage tracking
        auto [device_mem, unified_mem, pinned_mem] = gpu_tracker->GetGPUMemoryUsage();
        EXPECT_GT(device_mem, 0);

        // Test deallocation
        gpu_tracker->deallocate_raw(gpu_ptr, 1024);

        QUARISMA_LOG_INFO("GPU allocation tracking test passed");
    }
    else
    {
        QUARISMA_LOG_INFO("GPU allocation failed (expected if insufficient GPU memory)");
    }
}

/**
 * @brief Test typed allocation tracking
 */
QUARISMATEST(GpuAllocatorTracking, tracks_typed_allocations)
{
    auto& device_manager = gpu_device_manager::instance();
    device_manager.initialize();
    auto runtime_info = device_manager.get_runtime_info();

    if (!runtime_info.cuda_available)
    {
        QUARISMA_LOG_INFO("CUDA not available, skipping typed allocation tracking test");
        return;
    }

    auto gpu_tracker = std::make_unique<gpu_allocator_tracking>(device_enum::CUDA, 0, true, false);

    // Test typed allocations
    float*  float_ptr  = gpu_tracker->allocate<float>(1000);
    double* double_ptr = gpu_tracker->allocate<double>(500);

    if (float_ptr && double_ptr)
    {
        // Verify allocations are tracked
        auto [device_mem, unified_mem, pinned_mem] = gpu_tracker->GetGPUMemoryUsage();
        EXPECT_GT(device_mem, 0);

        // Test deallocations
        gpu_tracker->deallocate(float_ptr, 1000);
        gpu_tracker->deallocate(double_ptr, 500);

        QUARISMA_LOG_INFO("GPU typed allocation tracking test passed");
    }
    else
    {
        QUARISMA_LOG_INFO("GPU typed allocation failed (expected if insufficient GPU memory)");
    }
}

/**
 * @brief Test GPU timing statistics
 */
QUARISMATEST(GpuAllocatorTracking, provides_timing_statistics)
{
    auto& device_manager = gpu_device_manager::instance();
    device_manager.initialize();
    auto runtime_info = device_manager.get_runtime_info();

    if (!runtime_info.cuda_available)
    {
        QUARISMA_LOG_INFO("CUDA not available, skipping timing statistics test");
        return;
    }

    auto gpu_tracker = std::make_unique<gpu_allocator_tracking>(device_enum::CUDA, 0, true, false);

    // Reset timing stats
    gpu_tracker->ResetGPUTimingStats();

    // Perform allocations to generate timing data
    std::vector<void*> gpu_ptrs;
    const size_t       num_allocations = 5;
    const size_t       base_size       = 1024;

    for (size_t i = 0; i < num_allocations; ++i)
    {
        size_t alloc_size = base_size * (i + 1);
        void*  ptr        = gpu_tracker->allocate_raw(alloc_size, 256);

        if (ptr != nullptr)
        {
            gpu_ptrs.push_back(ptr);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // Get timing statistics
    auto timing_stats = gpu_tracker->GetGPUTimingStats();
    if (!gpu_ptrs.empty())
    {
        EXPECT_GT(timing_stats.total_allocations, 0);
    }

    // Clean up
    for (size_t i = 0; i < gpu_ptrs.size(); ++i)
    {
        size_t alloc_size = base_size * (i + 1);
        gpu_tracker->deallocate_raw(gpu_ptrs[i], alloc_size);
    }

    QUARISMA_LOG_INFO("GPU timing statistics test passed");
}

/**
 * @brief Test GPU logging levels
 */
QUARISMATEST(GpuAllocatorTracking, manages_logging_levels)
{
    auto& device_manager = gpu_device_manager::instance();
    device_manager.initialize();
    auto runtime_info = device_manager.get_runtime_info();

    if (!runtime_info.cuda_available)
    {
        QUARISMA_LOG_INFO("CUDA not available, skipping logging levels test");
        return;
    }

    auto gpu_tracker = std::make_unique<gpu_allocator_tracking>(device_enum::CUDA, 0, true, false);

    // Test different logging levels
    gpu_tracker->SetGPULoggingLevel(gpu_tracking_log_level::SILENT);
    EXPECT_EQ(gpu_tracking_log_level::SILENT, gpu_tracker->GetGPULoggingLevel());

    gpu_tracker->SetGPULoggingLevel(gpu_tracking_log_level::ERROR);
    EXPECT_EQ(gpu_tracking_log_level::ERROR, gpu_tracker->GetGPULoggingLevel());

    gpu_tracker->SetGPULoggingLevel(gpu_tracking_log_level::INFO);
    EXPECT_EQ(gpu_tracking_log_level::INFO, gpu_tracker->GetGPULoggingLevel());

    gpu_tracker->SetGPULoggingLevel(gpu_tracking_log_level::DEBUG_LEVEL);
    EXPECT_EQ(gpu_tracking_log_level::DEBUG_LEVEL, gpu_tracker->GetGPULoggingLevel());

    QUARISMA_LOG_INFO("GPU logging levels test passed");
}

/**
 * @brief Test GPU efficiency metrics calculation
 */
QUARISMATEST(GpuAllocatorTracking, calculates_efficiency_metrics)
{
    auto& device_manager = gpu_device_manager::instance();
    device_manager.initialize();
    auto runtime_info = device_manager.get_runtime_info();

    if (!runtime_info.cuda_available)
    {
        QUARISMA_LOG_INFO("CUDA not available, skipping efficiency metrics test");
        return;
    }

    auto gpu_tracker = std::make_unique<gpu_allocator_tracking>(device_enum::CUDA, 0, true, false);

    // Perform some allocations
    void* ptr1 = gpu_tracker->allocate_raw(2048, 256);
    void* ptr2 = gpu_tracker->allocate_raw(4096, 256);

    if (ptr1 && ptr2)
    {
        // Get efficiency metrics
        auto [coalescing_efficiency, memory_utilization, gpu_efficiency_score] =
            gpu_tracker->GetGPUEfficiencyMetrics();

        // Metrics should be valid (between 0.0 and 1.0)
        EXPECT_GE(coalescing_efficiency, 0.0);
        EXPECT_LE(coalescing_efficiency, 1.0);
        EXPECT_GE(memory_utilization, 0.0);
        EXPECT_LE(memory_utilization, 1.0);
        EXPECT_GE(gpu_efficiency_score, 0.0);
        EXPECT_LE(gpu_efficiency_score, 1.0);

        // Clean up
        gpu_tracker->deallocate_raw(ptr1, 2048);
        gpu_tracker->deallocate_raw(ptr2, 4096);

        QUARISMA_LOG_INFO("GPU efficiency metrics test passed");
    }
    else
    {
        QUARISMA_LOG_INFO("GPU allocation failed for efficiency metrics test");
    }
}

/**
 * @brief Test GPU report generation
 */
QUARISMATEST(GpuAllocatorTracking, generates_comprehensive_reports)
{
    auto& device_manager = gpu_device_manager::instance();
    device_manager.initialize();
    auto runtime_info = device_manager.get_runtime_info();

    if (!runtime_info.cuda_available)
    {
        QUARISMA_LOG_INFO("CUDA not available, skipping report generation test");
        return;
    }

    auto gpu_tracker = std::make_unique<gpu_allocator_tracking>(device_enum::CUDA, 0, true, false);

    // Perform some operations to generate data
    void* ptr = gpu_tracker->allocate_raw(1024, 256);

    // Generate reports
    std::string basic_report    = gpu_tracker->GenerateGPUReport(false, false);
    std::string detailed_report = gpu_tracker->GenerateGPUReport(true, true);

    // Reports should not be empty
    EXPECT_FALSE(basic_report.empty());
    EXPECT_FALSE(detailed_report.empty());

    // Detailed report should be longer
    EXPECT_GE(detailed_report.length(), basic_report.length());

    if (ptr)
    {
        gpu_tracker->deallocate_raw(ptr, 1024);
    }

    QUARISMA_LOG_INFO("GPU report generation test passed");
}

#endif  // PROJECT_HAS_CUDA
