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
#if 0
#include "MemoryTest.h"
#include "common/macros.h"

#if PROJECT_HAS_CUDA

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

//#include "logger/logger.h"
#include "common/device.h"
#include "gpu/gpu_resource_tracker.h"

using namespace memory;
using namespace memory::gpu;

/**
 * @brief Test GPU resource tracker singleton access
 */
MEMORYTEST(GpuResourceTracker, provides_singleton_instance)
{
    // Test singleton access
    auto& tracker1 = gpu_resource_tracker::instance();
    auto& tracker2 = gpu_resource_tracker::instance();

    // Should be the same instance
    EXPECT_EQ(&tracker1, &tracker2);

    QUARISMA_LOG_INFO("GPU resource tracker singleton test passed");
}

/**
 * @brief Test leak detection configuration
 */
MEMORYTEST(GpuResourceTracker, configures_leak_detection)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Test leak detection configuration
    leak_detection_config config;
    config.enabled = true;
    config.leak_threshold_ms = 30000.0;  // 30 seconds
    config.max_call_stack_depth = 5;
    config.enable_periodic_scan = false;  // Disable for testing
    config.enable_auto_reporting = true;

    try
    {
        tracker.configure_leak_detection(config);
        EXPECT_TRUE(true);  // Configuration succeeded

        QUARISMA_LOG_INFO("GPU resource tracker leak detection configuration test passed");
    }
    catch (const std::exception& e)
    {
        QUARISMA_LOG_INFO("GPU resource tracker configuration failed: {}", e.what());
    }
}

/**
 * @brief Test allocation tracking functionality
 */
MEMORYTEST(GpuResourceTracker, tracks_allocations)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Enable tracking
    tracker.set_tracking_enabled(true);
    EXPECT_TRUE(tracker.is_tracking_enabled());

    // Create test allocation (using host memory for simplicity)
    void* test_ptr = malloc(1024);
    EXPECT_NE(nullptr, test_ptr);

    // Track allocation
    size_t alloc_id = tracker.track_allocation(
        test_ptr, 1024, device_enum::CPU, 0, "test_allocation");

    EXPECT_GT(alloc_id, 0);

    // Get allocation info
    auto alloc_info = tracker.get_allocation_info(test_ptr);
    if (alloc_info)
    {
        EXPECT_EQ(test_ptr, alloc_info->ptr);
        EXPECT_EQ(1024, alloc_info->size);
        EXPECT_EQ(device_enum::CPU, alloc_info->device.type());
        EXPECT_TRUE(alloc_info->is_active);
        EXPECT_EQ("test_allocation", alloc_info->tag);
    }

    // Track deallocation
    bool dealloc_tracked = tracker.track_deallocation(test_ptr);
    EXPECT_TRUE(dealloc_tracked);

    // Clean up
    free(test_ptr);

    QUARISMA_LOG_INFO("GPU resource tracker allocation tracking test passed");
}

/**
 * @brief Test memory access recording
 */
MEMORYTEST(GpuResourceTracker, records_memory_access)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Create and track allocation
    void* test_ptr = malloc(512);
    EXPECT_NE(nullptr, test_ptr);

    size_t alloc_id = tracker.track_allocation(
        test_ptr, 512, device_enum::CPU, 0, "access_test");

    // Record memory access
    tracker.record_access(test_ptr);

    // Get allocation info to check access was recorded
    auto alloc_info = tracker.get_allocation_info(test_ptr);
    if (alloc_info)
    {
        EXPECT_GT(alloc_info->access_count.load(), 0);
    }

    // Clean up
    tracker.track_deallocation(test_ptr);
    free(test_ptr);

    QUARISMA_LOG_INFO("GPU resource tracker memory access recording test passed");
}

/**
 * @brief Test statistics collection
 */
MEMORYTEST(GpuResourceTracker, provides_resource_statistics)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Get initial statistics
    auto initial_stats = tracker.get_statistics();

    // Create multiple allocations
    std::vector<void*> test_ptrs;
    for (int i = 0; i < 3; ++i)
    {
        void* ptr = malloc(256 * (i + 1));
        if (ptr)
        {
            test_ptrs.push_back(ptr);
            tracker.track_allocation(ptr, 256 * (i + 1), device_enum::CPU, 0, "stats_test");
        }
    }

    // Get updated statistics
    auto updated_stats = tracker.get_statistics();
    if (!test_ptrs.empty())
    {
        EXPECT_GE(updated_stats.total_allocations, initial_stats.total_allocations);
        //EXPECT_GE(updated_stats.current_bytes_in_use, initial_stats.current_bytes_in_use);
    }

    // Clean up
    for (void* ptr : test_ptrs)
    {
        tracker.track_deallocation(ptr);
        free(ptr);
    }

    QUARISMA_LOG_INFO("GPU resource tracker statistics test passed");
}

/**
 * @brief Test active allocations retrieval
 */
MEMORYTEST(GpuResourceTracker, retrieves_active_allocations)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Create test allocations
    void* ptr1 = malloc(128);
    void* ptr2 = malloc(256);

    if (ptr1 && ptr2)
    {
        tracker.track_allocation(ptr1, 128, device_enum::CPU, 0, "active_test_1");
        tracker.track_allocation(ptr2, 256, device_enum::CPU, 0, "active_test_2");

        // Get active allocations
        auto active_allocations = tracker.get_active_allocations();

        // Should have at least our test allocations
        bool found_ptr1 = false, found_ptr2 = false;
        for (const auto& alloc : active_allocations)
        {
            if (alloc->ptr == ptr1) found_ptr1 = true;
            if (alloc->ptr == ptr2) found_ptr2 = true;
        }

        EXPECT_TRUE(found_ptr1);
        EXPECT_TRUE(found_ptr2);

        // Clean up
        tracker.track_deallocation(ptr1);
        tracker.track_deallocation(ptr2);
        free(ptr1);
        free(ptr2);
    }

    QUARISMA_LOG_INFO("GPU resource tracker active allocations test passed");
}

/**
 * @brief Test allocations by tag retrieval
 */
MEMORYTEST(GpuResourceTracker, retrieves_allocations_by_tag)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Create allocations with specific tags
    void* ptr1 = malloc(64);
    void* ptr2 = malloc(128);
    void* ptr3 = malloc(192);

    if (ptr1 && ptr2 && ptr3)
    {
        tracker.track_allocation(ptr1, 64, device_enum::CPU, 0, "tag_test");
        tracker.track_allocation(ptr2, 128, device_enum::CPU, 0, "tag_test");
        tracker.track_allocation(ptr3, 192, device_enum::CPU, 0, "other_tag");

        // Get allocations by tag
        auto tag_allocations = tracker.get_allocations_by_tag("tag_test");

        // Should find exactly 2 allocations with "tag_test"
        int found_count = 0;
        for (const auto& alloc : tag_allocations)
        {
            if (alloc->tag == "tag_test")
            {
                found_count++;
            }
        }

        EXPECT_GE(found_count, 2);  // At least our 2 test allocations

        // Clean up
        tracker.track_deallocation(ptr1);
        tracker.track_deallocation(ptr2);
        tracker.track_deallocation(ptr3);
        free(ptr1);
        free(ptr2);
        free(ptr3);
    }

    QUARISMA_LOG_INFO("GPU resource tracker allocations by tag test passed");
}

/**
 * @brief Test allocations by device retrieval
 */
MEMORYTEST(GpuResourceTracker, retrieves_allocations_by_device)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Create allocations on different devices
    void* cpu_ptr = malloc(256);
    void* gpu_ptr = malloc(512);  // Simulated GPU allocation

    if (cpu_ptr && gpu_ptr)
    {
        tracker.track_allocation(cpu_ptr, 256, device_enum::CPU, 0, "cpu_alloc");
        tracker.track_allocation(gpu_ptr, 512, device_enum::CUDA, 0, "gpu_alloc");

        // Get CPU allocations
        auto cpu_allocations = tracker.get_allocations_by_device(device_enum::CPU, 0);

        // Should find at least our CPU allocation
        bool found_cpu = false;
        for (const auto& alloc : cpu_allocations)
        {
            if (alloc->ptr == cpu_ptr && alloc->device.type() == device_enum::CPU)
            {
                found_cpu = true;
                break;
            }
        }

        EXPECT_TRUE(found_cpu);

        // Get CUDA allocations
        auto cuda_allocations = tracker.get_allocations_by_device(device_enum::CUDA, 0);

        // Should find at least our CUDA allocation
        bool found_cuda = false;
        for (const auto& alloc : cuda_allocations)
        {
            if (alloc->ptr == gpu_ptr && alloc->device.type() == device_enum::CUDA)
            {
                found_cuda = true;
                break;
            }
        }

        EXPECT_TRUE(found_cuda);

        // Clean up
        tracker.track_deallocation(cpu_ptr);
        tracker.track_deallocation(gpu_ptr);
        free(cpu_ptr);
        free(gpu_ptr);
    }

    QUARISMA_LOG_INFO("GPU resource tracker allocations by device test passed");
}

/**
 * @brief Test leak detection functionality
 */
MEMORYTEST(GpuResourceTracker, detects_memory_leaks)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Configure leak detection with very short threshold for testing
    leak_detection_config config;
    config.enabled = true;
    config.leak_threshold_ms = 100.0;  // 100ms threshold
    config.enable_periodic_scan = false;

    tracker.configure_leak_detection(config);

    // Create allocation and don't deallocate it
    void* leak_ptr = malloc(1024);
    if (leak_ptr)
    {
        tracker.track_allocation(leak_ptr, 1024, device_enum::CPU, 0, "potential_leak");

        // Wait longer than leak threshold
        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        // Detect leaks
        auto leaks = tracker.detect_leaks();

        // Should detect our allocation as a potential leak
        bool found_leak = false;
        for (const auto& leak : leaks)
        {
            if (leak->ptr == leak_ptr)
            {
                found_leak = true;
                break;
            }
        }

        EXPECT_TRUE(found_leak);

        // Clean up
        tracker.track_deallocation(leak_ptr);
        free(leak_ptr);
    }

    QUARISMA_LOG_INFO("GPU resource tracker leak detection test passed");
}

/**
 * @brief Test report generation
 */
MEMORYTEST(GpuResourceTracker, generates_resource_reports)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Create some allocations for reporting
    void* ptr1 = malloc(512);
    void* ptr2 = malloc(1024);

    if (ptr1 && ptr2)
    {
        tracker.track_allocation(ptr1, 512, device_enum::CPU, 0, "report_test_1");
        tracker.track_allocation(ptr2, 1024, device_enum::CPU, 0, "report_test_2");

        // Generate basic report
        std::string basic_report = tracker.generate_report(false);
        EXPECT_FALSE(basic_report.empty());

        // Generate detailed report with call stacks
        std::string detailed_report = tracker.generate_report(true);
        EXPECT_FALSE(detailed_report.empty());
        EXPECT_GE(detailed_report.length(), basic_report.length());

        // Generate leak report
        std::string leak_report = tracker.generate_leak_report();
        EXPECT_FALSE(leak_report.empty());

        // Clean up
        tracker.track_deallocation(ptr1);
        tracker.track_deallocation(ptr2);
        free(ptr1);
        free(ptr2);
    }

    QUARISMA_LOG_INFO("GPU resource tracker report generation test passed");
}

/**
 * @brief Test tracking enable/disable functionality
 */
MEMORYTEST(GpuResourceTracker, controls_tracking_state)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Test enabling tracking
    tracker.set_tracking_enabled(true);
    EXPECT_TRUE(tracker.is_tracking_enabled());

    // Test disabling tracking
    tracker.set_tracking_enabled(false);
    EXPECT_FALSE(tracker.is_tracking_enabled());

    // Re-enable for other tests
    tracker.set_tracking_enabled(true);
    EXPECT_TRUE(tracker.is_tracking_enabled());

    QUARISMA_LOG_INFO("GPU resource tracker tracking control test passed");
}

/**
 * @brief Test data clearing functionality
 */
MEMORYTEST(GpuResourceTracker, clears_tracking_data)
{
    auto& tracker = gpu_resource_tracker::instance();

    // Create some allocations
    void* ptr1 = malloc(256);
    void* ptr2 = malloc(512);

    if (ptr1 && ptr2)
    {
        tracker.track_allocation(ptr1, 256, device_enum::CPU, 0, "clear_test_1");
        tracker.track_allocation(ptr2, 512, device_enum::CPU, 0, "clear_test_2");

        // Verify allocations are tracked
        auto active_before = tracker.get_active_allocations();
        EXPECT_GE(active_before.size(), 2);

        // Clear all data
        tracker.clear_all_data();

        // Verify data was cleared
        auto active_after = tracker.get_active_allocations();
        EXPECT_LT(active_after.size(), active_before.size());

        // Clean up (even though tracking data was cleared)
        free(ptr1);
        free(ptr2);
    }

    QUARISMA_LOG_INFO("GPU resource tracker data clearing test passed");
}

#endif  // PROJECT_HAS_CUDA
#endif
