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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "MemoryTest.h"
#include "cpu/allocator_device.h"
#include "profiler/unified_memory_stats.h"

using namespace memory;
using namespace memory;

namespace memory
{

// Forward declarations of helper functions defined in TestCPUMemory.cpp
struct TestStruct;
bool IsAligned(void* ptr, size_t alignment);
bool ValidateMemory(const void* ptr, size_t size, uint8_t pattern);
}  // namespace memory

MEMORYTEST(CPUMemoryStats, unified_resource_stats_basic_functionality)
{
    unified_resource_stats stats;

    // Test initial state
    EXPECT_EQ(stats.num_allocs.load(), 0);
    EXPECT_EQ(stats.num_deallocs.load(), 0);
    EXPECT_EQ(stats.bytes_in_use.load(), 0);
    EXPECT_EQ(stats.peak_bytes_in_use.load(), 0);
    EXPECT_EQ(stats.active_allocations.load(), 0);

    // Test basic operations
    stats.num_allocs.fetch_add(10);
    stats.num_deallocs.fetch_add(5);
    stats.bytes_in_use.store(1024ULL);           // 1MB
    stats.peak_bytes_in_use.store(2 * 1024ULL);  // 2MB
    stats.active_allocations.store(5);
    stats.total_bytes_allocated.store(10 * 1024ULL);  // 10MB

    // Test derived metrics
    double efficiency = stats.memory_efficiency();
    EXPECT_NEAR(efficiency, 0.5, 0.01);  // 1MB / 2MB = 0.5

    double avg_size = stats.average_allocation_size();
    EXPECT_NEAR(avg_size, 1024.0, 1000.0);  // 10MB / 10 allocs = 1MB

    double success_rate = stats.allocation_success_rate();
    EXPECT_NEAR(success_rate, 100.0, 0.1);  // No failed allocations

    // Test with failed allocations
    stats.failed_allocations.store(2);
    success_rate = stats.allocation_success_rate();
    EXPECT_LT(std::abs(success_rate - 83.33), 0.1);  // 10/(10+2) ≈ 83.33%

    // Test debug string generation
    std::string debug_str = stats.debug_string();
    EXPECT_FALSE(debug_str.empty());
    EXPECT_NE(debug_str.find("unified_resource_stats"), std::string::npos);

    END_TEST();
}

MEMORYTEST(CPUMemoryStats, atomic_timing_stats_functionality)
{
    atomic_timing_stats stats;

    // Test initial state
    EXPECT_EQ(stats.total_allocations.load(), 0);
    EXPECT_EQ(stats.total_deallocations.load(), 0);
    EXPECT_EQ(stats.total_alloc_time_us.load(), 0);
    EXPECT_EQ(stats.min_alloc_time_us.load(), UINT64_MAX);
    EXPECT_EQ(stats.total_dealloc_time_us.load(), 0);

    // Simulate timing data
    stats.total_allocations.store(100);
    stats.total_deallocations.store(80);
    stats.total_alloc_time_us.store(50000);    // 50ms total
    stats.total_dealloc_time_us.store(20000);  // 20ms total
    stats.min_alloc_time_us.store(100);        // 100μs min
    stats.max_alloc_time_us.store(2000);       // 2ms max
    stats.min_dealloc_time_us.store(50);       // 50μs min
    stats.max_dealloc_time_us.store(1000);     // 1ms max

    // Test derived metrics
    double avg_alloc = stats.average_alloc_time_us();
    EXPECT_NEAR(avg_alloc, 500.0, 1.0);  // 50000μs / 100 = 500μs

    double avg_dealloc = stats.average_dealloc_time_us();
    EXPECT_NEAR(avg_dealloc, 250.0, 1.0);  // 20000μs / 80 = 250μs

    END_TEST();
}

MEMORYTEST(CPUMemoryStats, timing_stats_snapshot)
{
    // Create atomic version and populate it
    atomic_timing_stats atomic_stats;
    atomic_stats.total_allocations.store(50);
    atomic_stats.total_alloc_time_us.store(25000);
    atomic_stats.min_alloc_time_us.store(200);
    atomic_stats.max_alloc_time_us.store(1500);

    // Create snapshot from atomic version
    atomic_timing_stats snapshot = atomic_stats;

    // Verify snapshot contains correct values
    EXPECT_EQ(snapshot.total_allocations.load(), 50);
    EXPECT_EQ(snapshot.total_alloc_time_us.load(), 25000);
    EXPECT_EQ(snapshot.min_alloc_time_us.load(), 200);
    EXPECT_EQ(snapshot.max_alloc_time_us.load(), 1500);

    // Test derived metrics on snapshot
    double avg = snapshot.average_alloc_time_us();
    EXPECT_NEAR(avg, 500.0, 1.0);  // 25000 / 50 = 500

    // Test that snapshot is independent of original
    atomic_stats.total_allocations.store(100);
    EXPECT_EQ(snapshot.total_allocations.load(), 50);  // Unchanged

    END_TEST();
}

MEMORYTEST(CPUMemoryStats, unified_cache_stats_functionality)
{
    unified_cache_stats stats;

    // Test initial state
    EXPECT_EQ(stats.cache_hits.load(), 0);
    EXPECT_EQ(stats.cache_misses.load(), 0);
    EXPECT_EQ(stats.cache_hit_rate(), 0.0);

    // Simulate cache activity
    stats.cache_hits.store(80);
    stats.cache_misses.store(20);
    stats.bytes_cached.store(4 * 1024ULL);  // 4MB cached
    stats.driver_allocations.store(25);     // 25 driver calls
    stats.driver_frees.store(20);           // 20 driver frees

    // Test cache hit rate
    double hit_rate = stats.cache_hit_rate();
    EXPECT_NEAR(hit_rate, 0.8, 0.01);  // 80 / (80 + 20) = 0.8

    double efficiency = stats.cache_efficiency_percent();
    EXPECT_NEAR(efficiency, 80.0, 0.1);  // 80%

    // Test driver call reduction
    double reduction = stats.driver_call_reduction();
    EXPECT_LT(std::abs(reduction - 2.22), 0.01);  // 100 / 45 ≈ 2.22

    // Test reset functionality
    stats.reset();
    EXPECT_EQ(stats.cache_hits.load(), 0);
    EXPECT_EQ(stats.cache_misses.load(), 0);
    EXPECT_EQ(stats.bytes_cached.load(), 0);

    MEMORY_LOG_INFO("✓ unified_cache_stats functionality passed");
}

/**
     * @brief Test memory fragmentation metrics calculation
     */
MEMORYTEST(CPUMemoryStats, test_memory_fragmentation_metrics)
{
    MEMORY_LOG_INFO("Testing memory_fragmentation_metrics calculation...");

    // Test with empty free blocks
    std::vector<size_t> empty_blocks;
    auto                metrics = memory_fragmentation_metrics::calculate(1024, 1000, empty_blocks);
    EXPECT_EQ(metrics.total_free_blocks, 0);
    EXPECT_EQ(metrics.wasted_bytes, 0);

    // Test with realistic fragmentation scenario
    std::vector<size_t> free_blocks     = {64, 128, 256, 512, 1024, 32, 96};
    size_t              total_allocated = 8192;
    size_t              total_requested = 7500;

    metrics =
        memory_fragmentation_metrics::calculate(total_allocated, total_requested, free_blocks);

    // Verify basic statistics
    EXPECT_EQ(metrics.total_free_blocks, 7);
    EXPECT_GE(metrics.wasted_bytes, 0);  // Should be calculated based on fragmentation
    EXPECT_EQ(metrics.largest_free_block, 1024);

    // Calculate expected average: (64+128+256+512+1024+32+96) / 7 = 2112 / 7 ≈ 301.7
    EXPECT_LT(std::abs(metrics.average_free_block_size - 301.7), 1.0);

    // Test fragmentation ratio bounds
    EXPECT_GE(metrics.fragmentation_ratio, 0.0);
    EXPECT_LE(metrics.fragmentation_ratio, 1.0);

    // Test internal fragmentation calculation
    EXPECT_GE(metrics.internal_fragmentation, 0.0);
    EXPECT_LE(metrics.internal_fragmentation, 100.0);
    EXPECT_GT(metrics.internal_fragmentation, 0.0);

    MEMORY_LOG_INFO("  Fragmentation ratio: {}", metrics.fragmentation_ratio);
    MEMORY_LOG_INFO("  Internal fragmentation: {}%", metrics.internal_fragmentation);
    MEMORY_LOG_INFO("  External fragmentation: {}%", metrics.external_fragmentation);

    MEMORY_LOG_INFO("✓ memory_fragmentation_metrics calculation passed");
}

/**
     * @brief Test comprehensive memory statistics integration
     */
MEMORYTEST(CPUMemoryStats, test_comprehensive_memory_stats)
{
    MEMORY_LOG_INFO("Testing comprehensive_memory_stats integration...");

    comprehensive_memory_stats stats("TestAllocator");

    // Populate resource statistics
    stats.resource_stats.num_allocs.store(1000);
    stats.resource_stats.num_deallocs.store(800);
    stats.resource_stats.bytes_in_use.store(5 * 1024ULL);       // 5MB
    stats.resource_stats.peak_bytes_in_use.store(8 * 1024ULL);  // 8MB

    // Populate timing statistics
    stats.timing_stats.total_allocations.store(1000);
    stats.timing_stats.total_alloc_time_us.store(500000);  // 500ms total

    // Populate cache statistics
    stats.cache_stats.cache_hits.store(750);
    stats.cache_stats.cache_misses.store(250);

    // Test overall efficiency calculation
    double efficiency = stats.overall_efficiency();
    EXPECT_GE(efficiency, 0.0);

    // Test operations per second calculation
    double ops_per_sec = stats.operations_per_second();
    EXPECT_GT(ops_per_sec, 0.0);

    // Test report generation
    const std::string report = stats.generate_report();
    MEMORY_LOG_INFO("{}", report);
    EXPECT_FALSE(report.empty());
    //EXPECT_NE(report.find("CPU_Allocator"), std::string::npos);
    EXPECT_NE(report.find("Resource Stats"), std::string::npos);
    EXPECT_NE(report.find("Cache Performance"), std::string::npos);
    EXPECT_NE(report.find("Overall Efficiency"), std::string::npos);

    MEMORY_LOG_INFO("  Overall efficiency: {}%", (efficiency * 100.0));
    MEMORY_LOG_INFO("  Operations per second: {}", ops_per_sec);

    // Test individual reset functionality
    stats.resource_stats.reset();
    stats.cache_stats.reset();
    stats.timing_stats.reset();
    EXPECT_EQ(stats.resource_stats.num_allocs.load(), 0);
    EXPECT_EQ(stats.cache_stats.cache_hits.load(), 0);
    EXPECT_EQ(stats.timing_stats.total_allocations.load(), 0);

    MEMORY_LOG_INFO("✓ comprehensive_memory_stats integration passed");
}

MEMORYTEST(CPUMemoryStats, comprehensive_tests)
{
    // Test unified resource statistics
    MEMORYTEST_CALL(CPUMemoryStats, unified_resource_stats_basic_functionality);

    // Test atomic timing statistics
    MEMORYTEST_CALL(CPUMemoryStats, atomic_timing_stats_functionality);

    // Test timing stats snapshot
    MEMORYTEST_CALL(CPUMemoryStats, timing_stats_snapshot);

    // Test unified cache statistics
    MEMORYTEST_CALL(CPUMemoryStats, unified_cache_stats_functionality);

    END_TEST();
}
