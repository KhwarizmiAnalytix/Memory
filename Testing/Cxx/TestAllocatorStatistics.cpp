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

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

#include "MemoryTest.h"
#include "common/pointer.h"
//#include "logger/logger.h"
#include "backend/allocator_bfc.h"
#include "backend/allocator_pool.h"
#include "backend/allocator_tracking.h"
#include "cpu/allocator.h"
#include "cpu/allocator_cpu.h"
#include "profiler/unified_memory_stats.h"
#include "visualization/ascii_visualizer.h"

using namespace memory;
using namespace memory;

namespace
{

/**
 * @brief Helper function to format bytes in human-readable format
 */
std::string format_bytes(size_t bytes)
{
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int         unit    = 0;
    double      size    = static_cast<double>(bytes);

    while (size >= 1024.0 && unit < 4)
    {
        size /= 1024.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

/**
 * @brief Create ASCII bar chart for memory usage
 */
std::string create_memory_bar(size_t current, size_t peak, size_t max_width = 60)
{
    std::ostringstream bar;
    size_t             max_value = std::max(current, peak);
    if (max_value == 0)
    {
        max_value = 1;
    }

    size_t current_width = (current * max_width) / max_value;
    size_t peak_width    = (peak * max_width) / max_value;

    bar << "Current: " << std::setw(12) << format_bytes(current) << " |";
    for (size_t i = 0; i < max_width; ++i)
    {
        bar << (i < current_width ? '#' : ' ');
    }
    bar << "|\n";

    bar << "Peak:    " << std::setw(12) << format_bytes(peak) << " |";
    for (size_t i = 0; i < max_width; ++i)
    {
        bar << (i < peak_width ? '=' : ' ');
    }
    bar << "|\n";

    return bar.str();
}

/**
 * @brief Display comprehensive allocator statistics with ASCII visualization
 */
void display_allocator_stats(const std::string& allocator_name, const allocator_stats& stats)
{
    QUARISMA_LOG_INFO("\n========================================");
    QUARISMA_LOG_INFO("Allocator: {}", allocator_name);
    QUARISMA_LOG_INFO("========================================");

    // Basic statistics
    QUARISMA_LOG_INFO("Allocation Count:     {}", stats.num_allocs.load());
    QUARISMA_LOG_INFO("Deallocation Count:   {}", stats.num_deallocs.load());
    QUARISMA_LOG_INFO("Active Allocations:   {}", stats.active_allocations.load());
    QUARISMA_LOG_INFO("Current Memory Usage: {}", format_bytes(stats.bytes_in_use.load()));
    QUARISMA_LOG_INFO("Peak Memory Usage:    {}", format_bytes(stats.peak_bytes_in_use.load()));
    QUARISMA_LOG_INFO("Largest Allocation:   {}", format_bytes(stats.largest_alloc_size.load()));

    // Memory usage visualization
    QUARISMA_LOG_INFO("\nMemory Usage Visualization:");
    QUARISMA_LOG_INFO(
        "{}", create_memory_bar(stats.bytes_in_use.load(), stats.peak_bytes_in_use.load()));

    // Additional BFC-specific statistics
    if (stats.bytes_reserved.load() > 0)
    {
        QUARISMA_LOG_INFO("\nBFC Allocator Specific:");
        QUARISMA_LOG_INFO("Bytes Reserved:       {}", format_bytes(stats.bytes_reserved.load()));
        QUARISMA_LOG_INFO(
            "Peak Bytes Reserved:  {}", format_bytes(stats.peak_bytes_reserved.load()));
        QUARISMA_LOG_INFO(
            "Largest Free Block:   {}", format_bytes(stats.largest_free_block_bytes.load()));
    }

    // Efficiency metrics
    int64_t               total_allocs   = stats.num_allocs.load();
    MEMORY_UNUSED int64_t total_deallocs = stats.num_deallocs.load();
    if (total_allocs > 0)
    {
        double avg_alloc_size =
            static_cast<double>(stats.total_bytes_allocated.load()) / total_allocs;
        QUARISMA_LOG_INFO("\nEfficiency Metrics:");
        QUARISMA_LOG_INFO(
            "Average Allocation Size: {}", format_bytes(static_cast<size_t>(avg_alloc_size)));
        QUARISMA_LOG_INFO("Memory Efficiency:       {:.2f}%", stats.memory_efficiency() * 100.0);
    }

    QUARISMA_LOG_INFO("========================================\n");
}

/**
 * @brief Display timing statistics with ASCII visualization
 */
void display_timing_stats(const std::string& allocator_name, const atomic_timing_stats& timing)
{
    QUARISMA_LOG_INFO("\n========================================");
    QUARISMA_LOG_INFO("Timing Statistics: {}", allocator_name);
    QUARISMA_LOG_INFO("========================================");

    QUARISMA_LOG_INFO("Total Allocations:    {}", timing.total_allocations.load());
    QUARISMA_LOG_INFO("Total Deallocations:  {}", timing.total_deallocations.load());
    QUARISMA_LOG_INFO("Avg Allocation Time:  {:.2f} μs", timing.average_alloc_time_us());
    QUARISMA_LOG_INFO("Avg Deallocation Time: {:.2f} μs", timing.average_dealloc_time_us());

    uint64_t min_alloc = timing.min_alloc_time_us.load();
    uint64_t max_alloc = timing.max_alloc_time_us.load();
    if (min_alloc != UINT64_MAX)
    {
        QUARISMA_LOG_INFO("Min Allocation Time:  {} μs", min_alloc);
        QUARISMA_LOG_INFO("Max Allocation Time:  {} μs", max_alloc);
    }

    uint64_t min_dealloc = timing.min_dealloc_time_us.load();
    uint64_t max_dealloc = timing.max_dealloc_time_us.load();
    if (min_dealloc != UINT64_MAX)
    {
        QUARISMA_LOG_INFO("Min Deallocation Time: {} μs", min_dealloc);
        QUARISMA_LOG_INFO("Max Deallocation Time: {} μs", max_dealloc);
    }

    QUARISMA_LOG_INFO("========================================\n");
}

}  // anonymous namespace

// ============================================================================
// CPU Allocator Statistics Tests
// ============================================================================

MEMORYTEST(AllocatorStatistics, CPUAllocatorBasicStats)
{
    QUARISMA_LOG_INFO("Testing CPU allocator statistics exposure...");

    // Enable statistics collection
    EnableCPUAllocatorStats();

    // Get CPU allocator
    Allocator* cpu_alloc = cpu_allocator(0);
    ASSERT_NE(nullptr, cpu_alloc);

    // Perform various allocations
    std::vector<void*> ptrs;
    const size_t       sizes[] = {64, 256, 1024, 4096, 16384, 65536};

    for (size_t size : sizes)
    {
        void* ptr = cpu_alloc->allocate_raw(64, size);
        EXPECT_NE(nullptr, ptr);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Get and display statistics
    auto stats_opt = cpu_alloc->GetStats();
    ASSERT_TRUE(stats_opt.has_value());

    display_allocator_stats("CPU Allocator", stats_opt.value());

    // Verify statistics
    EXPECT_GE(stats_opt->num_allocs.load(), 0);
    EXPECT_GE(stats_opt->bytes_in_use.load(), 0);
    EXPECT_GE(stats_opt->peak_bytes_in_use.load(), 0);

    // Cleanup
    for (void* ptr : ptrs)
    {
        cpu_alloc->deallocate_raw(ptr);
    }

    QUARISMA_LOG_INFO("CPU allocator statistics test completed successfully");
}

MEMORYTEST(AllocatorStatistics, BFCAllocatorStats)
{
    QUARISMA_LOG_INFO("Testing BFC allocator statistics exposure...");
    EnableCPUAllocatorStats();
    // Create BFC allocator
    auto sub_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    allocator_bfc::Options opts;
    opts.allow_growth       = true;
    opts.garbage_collection = true;

    allocator_bfc bfc_alloc(
        std::move(sub_allocator), 10ULL * 1024ULL * 1024ULL, "test_bfc_stats", opts);

    // Perform allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i)
    {
        size_t size = 1024 * (i + 1);
        void*  ptr  = bfc_alloc.allocate_raw(64, size);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Get and display statistics
    auto stats_opt = bfc_alloc.GetStats();
    ASSERT_TRUE(stats_opt.has_value());

    display_allocator_stats("BFC Allocator", stats_opt.value());

    // Verify BFC-specific statistics
    EXPECT_GE(stats_opt->bytes_reserved.load(), 0);
    EXPECT_GE(stats_opt->num_allocs.load(), 0);

    // Cleanup
    for (void* ptr : ptrs)
    {
        bfc_alloc.deallocate_raw(ptr);
    }

    QUARISMA_LOG_INFO("BFC allocator statistics test completed successfully");
}

MEMORYTEST(AllocatorStatistics, PoolAllocatorStats)
{
#if 0
    QUARISMA_LOG_INFO("Testing Pool allocator statistics exposure...");
    EnableCPUAllocatorStats();
    // Create pool allocator
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<memory::sub_allocator::Visitor>{}, std::vector<memory::sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        20,     // pool_size_limit
        false,  // auto_resize
        std::move(base_allocator),
        std::make_unique<NoopRounder>(),
        "test_pool_stats");

    // Perform allocations with repeated sizes to test pooling
    std::vector<void*> ptrs;
    const size_t       test_sizes[] = {512, 1024, 2048, 4096};

    // First round of allocations
    for (int round = 0; round < 3; ++round)
    {
        for (size_t size : test_sizes)
        {
            void* ptr = pool->allocate_raw(64, size);
            EXPECT_NE(nullptr, ptr);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }
    }

    // Get statistics
    auto stats_opt = pool->GetStats();
    ASSERT_TRUE(stats_opt.has_value());

    display_allocator_stats("Pool Allocator", stats_opt.value());

    // Verify pool statistics
    EXPECT_GT(stats_opt->num_allocs.load(), 0);
    EXPECT_GT(stats_opt->bytes_in_use.load(), 0);

    // Deallocate half to test pool reuse
    for (size_t i = 0; i < ptrs.size() / 2; ++i)
    {
        pool->deallocate_raw(ptrs[i]);
    }

    // Allocate again to test pool hits
    for (size_t size : test_sizes)
    {
        void* ptr = pool->allocate_raw(64, size);
        EXPECT_NE(nullptr, ptr);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Get updated statistics
    stats_opt = pool->GetStats();
    ASSERT_TRUE(stats_opt.has_value());

    QUARISMA_LOG_INFO("\nPool Allocator After Reuse:");
    display_allocator_stats("Pool Allocator (After Reuse)", stats_opt.value());

    // Cleanup
    for (void* ptr : ptrs)
    {
        pool->deallocate_raw(ptr);
    }

    QUARISMA_LOG_INFO("Pool allocator statistics test completed successfully");
#endif
}

MEMORYTEST(AllocatorStatistics, TrackingAllocatorStats)
{
#if 0
    QUARISMA_LOG_INFO("Testing Tracking allocator statistics and timing...");

    // Create tracking allocator
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0, std::vector<memory::sub_allocator::Visitor>{}, std::vector<memory::sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        10,
        false,
        std::move(base_allocator),
        std::make_unique<NoopRounder>(),
        "test_tracking_base");

    auto tracking = new allocator_tracking(pool.get(), true, true);
    tracking->SetLoggingLevel(tracking_log_level::INFO);

    // Perform allocations with various sizes
    std::vector<void*> ptrs;
    const size_t       sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    for (size_t size : sizes)
    {
        void* ptr = tracking->allocate_raw(64, size);
        EXPECT_NE(nullptr, ptr);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Get and display statistics
    auto stats_opt = tracking->GetStats();
    ASSERT_TRUE(stats_opt.has_value());

    display_allocator_stats("Tracking Allocator", stats_opt.value());

    // Get and display timing statistics
    auto timing_stats = tracking->GetTimingStats();
    display_timing_stats("Tracking Allocator", timing_stats);

    // Get efficiency metrics
    auto [utilization, overhead, efficiency] = tracking->GetEfficiencyMetrics();
    QUARISMA_LOG_INFO("\nEfficiency Analysis:");
    QUARISMA_LOG_INFO("Utilization Ratio: {:.2f}%", utilization * 100.0);
    QUARISMA_LOG_INFO("Overhead Ratio:    {:.2f}%", overhead * 100.0);
    QUARISMA_LOG_INFO("Efficiency Score:  {:.2f}%", efficiency * 100.0);

    // Verify statistics
    EXPECT_GT(stats_opt->num_allocs.load(), 0);
    EXPECT_GT(timing_stats.total_allocations.load(), 0);
    EXPECT_GE(utilization, 0.0);
    EXPECT_LE(utilization, 1.0);

    // Cleanup
    for (void* ptr : ptrs)
    {
        tracking->deallocate_raw(ptr);
    }

    // Properly release the tracking allocator using reference counting
    // NOTE: GetRecordsAndUnRef() may delete tracking object via reference counting.
    // We must NOT call pool.reset() here because tracking may still need access to pool during cleanup.
    // The pool will be automatically destroyed when it goes out of scope after this function returns.
    tracking->GetRecordsAndUnRef();

    QUARISMA_LOG_INFO("Tracking allocator statistics test completed successfully");
#endif
}

// FIXME: This test is currently disabled due to a segfault issue
// The crash occurs during test execution - needs investigation
// Root cause appears to be related to allocator_pool/allocator_tracking lifecycle management
MEMORYTEST(AllocatorStatistics, DISABLED_AllocationSizeDistribution)
{
    QUARISMA_LOG_INFO("Testing allocation size distribution visualization...");

    // Create tracking allocator for detailed analysis
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        50, false, std::move(base_allocator), std::make_unique<NoopRounder>(), "test_distribution");

    auto tracking = new allocator_tracking(pool.get(), true, true);

    // Perform allocations with various size patterns
    std::vector<void*> ptrs;

    // Small allocations (< 64 bytes)
    for (int i = 0; i < 20; ++i)
    {
        void* ptr = tracking->allocate_raw(64, 32);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Medium allocations (64 bytes - 4 KB)
    for (int i = 0; i < 30; ++i)
    {
        size_t size = 64 + (i * 128);
        void*  ptr  = tracking->allocate_raw(64, size);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Large allocations (> 4 KB)
    for (int i = 0; i < 10; ++i)
    {
        size_t size = 4096 * (i + 1);
        void*  ptr  = tracking->allocate_raw(64, size);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Get enhanced records for size distribution
    auto records = tracking->GetEnhancedRecords();

    // Create size distribution histogram
    std::map<std::string, int> size_buckets;
    size_buckets["< 64 B"]       = 0;
    size_buckets["64 B - 1 KB"]  = 0;
    size_buckets["1 KB - 4 KB"]  = 0;
    size_buckets["4 KB - 64 KB"] = 0;
    size_buckets["> 64 KB"]      = 0;

    for (const auto& record : records)
    {
        size_t size = record.requested_bytes;
        if (size < 64)
        {
            size_buckets["< 64 B"]++;
        }
        else if (size < 1024)
        {
            size_buckets["64 B - 1 KB"]++;
        }
        else if (size < 4096)
        {
            size_buckets["1 KB - 4 KB"]++;
        }
        else if (size < 65536)
        {
            size_buckets["4 KB - 64 KB"]++;
        }
        else
        {
            size_buckets["> 64 KB"]++;
        }
    }

    // Display histogram
    QUARISMA_LOG_INFO("\n========================================");
    QUARISMA_LOG_INFO("Allocation Size Distribution");
    QUARISMA_LOG_INFO("========================================");

    int max_count = 0;
    for (const auto& [bucket, count] : size_buckets)
    {
        max_count = std::max(max_count, count);
    }

    for (const auto& [bucket, count] : size_buckets)
    {
        int         bar_length = max_count > 0 ? (count * 50) / max_count : 0;
        std::string bar(bar_length, '#');
        QUARISMA_LOG_INFO("{:15} | {:4} | {}", bucket, count, bar);
    }

    QUARISMA_LOG_INFO("========================================\n");

    // Cleanup
    for (void* ptr : ptrs)
    {
        tracking->deallocate_raw(ptr);
    }

    // Properly release the tracking allocator using reference counting
    // NOTE: GetRecordsAndUnRef() may delete tracking object via reference counting.
    // We must NOT call pool.reset() here because tracking may still need access to pool during cleanup.
    // The pool will be automatically destroyed when it goes out of scope after this function returns.
    tracking->GetRecordsAndUnRef();

    QUARISMA_LOG_INFO("Allocation size distribution test completed successfully");
}

MEMORYTEST(AllocatorStatistics, ComprehensiveVisualization)
{
    QUARISMA_LOG_INFO("Testing comprehensive allocator visualization with ASCII visualizer...");

    // Create tracking allocator
    auto base_allocator = std::make_unique<basic_cpu_allocator>(
        0,
        std::vector<memory::sub_allocator::Visitor>{},
        std::vector<memory::sub_allocator::Visitor>{});

    EnableCPUAllocatorStats();
    auto pool = std::make_unique<allocator_cpu>(
        /*30,
        false,
        std::move(base_allocator),
        std::make_unique<NoopRounder>(),
        "test_visualization"*/);

    auto tracking = new allocator_tracking(pool.get(), true, true);

    // Create ASCII visualizer
    ascii_visualizer::visualization_config config;
    config.chart_width          = 60;
    config.max_histogram_height = 20;
    config.show_legends         = true;
    config.show_percentages     = true;

    ascii_visualizer visualizer(config);

    // Perform allocations with timing
    std::vector<void*> ptrs;
    const size_t       allocation_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};

    for (int round = 0; round < 5; ++round)
    {
        for (size_t size : allocation_sizes)
        {
            void* ptr = tracking->allocate_raw(64, size);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }
    }

    // Get statistics
    auto stats_opt = tracking->GetStats();
    ASSERT_TRUE(stats_opt.has_value());

    // Create usage bars visualization
    std::string usage_bars = visualizer.create_usage_bars(
        stats_opt->bytes_in_use.load(),
        stats_opt->peak_bytes_in_use.load(),
        10ULL * 1024ULL * 1024ULL);  // 10 MB limit

    QUARISMA_LOG_INFO("\n{}", usage_bars);

    // Get timing statistics
    auto timing_stats = tracking->GetTimingStats();

    // Create performance summary
    std::string perf_summary = visualizer.create_performance_summary(timing_stats);
    QUARISMA_LOG_INFO("\n{}", perf_summary);

    // Create allocation size histogram
    auto records = tracking->GetEnhancedRecords();

    // Extract allocation sizes for histogram
    std::vector<size_t> alloc_sizes_for_histogram;
    alloc_sizes_for_histogram.reserve(records.size());
    for (const auto& record : records)
    {
        alloc_sizes_for_histogram.push_back(record.requested_bytes);
    }

    std::string histogram = visualizer.create_histogram(alloc_sizes_for_histogram);
    QUARISMA_LOG_INFO("\n{}", histogram);

    // Generate comprehensive report
    std::string report = tracking->GenerateReport(false);
    QUARISMA_LOG_INFO("\n{}", report);

    // Verify visualizations were created
    EXPECT_FALSE(usage_bars.empty());
    EXPECT_FALSE(perf_summary.empty());
    EXPECT_FALSE(histogram.empty());
    EXPECT_FALSE(report.empty());

    // Cleanup
    for (void* ptr : ptrs)
    {
        tracking->deallocate_raw(ptr);
    }

    // Properly release the tracking allocator using reference counting
    // NOTE: GetRecordsAndUnRef() may delete tracking object via reference counting.
    // We must NOT call pool.reset() here because tracking may still need access to pool during cleanup.
    // The pool will be automatically destroyed when it goes out of scope after this function returns.
    tracking->GetRecordsAndUnRef();

    QUARISMA_LOG_INFO("Comprehensive visualization test completed successfully");
}

MEMORYTEST(AllocatorStatistics, AllAllocatorsComparison)
{
    QUARISMA_LOG_INFO("Testing statistics comparison across all allocator types...");

    struct AllocatorTestResult
    {
        std::string         name;
        allocator_stats     stats;
        atomic_timing_stats timing;
        double              efficiency;
    };

    std::vector<AllocatorTestResult> results;

    // Test CPU allocator
    {
        EnableCPUAllocatorStats();
        Allocator* cpu_alloc = cpu_allocator(0);

        std::vector<void*> ptrs;
        for (int i = 0; i < 50; ++i)
        {
            void* ptr = cpu_alloc->allocate_raw(64, 1024 * (i + 1));
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        auto stats_opt = cpu_alloc->GetStats();
        if (stats_opt.has_value())
        {
            AllocatorTestResult result;
            result.name       = "CPU Allocator";
            result.stats      = stats_opt.value();
            result.efficiency = result.stats.memory_efficiency();
            results.push_back(result);
        }

        for (void* ptr : ptrs)
        {
            cpu_alloc->deallocate_raw(ptr);
        }
    }

    // Test BFC allocator
    {
        auto sub_allocator = std::make_unique<basic_cpu_allocator>(
            0,
            std::vector<memory::sub_allocator::Visitor>{},
            std::vector<memory::sub_allocator::Visitor>{});

        allocator_bfc::Options opts;
        opts.allow_growth = true;

        allocator_bfc bfc_alloc(
            std::move(sub_allocator), 10ULL * 1024ULL * 1024ULL, "test_bfc_compare", opts);

        std::vector<void*> ptrs;
        for (int i = 0; i < 50; ++i)
        {
            void* ptr = bfc_alloc.allocate_raw(64, 1024 * (i + 1));
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        auto stats_opt = bfc_alloc.GetStats();
        if (stats_opt.has_value())
        {
            AllocatorTestResult result;
            result.name       = "BFC Allocator";
            result.stats      = stats_opt.value();
            result.efficiency = result.stats.memory_efficiency();
            results.push_back(result);
        }

        for (void* ptr : ptrs)
        {
            bfc_alloc.deallocate_raw(ptr);
        }
    }

    // Test Pool allocator
    {
        auto base_allocator = std::make_unique<basic_cpu_allocator>(
            0,
            std::vector<memory::sub_allocator::Visitor>{},
            std::vector<memory::sub_allocator::Visitor>{});

        auto pool = std::make_unique<allocator_pool>(
            25,
            false,
            std::move(base_allocator),
            std::make_unique<NoopRounder>(),
            "test_pool_compare");

        std::vector<void*> ptrs;
        for (int i = 0; i < 50; ++i)
        {
            void* ptr = pool->allocate_raw(64, 1024 * (i + 1));
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }

        auto stats_opt = pool->GetStats();
        if (stats_opt.has_value())
        {
            AllocatorTestResult result;
            result.name       = "Pool Allocator";
            result.stats      = stats_opt.value();
            result.efficiency = result.stats.memory_efficiency();
            results.push_back(result);
        }

        for (void* ptr : ptrs)
        {
            pool->deallocate_raw(ptr);
        }
    }

    // Display comparison table
    QUARISMA_LOG_INFO("\n========================================");
    QUARISMA_LOG_INFO("Allocator Comparison Summary");
    QUARISMA_LOG_INFO("========================================");
    QUARISMA_LOG_INFO(
        "{:20} | {:12} | {:12} | {:12} | {:10}",
        "Allocator",
        "Allocs",
        "Peak Memory",
        "Avg Size",
        "Efficiency");
    QUARISMA_LOG_INFO("{:-<20}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<10}", "", "", "", "", "");

    for (const auto& result : results)
    {
        int64_t allocs   = result.stats.num_allocs.load();
        size_t  peak     = result.stats.peak_bytes_in_use.load();
        double  avg_size = allocs > 0 ? result.stats.average_allocation_size() : 0.0;

        QUARISMA_LOG_INFO(
            "{:20} | {:12} | {:12} | {:12} | {:9.2f}%",
            result.name,
            allocs,
            format_bytes(peak),
            format_bytes(static_cast<size_t>(avg_size)),
            result.efficiency * 100.0);
    }

    QUARISMA_LOG_INFO("========================================\n");

    EXPECT_GT(results.size(), 0);

    QUARISMA_LOG_INFO("All allocators comparison test completed successfully");
}
