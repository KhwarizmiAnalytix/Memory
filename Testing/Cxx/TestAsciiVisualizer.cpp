/**
 * @file TestAsciiVisualizer.cpp
 * @brief Comprehensive test suite for ASCII visualization components
 *
 * Tests all public methods and functionality of the ascii_visualizer class including:
 * - Histogram generation with allocation size distribution
 * - Timeline visualization of memory usage over time
 * - Fragmentation analysis and memory layout maps
 * - Usage bar charts for current vs peak memory
 * - Performance summary displays
 * - Configuration options and customization
 * - Demonstration of visual output capabilities
 */

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#include "MemoryTest.h"
#include "helper/memory_allocator.h"
#include "profiler/unified_memory_stats.h"
#include "visualization/ascii_visualizer.h"

using namespace memory;
using namespace memory;

namespace
{

/**
 * @brief Helper function to create sample allocation sizes for testing
 */
std::vector<size_t> create_sample_allocation_sizes()
{
    return {256,  512, 1024,  2048, 4096, 1024, 512,  256, 8192,  1024,
            2048, 512, 16384, 1024, 256,  4096, 1024, 512, 32768, 2048};
}

/**
 * @brief Helper function to create sample timeline data
 */
std::vector<ascii_visualizer::timeline_point> create_sample_timeline()
{
    std::vector<ascii_visualizer::timeline_point> timeline;
    auto                                          start_time = std::chrono::steady_clock::now();

    ascii_visualizer::timeline_point point1;
    point1.timestamp      = std::chrono::microseconds(0);
    point1.size_delta     = 1024;
    point1.total_usage    = 1024;
    point1.operation_type = "ALLOC";
    timeline.push_back(point1);

    ascii_visualizer::timeline_point point2;
    point2.timestamp      = std::chrono::microseconds(100000);  // 100ms in microseconds
    point2.size_delta     = 2048;
    point2.total_usage    = 3072;
    point2.operation_type = "ALLOC";
    timeline.push_back(point2);

    ascii_visualizer::timeline_point point3;
    point3.timestamp      = std::chrono::microseconds(200000);  // 200ms in microseconds
    point3.size_delta     = 4096;
    point3.total_usage    = 7168;
    point3.operation_type = "ALLOC";
    timeline.push_back(point3);

    ascii_visualizer::timeline_point point4;
    point4.timestamp      = std::chrono::microseconds(300000);  // 300ms in microseconds
    point4.size_delta     = -1024;
    point4.total_usage    = 6144;
    point4.operation_type = "FREE";
    timeline.push_back(point4);

    ascii_visualizer::timeline_point point5;
    point5.timestamp      = std::chrono::microseconds(400000);  // 400ms in microseconds
    point5.size_delta     = -2048;
    point5.total_usage    = 4096;
    point5.operation_type = "FREE";
    timeline.push_back(point5);

    return timeline;
}

/**
 * @brief Helper function to create sample fragmentation metrics
 */
memory_fragmentation_metrics create_sample_fragmentation_metrics()
{
    memory_fragmentation_metrics metrics;
    metrics.total_free_blocks       = 15;
    metrics.largest_free_block      = 8192;
    metrics.smallest_free_block     = 256;
    metrics.average_free_block_size = 1024;
    metrics.fragmentation_ratio     = 0.25;
    metrics.external_fragmentation  = 0.15;
    metrics.internal_fragmentation  = 0.08;
    metrics.wasted_bytes            = 2048;
    return metrics;
}

/**
 * @brief Helper function to create sample timing statistics
 */
allocation_timing_stats create_sample_timing_stats()
{
    allocation_timing_stats stats;
    stats.total_allocations.store(1000);
    stats.total_deallocations.store(800);
    stats.total_alloc_time_us.store(50000);
    stats.total_dealloc_time_us.store(30000);
    stats.min_alloc_time_us.store(10);
    stats.max_alloc_time_us.store(500);
    stats.min_dealloc_time_us.store(5);
    stats.max_dealloc_time_us.store(200);
    return stats;
}

}  // anonymous namespace

/**
 * @brief Helper function to create a default ASCII visualizer for testing
 */
std::unique_ptr<ascii_visualizer> create_test_visualizer()
{
    return std::make_unique<ascii_visualizer>();
}

/**
 * @brief Helper function to create a custom ASCII visualizer for testing
 */
std::unique_ptr<ascii_visualizer> create_custom_test_visualizer()
{
    ascii_visualizer::visualization_config custom_config;
    custom_config.chart_width          = 40;
    custom_config.max_histogram_height = 15;
    custom_config.filled_char          = '#';
    custom_config.empty_char           = '.';
    custom_config.fragmented_char      = 'X';
    custom_config.show_percentages     = true;
    custom_config.show_legends         = true;

    return std::make_unique<ascii_visualizer>(custom_config);
}

/**
 * @brief Test histogram creation with allocation size distribution
 */
MEMORYTEST(AsciiVisualizer, create_histogram)
{
    auto visualizer        = create_test_visualizer();
    auto custom_visualizer = create_custom_test_visualizer();
    auto allocation_sizes  = create_sample_allocation_sizes();

    // Create histogram
    std::string histogram = visualizer->create_histogram(allocation_sizes);

    // Verify histogram is not empty
    EXPECT_FALSE(histogram.empty());

    // Verify histogram contains expected elements
    EXPECT_NE(histogram.find("Allocation Size Distribution"), std::string::npos);
    EXPECT_NE(histogram.find("Size Range"), std::string::npos);
    EXPECT_NE(histogram.find("Count"), std::string::npos);
    EXPECT_NE(histogram.find("Histogram"), std::string::npos);

    // Display histogram for visual inspection
    std::cout << "\n=== HISTOGRAM DEMONSTRATION ===\n";
    std::cout << histogram << std::endl;

    // Test with custom configuration
    std::string custom_histogram = custom_visualizer->create_histogram(allocation_sizes);
    EXPECT_FALSE(custom_histogram.empty());

    std::cout << "\n=== CUSTOM HISTOGRAM DEMONSTRATION ===\n";
    std::cout << custom_histogram << std::endl;

    END_TEST();
}

/**
 * @brief Test timeline visualization of memory usage over time
 */
MEMORYTEST(AsciiVisualizer, create_timeline)
{
    auto visualizer    = create_test_visualizer();
    auto timeline_data = create_sample_timeline();

    // Create timeline visualization
    std::string timeline = visualizer->create_timeline(timeline_data);

    // Verify timeline is not empty
    EXPECT_FALSE(timeline.empty());

    // Verify timeline contains expected elements
    EXPECT_NE(timeline.find("Memory Usage Timeline"), std::string::npos);
    EXPECT_NE(timeline.find("Time"), std::string::npos);
    EXPECT_NE(timeline.find("Memory"), std::string::npos);
    EXPECT_NE(timeline.find("Operations"), std::string::npos);
    EXPECT_NE(timeline.find("ALLOC"), std::string::npos);
    EXPECT_NE(timeline.find("FREE"), std::string::npos);

    // Display timeline for visual inspection
    std::cout << "\n=== TIMELINE DEMONSTRATION ===\n";
    std::cout << timeline << std::endl;

    END_TEST();
}

/**
 * @brief Test fragmentation analysis and memory layout visualization
 */
MEMORYTEST(AsciiVisualizer, create_fragmentation_map)
{
    auto visualizer            = create_test_visualizer();
    auto custom_visualizer     = create_custom_test_visualizer();
    auto fragmentation_metrics = create_sample_fragmentation_metrics();

    // Create fragmentation map
    std::string fragmentation_map = visualizer->create_fragmentation_map(fragmentation_metrics);

    // Verify fragmentation map is not empty
    EXPECT_FALSE(fragmentation_map.empty());

    // Verify fragmentation map contains expected elements
    EXPECT_NE(fragmentation_map.find("Memory Fragmentation Analysis"), std::string::npos);
    EXPECT_NE(fragmentation_map.find("External Fragmentation"), std::string::npos);
    EXPECT_NE(fragmentation_map.find("Internal Fragmentation"), std::string::npos);
    EXPECT_NE(fragmentation_map.find("Free Block Count"), std::string::npos);
    EXPECT_NE(fragmentation_map.find("Memory Layout"), std::string::npos);

    // Display fragmentation map for visual inspection
    std::cout << "\n=== FRAGMENTATION MAP DEMONSTRATION ===\n";
    std::cout << fragmentation_map << std::endl;

    // Test with custom configuration
    std::string custom_fragmentation_map =
        custom_visualizer->create_fragmentation_map(fragmentation_metrics);
    EXPECT_FALSE(custom_fragmentation_map.empty());

    std::cout << "\n=== CUSTOM FRAGMENTATION MAP DEMONSTRATION ===\n";
    std::cout << custom_fragmentation_map << std::endl;

    END_TEST();
}

/**
 * @brief Test memory usage bar charts
 */
MEMORYTEST(AsciiVisualizer, create_usage_bars)
{
    auto   visualizer    = create_test_visualizer();
    size_t current_usage = 512 * 1024ULL;      // 512 MB
    size_t peak_usage    = 1024ULL * 1024;     // 1 GB
    size_t limit_usage   = 2048ULL * 1024ULL;  // 2 GB

    // Create usage bars with limit
    std::string usage_bars_with_limit =
        visualizer->create_usage_bars(current_usage, peak_usage, limit_usage);

    // Verify usage bars are not empty
    EXPECT_FALSE(usage_bars_with_limit.empty());

    // Verify usage bars contain expected elements
    EXPECT_NE(usage_bars_with_limit.find("Memory Usage Summary"), std::string::npos);
    EXPECT_NE(usage_bars_with_limit.find("Current"), std::string::npos);
    EXPECT_NE(usage_bars_with_limit.find("Peak"), std::string::npos);
    EXPECT_NE(usage_bars_with_limit.find("Limit"), std::string::npos);

    // Display usage bars for visual inspection
    std::cout << "\n=== USAGE BARS WITH LIMIT DEMONSTRATION ===\n";
    std::cout << usage_bars_with_limit << std::endl;

    // Create usage bars without limit
    std::string usage_bars_no_limit = visualizer->create_usage_bars(current_usage, peak_usage);

    EXPECT_FALSE(usage_bars_no_limit.empty());
    EXPECT_NE(usage_bars_no_limit.find("Current"), std::string::npos);
    EXPECT_NE(usage_bars_no_limit.find("Peak"), std::string::npos);

    std::cout << "\n=== USAGE BARS WITHOUT LIMIT DEMONSTRATION ===\n";
    std::cout << usage_bars_no_limit << std::endl;

    END_TEST();
}

/**
 * @brief Test performance summary display
 */
MEMORYTEST(AsciiVisualizer, create_performance_summary)
{
    auto visualizer   = create_test_visualizer();
    auto timing_stats = create_sample_timing_stats();

    // Create performance summary
    std::string performance_summary = visualizer->create_performance_summary(timing_stats);

    // Verify performance summary is not empty
    EXPECT_FALSE(performance_summary.empty());

    // Verify performance summary contains expected elements
    EXPECT_NE(performance_summary.find("Performance Summary"), std::string::npos);
    EXPECT_NE(performance_summary.find("Total Allocations"), std::string::npos);
    EXPECT_NE(performance_summary.find("Total Deallocations"), std::string::npos);
    EXPECT_NE(performance_summary.find("Average"), std::string::npos);
    //EXPECT_NE(performance_summary.find("Min"), std::string::npos);
    //EXPECT_NE(performance_summary.find("Max"), std::string::npos);

    // Display performance summary for visual inspection
    std::cout << "\n=== PERFORMANCE SUMMARY DEMONSTRATION ===\n";
    std::cout << performance_summary << std::endl;

    END_TEST();
}

/**
 * @brief Test size bucket creation for histograms
 */
MEMORYTEST(AsciiVisualizer, create_size_buckets)
{
    auto visualizer       = create_test_visualizer();
    auto allocation_sizes = create_sample_allocation_sizes();

    // Create size buckets with default number
    auto buckets = visualizer->create_size_buckets(allocation_sizes);

    // Verify buckets were created
    EXPECT_FALSE(buckets.empty());
    EXPECT_LE(buckets.size(), 10);  // Should be reasonable number of buckets

    // Verify bucket properties
    for (const auto& bucket : buckets)
    {
        EXPECT_LE(bucket.min_size, bucket.max_size);
        EXPECT_GE(bucket.count, 0);
        EXPECT_GE(bucket.percentage, 0.0);
        EXPECT_LE(bucket.percentage, 100.0);
    }

    // Test with specific number of buckets
    auto custom_buckets = visualizer->create_size_buckets(allocation_sizes, 5);
    EXPECT_EQ(custom_buckets.size(), 5);

    // Display bucket information
    std::cout << "\n=== SIZE BUCKETS DEMONSTRATION ===\n";
    for (auto& bucket : buckets)
    {
        std::cout << "Bucket: " << bucket.min_size << "-" << bucket.max_size
                  << " bytes, count=" << bucket.count << ", percentage=" << bucket.percentage
                  << "%\n";
    }

    END_TEST();
}

/**
 * @brief Test configuration management
 */
MEMORYTEST(AsciiVisualizer, configuration_management)
{
    auto visualizer = create_test_visualizer();

    // Test getting default configuration
    auto default_config = visualizer->get_config();
    EXPECT_EQ(default_config.chart_width, 60);
    EXPECT_EQ(default_config.max_histogram_height, 20);
    EXPECT_EQ(default_config.filled_char, '#');
    EXPECT_EQ(default_config.empty_char, '.');
    EXPECT_EQ(default_config.fragmented_char, 'X');
    EXPECT_TRUE(default_config.show_percentages);
    EXPECT_TRUE(default_config.show_legends);

    // Test setting custom configuration
    ascii_visualizer::visualization_config new_config;
    new_config.chart_width          = 80;
    new_config.max_histogram_height = 25;
    new_config.filled_char          = '*';
    new_config.empty_char           = '-';
    new_config.fragmented_char      = '?';
    new_config.show_percentages     = false;
    new_config.show_legends         = false;

    visualizer->set_config(new_config);

    // Verify configuration was updated
    auto updated_config = visualizer->get_config();
    EXPECT_EQ(updated_config.chart_width, 80);
    EXPECT_EQ(updated_config.max_histogram_height, 25);
    EXPECT_EQ(updated_config.filled_char, '*');
    EXPECT_EQ(updated_config.empty_char, '-');
    EXPECT_EQ(updated_config.fragmented_char, '?');
    EXPECT_FALSE(updated_config.show_percentages);
    EXPECT_FALSE(updated_config.show_legends);

    END_TEST();
}

/**
 * @brief Test edge cases and error handling
 */
MEMORYTEST(AsciiVisualizer, edge_cases_and_error_handling)
{
    auto visualizer = create_test_visualizer();

    // Test with empty allocation sizes
    std::vector<size_t> empty_sizes;
    std::string         empty_histogram = visualizer->create_histogram(empty_sizes);
    EXPECT_FALSE(empty_histogram.empty());  // Should handle gracefully

    // Test with single allocation size
    std::vector<size_t> single_size      = {1024};
    std::string         single_histogram = visualizer->create_histogram(single_size);
    EXPECT_FALSE(single_histogram.empty());

    // Test with empty timeline
    std::vector<ascii_visualizer::timeline_point> empty_timeline;
    std::string empty_timeline_str = visualizer->create_timeline(empty_timeline);
    EXPECT_FALSE(empty_timeline_str.empty());  // Should handle gracefully

    // Test with zero usage values
    std::string zero_usage_bars = visualizer->create_usage_bars(0, 0, 0);
    EXPECT_FALSE(zero_usage_bars.empty());

    // Test with zero timing stats
    allocation_timing_stats zero_stats;
    zero_stats.total_allocations.store(0);
    zero_stats.total_deallocations.store(0);
    zero_stats.total_alloc_time_us.store(0);
    zero_stats.total_dealloc_time_us.store(0);
    zero_stats.min_alloc_time_us.store(0);
    zero_stats.max_alloc_time_us.store(0);
    zero_stats.min_dealloc_time_us.store(0);
    zero_stats.max_dealloc_time_us.store(0);

    std::string zero_performance = visualizer->create_performance_summary(zero_stats);
    EXPECT_FALSE(zero_performance.empty());

    END_TEST();
}

/**
 * @brief Test comprehensive visualization demonstration
 */
TEST(AsciiVisualizerDemo, comprehensive_demonstration)
{
    ascii_visualizer visualizer;

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "COMPREHENSIVE ASCII VISUALIZATION DEMONSTRATION\n";
    std::cout << std::string(80, '=') << "\n";

    // Create realistic sample data
    std::vector<size_t> realistic_sizes = {64,    128,  256,  512, 1024, 2048,  4096, 8192,
                                           1024,  512,  256,  128, 64,   2048,  1024, 512,
                                           4096,  1024, 256,  512, 8192, 2048,  1024, 256,
                                           16384, 4096, 1024, 512, 256,  32768, 8192, 2048};

    // Demonstrate histogram
    std::cout << "\n1. ALLOCATION SIZE HISTOGRAM:\n";
    std::cout << visualizer.create_histogram(realistic_sizes);

    // Demonstrate timeline
    auto timeline = create_sample_timeline();
    std::cout << "\n2. MEMORY USAGE TIMELINE:\n";
    std::cout << visualizer.create_timeline(timeline);

    // Demonstrate fragmentation map
    auto fragmentation = create_sample_fragmentation_metrics();
    std::cout << "\n3. FRAGMENTATION ANALYSIS:\n";
    std::cout << visualizer.create_fragmentation_map(fragmentation);

    // Demonstrate usage bars
    std::cout << "\n4. MEMORY USAGE BARS:\n";
    std::cout << visualizer.create_usage_bars(
        768 * 1024ULL,     // 768 MB current
        1536 * 1024ULL,    // 1.5 GB peak
        2048ULL * 1024ULL  // 2 GB limit
    );

    // Demonstrate performance summary
    auto timing = create_sample_timing_stats();
    std::cout << "\n5. PERFORMANCE SUMMARY:\n";
    std::cout << visualizer.create_performance_summary(timing);

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "END OF DEMONSTRATION\n";
    std::cout << std::string(80, '=') << "\n";

    // Test passes if all visualizations were generated without errors
    EXPECT_TRUE(true);
}
