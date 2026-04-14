#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "backend/allocator_tracking.h"
#include "cpu/allocator.h"
#include "profiler/unified_memory_stats.h"
#include "common/memory_export.h"

namespace memory
{

/**
 * @brief ASCII-based visualization for memory allocation statistics
 * 
 * Provides text-based charts and graphs for memory allocation patterns,
 * fragmentation analysis, and performance metrics. Designed for console
 * output and logging systems.
 * 
 * **Thread Safety**: Not thread-safe - external synchronization required
 * **Performance**: O(n) where n is the number of data points
 */
class MEMORY_VISIBILITY ascii_visualizer
{
public:
    /**
     * @brief Configuration for visualization appearance
     */
    struct visualization_config
    {
        size_t chart_width          = 60;    ///< Width of charts in characters
        size_t max_histogram_height = 20;    ///< Maximum height of histograms
        char   filled_char          = '#';   ///< Character for filled areas
        char   empty_char           = '.';   ///< Character for empty areas
        char   fragmented_char      = 'X';   ///< Character for fragmented areas
        bool   show_percentages     = true;  ///< Show percentage values
        bool   show_legends         = true;  ///< Show chart legends
    };

    /**
     * @brief Allocation size bucket for histogram generation
     */
    struct size_bucket
    {
        size_t min_size;    ///< Minimum size in bucket
        size_t max_size;    ///< Maximum size in bucket
        size_t count;       ///< Number of allocations in bucket
        double percentage;  ///< Percentage of total allocations
    };

    /**
     * @brief Timeline data point for memory usage over time
     */
    struct timeline_point
    {
        std::chrono::microseconds timestamp;    ///< Time of allocation/deallocation
        int64_t                   size_delta;   ///< Size change (positive=alloc, negative=free)
        size_t                    total_usage;  ///< Total memory usage at this point
        std::string operation_type;             ///< Type of operation ("ALLOC", "FREE", "REALLOC")
    };

    /**
     * @brief Memory layout visualization data
     */
    struct memory_layout
    {
        size_t            total_size;       ///< Total memory size
        size_t            used_size;        ///< Currently used memory
        size_t            free_size;        ///< Free memory
        size_t            fragmented_size;  ///< Fragmented memory
        std::vector<bool> usage_map;        ///< Per-block usage (true=used, false=free)
        size_t            block_size;       ///< Size per block in usage_map
    };

public:
    /**
     * @brief Construct ASCII visualizer with default configuration
     */
    ascii_visualizer() = default;

    /**
     * @brief Construct ASCII visualizer with custom configuration
     * @param config Visualization configuration
     */
    explicit ascii_visualizer(const visualization_config& config) : config_(config) {}

    /**
     * @brief Create histogram of allocation sizes
     * @param allocation_sizes Vector of allocation sizes in bytes
     * @return ASCII histogram as string
     * 
     * Creates a histogram showing the distribution of allocation sizes
     * with logarithmic bucketing for better visualization of wide ranges.
     * 
     * **Example Output**:
     * ```
     * Allocation Size Distribution:
     * Size Range          | Count     | Histogram
     * --------------------|-----------|----------------------------------
     * 256B - 511B         |       150 | ████████████████████████████████
     * 512B - 1023B        |        75 | ████████████████
     * 1KB - 2KB           |        30 | ██████
     * ```
     */
    MEMORY_API std::string create_histogram(const std::vector<size_t>& allocation_sizes) const;

    /**
     * @brief Create timeline visualization of memory usage
     * @param timeline_data Vector of timeline points
     * @return ASCII timeline as string
     * 
     * Shows memory usage over time with allocation/deallocation events.
     * 
     * **Example Output**:
     * ```
     * Memory Usage Timeline:
     * Time (s) | Memory (MB) | Operations
     * ---------|-------------|------------
     *     0.00 |        0.00 | ALLOC 1024 bytes
     *     0.15 |        1.50 | ALLOC 512000 bytes
     *     0.30 |        1.00 | FREE 512000 bytes
     * ```
     */
    MEMORY_API std::string create_timeline(
        const std::vector<timeline_point>& timeline_data) const;

    /**
     * @brief Create fragmentation analysis visualization
     * @param fragmentation_metrics Memory fragmentation data
     * @return ASCII fragmentation map as string
     * 
     * Visualizes memory fragmentation with layout map and statistics.
     * 
     * **Example Output**:
     * ```
     * Memory Fragmentation Analysis:
     * External Fragmentation: 15.2%
     * Internal Fragmentation: 8.7%
     * Largest Free Block: 2.5 MB
     * 
     * Memory Layout (each '.' = 64KB):
     * Used: █  Free: ░  Fragmented: ▓
     * ████░░▓▓████░░░░████▓▓░░████
     * ```
     */
    MEMORY_API std::string create_fragmentation_map(
        const memory_fragmentation_metrics& metrics) const;

    /**
     * @brief Create memory usage bar chart
     * @param current_usage Current memory usage in bytes
     * @param peak_usage Peak memory usage in bytes
     * @param limit_usage Memory limit in bytes (0 = no limit)
     * @return ASCII bar chart as string
     * 
     * Shows current vs peak vs limit memory usage as horizontal bars.
     */
    MEMORY_API std::string create_usage_bars(
        size_t current_usage, size_t peak_usage, size_t limit_usage = 0) const;

    /**
     * @brief Create performance metrics summary
     * @param timing_stats Allocation timing statistics
     * @return ASCII performance summary as string
     * 
     * Displays allocation performance metrics in tabular format.
     */
    MEMORY_API std::string create_performance_summary(
        const allocation_timing_stats& timing_stats) const;

    /**
     * @brief Convert allocation records to timeline data
     * @param records Vector of allocation records
     * @return Timeline data suitable for create_timeline()
     * 
     * Helper function to convert allocator tracking records into
     * timeline visualization format.
     */
    static MEMORY_API std::vector<timeline_point> convert_records_to_timeline(
        const std::vector<alloc_record>& records);

    /**
     * @brief Create size buckets from allocation sizes
     * @param allocation_sizes Vector of allocation sizes
     * @param num_buckets Number of buckets to create (default: auto-determine)
     * @return Vector of size buckets for histogram
     * 
     * Creates logarithmic size buckets for histogram visualization.
     */
    static MEMORY_API std::vector<size_bucket> create_size_buckets(
        const std::vector<size_t>& allocation_sizes, size_t num_buckets = 0);

    /**
     * @brief Set visualization configuration
     * @param config New configuration
     */
    void set_config(const visualization_config& config) { config_ = config; }

    /**
     * @brief Get current visualization configuration
     * @return Current configuration
     */
    const visualization_config& get_config() const { return config_; }

private:
    visualization_config config_;

    /**
     * @brief Create horizontal bar for charts
     * @param value Current value
     * @param max_value Maximum value for scaling
     * @param width Bar width in characters
     * @return ASCII bar string
     */
    std::string create_bar(size_t value, size_t max_value, size_t width) const;

    /**
     * @brief Format byte size with appropriate units
     * @param bytes Size in bytes
     * @return Formatted string (e.g., "1.5 MB")
     */
    static std::string format_bytes(size_t bytes);

    /**
     * @brief Format size range for histogram buckets
     * @param min_size Minimum size in bucket
     * @param max_size Maximum size in bucket
     * @return Formatted range string
     */
    static std::string format_size_range(size_t min_size, size_t max_size);

    /**
     * @brief Calculate logarithmic bucket boundaries
     * @param min_size Minimum allocation size
     * @param max_size Maximum allocation size
     * @param num_buckets Number of buckets to create
     * @return Vector of bucket boundaries
     */
    static std::vector<size_t> calculate_bucket_boundaries(
        size_t min_size, size_t max_size, size_t num_buckets);

    /**
     * @brief Create memory layout visualization string
     * @param layout Memory layout data
     * @return ASCII layout visualization
     */
    std::string create_layout_visualization(const memory_layout& layout) const;
};

}  // namespace memory
