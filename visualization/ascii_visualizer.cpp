#include "visualization/ascii_visualizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

#include "backend/allocator_tracking.h"
#include "profiler/unified_memory_stats.h"

namespace memory
{

std::string ascii_visualizer::create_histogram(const std::vector<size_t>& allocation_sizes) const
{
    if (allocation_sizes.empty())
    {
        return "No allocation data available for histogram.\n";
    }

    // Create size buckets
    auto buckets = create_size_buckets(allocation_sizes);

    if (buckets.empty())
    {
        return "Unable to create histogram buckets.\n";
    }

    std::ostringstream hist;

    if (config_.show_legends)
    {
        hist << "Allocation Size Distribution:\n";
        hist << "Size Range          | Count     | Percentage | Histogram\n";
        hist << "--------------------|-----------|------------|";
        hist << std::string(config_.chart_width, '-') << "\n";
    }

    // Find maximum count for scaling
    size_t const max_count = std::max_element(
                                 buckets.begin(),
                                 buckets.end(),
                                 [](const auto& a, const auto& b) { return a.count < b.count; })
                                 ->count;

    for (const auto& bucket : buckets)
    {
        hist << std::setw(19) << std::left << format_size_range(bucket.min_size, bucket.max_size)
             << " | " << std::setw(9) << std::right << bucket.count << " | ";

        if (config_.show_percentages)
        {
            hist << std::setw(9) << std::fixed << std::setprecision(1) << bucket.percentage
                 << "% | ";
        }

        hist << create_bar(bucket.count, max_count, config_.chart_width) << "\n";
    }

    return hist.str();
}

std::string ascii_visualizer::create_timeline(
    const std::vector<timeline_point>& timeline_data) const
{
    if (timeline_data.empty())
    {
        return "No timeline data available.\n";
    }

    std::ostringstream timeline;

    if (config_.show_legends)
    {
        timeline << "Memory Usage Timeline:\n";
        timeline << "Time (s) | Memory (MB) | Operations\n";
        timeline << "---------|-------------|------------\n";
    }

    auto start_time = timeline_data[0].timestamp;

    for (const auto& point : timeline_data)
    {
        double const time_sec = std::chrono::duration<double>(point.timestamp - start_time).count();
        double const memory_mb = static_cast<double>(point.total_usage) / (1024.0 * 1024.0);

        timeline << std::fixed << std::setprecision(2) << std::setw(8) << time_sec << " | "
                 << std::setw(11) << std::setprecision(2) << memory_mb << " | "
                 << point.operation_type << " " << format_bytes(std::abs(point.size_delta)) << "\n";
    }

    return timeline.str();
}

std::string ascii_visualizer::create_fragmentation_map(
    const memory_fragmentation_metrics& metrics) const
{
    std::ostringstream map;

    if (config_.show_legends)
    {
        map << "Memory Fragmentation Analysis:\n";
        map << "External Fragmentation: " << std::fixed << std::setprecision(1)
            << (metrics.external_fragmentation * 100) << "%\n";
        map << "Internal Fragmentation: " << std::fixed << std::setprecision(1)
            << (metrics.internal_fragmentation * 100) << "%\n";
        map << "Largest Free Block: " << format_bytes(metrics.largest_free_block) << "\n";
        map << "Free Block Count: " << metrics.total_free_blocks << "\n";
        map << "Wasted Bytes: " << format_bytes(metrics.wasted_bytes) << "\n";
        map << "\n";
    }

    // Create memory layout visualization
    memory_layout layout;
    // Since the actual memory sizes aren't available in fragmentation metrics,
    // we'll use the available data to create a representative layout
    size_t const estimated_total = metrics.largest_free_block * 10;  // Rough estimate
    layout.total_size            = estimated_total;
    layout.used_size             = estimated_total - metrics.wasted_bytes;
    layout.free_size             = metrics.largest_free_block;
    layout.fragmented_size       = metrics.wasted_bytes;
    layout.block_size            = std::max(size_t{1024}, estimated_total / config_.chart_width);

    // Create usage map based on fragmentation data
    size_t const num_blocks = config_.chart_width;
    layout.usage_map.resize(num_blocks);

    // Simulate memory layout based on fragmentation metrics
    size_t const used_blocks       = (layout.used_size * num_blocks) / layout.total_size;
    size_t const fragmented_blocks = (layout.fragmented_size * num_blocks) / layout.total_size;

    // Fill usage map with pattern that reflects fragmentation
    for (size_t i = 0; i < num_blocks; ++i)
    {
        if (i < used_blocks)
        {
            // Create fragmentation pattern
            layout.usage_map[i] = (i % 4 != 3) || (fragmented_blocks == 0);
        }
        else
        {
            layout.usage_map[i] = false;
        }
    }

    if (config_.show_legends)
    {
        map << "Memory Layout (each '" << config_.filled_char
            << "' = " << format_bytes(layout.block_size) << "):\n";
        map << "Used: " << config_.filled_char << "  Free: " << config_.empty_char
            << "  Fragmented: " << config_.fragmented_char << "\n";
    }

    map << create_layout_visualization(layout) << "\n";

    return map.str();
}

std::string ascii_visualizer::create_usage_bars(
    size_t current_usage, size_t peak_usage, size_t limit_usage) const
{
    std::ostringstream bars;

    if (config_.show_legends)
    {
        bars << "Memory Usage Summary:\n";
    }

    size_t max_value = std::max({current_usage, peak_usage, limit_usage});
    if (max_value == 0)
    {
        max_value = 1;  // Avoid division by zero
    }

    bars << "Current: " << std::setw(10) << format_bytes(current_usage) << " | "
         << create_bar(current_usage, max_value, config_.chart_width) << "\n";

    bars << "Peak:    " << std::setw(10) << format_bytes(peak_usage) << " | "
         << create_bar(peak_usage, max_value, config_.chart_width) << "\n";

    if (limit_usage > 0)
    {
        bars << "Limit:   " << std::setw(10) << format_bytes(limit_usage) << " | "
             << create_bar(limit_usage, max_value, config_.chart_width) << "\n";

        if (config_.show_percentages)
        {
            double const usage_percent = (static_cast<double>(current_usage) / limit_usage) * 100.0;
            bars << "Usage: " << std::fixed << std::setprecision(1) << usage_percent
                 << "% of limit\n";
        }
    }

    return bars.str();
}

std::string ascii_visualizer::create_performance_summary(
    const allocation_timing_stats& timing_stats) const
{
    std::ostringstream summary;

    if (config_.show_legends)
    {
        summary << "Allocation Performance Summary:\n";
        summary << "Metric                    | Value\n";
        summary << "--------------------------|------------------\n";
    }

    summary << "Total Allocations         | " << timing_stats.total_allocations.load() << "\n";
    summary << "Total Deallocations       | " << timing_stats.total_deallocations.load() << "\n";

    // Calculate average times
    uint64_t const total_allocs   = timing_stats.total_allocations.load();
    uint64_t const total_deallocs = timing_stats.total_deallocations.load();
    double const   avg_alloc_time =
        (total_allocs > 0)
              ? static_cast<double>(timing_stats.total_alloc_time_us.load()) / total_allocs
              : 0.0;
    double const avg_dealloc_time =
        (total_deallocs > 0)
            ? static_cast<double>(timing_stats.total_dealloc_time_us.load()) / total_deallocs
            : 0.0;

    summary << "Average Alloc Time        | " << std::fixed << std::setprecision(2)
            << avg_alloc_time << " μs\n";
    summary << "Average Dealloc Time      | " << std::fixed << std::setprecision(2)
            << avg_dealloc_time << " μs\n";
    summary << "Peak Alloc Time           | " << std::fixed << std::setprecision(2)
            << timing_stats.max_alloc_time_us.load() << " μs\n";
    summary << "Peak Dealloc Time         | " << std::fixed << std::setprecision(2)
            << timing_stats.max_dealloc_time_us.load() << " μs\n";

    return summary.str();
}

std::vector<ascii_visualizer::timeline_point> ascii_visualizer::convert_records_to_timeline(
    const std::vector<alloc_record>& records)
{
    std::vector<timeline_point> timeline_data;
    timeline_data.reserve(records.size());

    size_t current_usage = 0;

    for (const auto& record : records)
    {
        timeline_point point;
        point.timestamp  = std::chrono::microseconds(record.alloc_micros);
        point.size_delta = record.alloc_bytes;

        current_usage += record.alloc_bytes;
        point.total_usage = current_usage;

        if (record.alloc_bytes > 0)
        {
            point.operation_type = "ALLOC";
        }
        else if (record.alloc_bytes < 0)
        {
            point.operation_type = "FREE";
        }
        else
        {
            point.operation_type = "NOOP";
        }

        timeline_data.push_back(point);
    }

    return timeline_data;
}

std::vector<ascii_visualizer::size_bucket> ascii_visualizer::create_size_buckets(
    const std::vector<size_t>& allocation_sizes, size_t num_buckets)
{
    if (allocation_sizes.empty())
    {
        return {};
    }

    auto         min_max  = std::minmax_element(allocation_sizes.begin(), allocation_sizes.end());
    size_t const min_size = *min_max.first;
    size_t const max_size = *min_max.second;

    if (min_size == max_size)
    {
        // All allocations are the same size
        size_bucket bucket;
        bucket.min_size   = min_size;
        bucket.max_size   = max_size;
        bucket.count      = allocation_sizes.size();
        bucket.percentage = 100.0;
        return {bucket};
    }

    // Auto-determine number of buckets if not specified
    if (num_buckets == 0)
    {
        num_buckets = std::min(size_t{20}, static_cast<size_t>(std::sqrt(allocation_sizes.size())));
        num_buckets = std::max(num_buckets, size_t{5});
    }

    auto boundaries = calculate_bucket_boundaries(min_size, max_size, num_buckets);

    std::vector<size_bucket> buckets(boundaries.size() - 1);

    size_t i = 0;
    // Initialize buckets
    for (auto& bucket : buckets)
    {
        bucket.min_size   = boundaries[i];
        bucket.max_size   = boundaries[i + 1] - 1;
        bucket.count      = 0;
        bucket.percentage = 0.0;
        i++;
    }

    // Count allocations in each bucket
    for (size_t size : allocation_sizes)
    {
        // Find appropriate bucket
        // for (auto & bucket : buckets)
        // {
        //     if (size >= bucket.min_size && size <= bucket.max_size)
        //     {
        //         bucket.count++;
        //         break;
        //     }
        // }
        std::for_each(
            buckets.begin(),
            buckets.end(),
            [size](auto& bucket)
            {
                if (size >= bucket.min_size && size <= bucket.max_size)
                {
                    bucket.count++;
                }
            });
    }

    // Calculate percentages
    size_t const total_count = allocation_sizes.size();
    for (auto& bucket : buckets)
    {
        bucket.percentage = (static_cast<double>(bucket.count) / total_count) * 100.0;
    }

    // Remove empty buckets
    buckets.erase(
        std::remove_if(
            buckets.begin(),
            buckets.end(),
            [](const size_bucket& bucket) { return bucket.count == 0; }),
        buckets.end());

    return buckets;
}

std::string ascii_visualizer::create_bar(size_t value, size_t max_value, size_t width) const
{
    if (max_value == 0)
    {
        return std::string(width, config_.empty_char);  // NOLINT(modernize-return-braced-init-list)
    }

    size_t const filled_length = (value * width) / max_value;
    size_t const empty_length  = width - filled_length;

    return std::string(filled_length, config_.filled_char) +
           std::string(empty_length, config_.empty_char);
}

std::string ascii_visualizer::format_bytes(size_t bytes)
{
    const char* units[]    = {"B", "KB", "MB", "GB", "TB"};  //NOLINT
    size_t      unit_index = 0;
    auto        size       = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_index < 4)
    {
        size /= 1024.0;
        unit_index++;
    }

    std::ostringstream formatted;
    if (size < 10.0)
    {
        formatted << std::fixed << std::setprecision(1) << size << " " << units[unit_index];
    }
    else
    {
        formatted << std::fixed << std::setprecision(0) << size << " " << units[unit_index];
    }

    return formatted.str();
}

std::string ascii_visualizer::format_size_range(size_t min_size, size_t max_size)
{
    return format_bytes(min_size) + " - " + format_bytes(max_size);
}

std::vector<size_t> ascii_visualizer::calculate_bucket_boundaries(
    size_t min_size, size_t max_size, size_t num_buckets)
{
    std::vector<size_t> boundaries;
    boundaries.reserve(num_buckets + 1);

    // Use logarithmic scale for better distribution
    double const log_min  = std::log2(static_cast<double>(min_size));
    double const log_max  = std::log2(static_cast<double>(max_size));
    double const log_step = (log_max - log_min) / num_buckets;

    for (size_t i = 0; i <= num_buckets; ++i)
    {
        double const log_value = log_min + (i * log_step);
        auto const   boundary  = static_cast<size_t>(std::pow(2.0, log_value));
        boundaries.push_back(boundary);
    }

    // Ensure last boundary includes max_size
    boundaries.back() = max_size + 1;

    return boundaries;
}

std::string ascii_visualizer::create_layout_visualization(const memory_layout& layout) const
{
    std::ostringstream vis;

    for (size_t i = 0; i < layout.usage_map.size(); ++i)
    {
        if (layout.usage_map[i])
        {
            // Check if this should be shown as fragmented
            bool is_fragmented = false;
            if (i > 0 && i < layout.usage_map.size() - 1)
            {
                // Show as fragmented if surrounded by free blocks
                is_fragmented = !layout.usage_map[i - 1] || !layout.usage_map[i + 1];
            }

            vis << (is_fragmented ? config_.fragmented_char : config_.filled_char);
        }
        else
        {
            vis << config_.empty_char;
        }
    }

    return vis.str();
}

}  // namespace memory
