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

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/memory_macros.h"
#include "common/device.h"
#include "profiler/unified_memory_stats.h"
#include "common/memory_export.h"

namespace memory
{
namespace gpu
{

/**
 * @brief GPU resource allocation information
 *
 * Contains detailed information about a GPU resource allocation
 * including timing, location, and usage statistics for debugging
 * and performance analysis.
 */
struct MEMORY_VISIBILITY gpu_allocation_info
{
    /** @brief Unique allocation ID */
    size_t allocation_id = 0;

    /** @brief Pointer to allocated memory */
    void* ptr = nullptr;

    /** @brief Size of allocation in bytes */
    size_t size = 0;

    /** @brief Device where memory is allocated */
    device_option device{device_enum::CPU, 0};

    /** @brief Allocation timestamp */
    std::chrono::high_resolution_clock::time_point allocation_time;

    /** @brief Deallocation timestamp (if deallocated) */
    std::chrono::high_resolution_clock::time_point deallocation_time;

    /** @brief Whether allocation is still active */
    bool is_active = true;

    /** @brief Source file where allocation occurred */
    std::string source_file;

    /** @brief Line number where allocation occurred */
    int source_line = 0;

    /** @brief Function name where allocation occurred */
    std::string function_name;

    /** @brief Call stack trace (if available) */
    std::vector<std::string> call_stack;

    /** @brief Number of times this allocation was accessed */
    std::atomic<size_t> access_count{0};

    /** @brief Last access timestamp */
    std::chrono::high_resolution_clock::time_point last_access_time;

    /** @brief User-defined tag for categorization */
    std::string tag;

    /**
     * @brief Get allocation lifetime in milliseconds
     * @return Lifetime in milliseconds (0 if still active)
     */
    double get_lifetime_ms() const
    {
        if (!is_active)
        {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                deallocation_time - allocation_time);
            return duration.count() / 1000.0;
        }
        return 0.0;
    }

    /**
     * @brief Get time since last access in milliseconds
     * @return Time since last access in milliseconds
     */
    double get_time_since_last_access_ms() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(now - last_access_time);
        return duration.count() / 1000.0;
    }
};

/**
 * @brief Memory leak detection configuration
 */
struct MEMORY_VISIBILITY leak_detection_config
{
    /** @brief Enable leak detection */
    bool enabled = true;

    /** @brief Threshold for considering allocation as potential leak (in milliseconds) */
    double leak_threshold_ms = 60000.0;  // 1 minute

    /** @brief Maximum number of call stack frames to capture */
    size_t max_call_stack_depth = 10;

    /** @brief Enable periodic leak scanning */
    bool enable_periodic_scan = true;

    /** @brief Leak scan interval in milliseconds */
    double scan_interval_ms = 30000.0;  // 30 seconds

    /** @brief Enable automatic leak reporting */
    bool enable_auto_reporting = true;
};

/**
 * @brief GPU resource tracker for memory usage monitoring and leak detection
 *
 * Provides comprehensive GPU resource tracking capabilities including
 * memory allocation monitoring, leak detection, performance profiling,
 * and detailed usage statistics for debugging and optimization.
 *
 * Key features:
 * - Real-time memory allocation tracking with call stack capture
 * - Automatic memory leak detection with configurable thresholds
 * - Performance profiling with allocation/deallocation timing
 * - Resource usage statistics and fragmentation analysis
 * - Thread-safe operations with minimal performance overhead
 * - Integration with existing GPU memory management systems
 * - Detailed reporting and visualization capabilities
 *
 * The tracker uses efficient data structures to minimize overhead
 * while providing comprehensive monitoring capabilities. It can be
 * configured to operate in different modes from lightweight monitoring
 * to full debugging with call stack capture.
 *
 * @example
 * ```cpp
 * auto& tracker = gpu_resource_tracker::instance();
 *
 * // Configure leak detection
 * leak_detection_config config;
 * config.leak_threshold_ms = 30000.0; // 30 seconds
 * tracker.configure_leak_detection(config);
 *
 * // Track allocation
 * void* ptr = allocate_gpu_memory(1024ULL);
 * tracker.track_allocation(ptr, 1024ULL, device_enum::CUDA, 0, "simulation_data");
 *
 * // ... use memory ...
 *
 * // Track deallocation
 * tracker.track_deallocation(ptr);
 * free_gpu_memory(ptr);
 *
 * // Get statistics
 * auto stats = tracker.get_statistics();
 * std::cout << "Peak memory usage: " << stats.peak_bytes_in_use / 1024 / 1024 << " MB\n";
 * ```
 */
class MEMORY_VISIBILITY gpu_resource_tracker
{
public:
    /**
     * @brief Get the singleton instance of the resource tracker
     * @return Reference to the global resource tracker instance
     */
    MEMORY_API static gpu_resource_tracker& instance();

    /**
     * @brief Virtual destructor
     */
    MEMORY_API virtual ~gpu_resource_tracker() = default;

    /**
     * @brief Configure leak detection parameters
     * @param config Leak detection configuration
     */
    MEMORY_API virtual void configure_leak_detection(const leak_detection_config& config) = 0;

    /**
     * @brief Track a new GPU memory allocation
     * @param ptr Pointer to allocated memory
     * @param size Size of allocation in bytes
     * @param device_type Device type where memory is allocated
     * @param device_index Device index
     * @param tag Optional tag for categorization
     * @param source_file Source file name (automatically filled by macro)
     * @param source_line Source line number (automatically filled by macro)
     * @param function_name Function name (automatically filled by macro)
     * @return Allocation ID for tracking
     */
    MEMORY_API virtual size_t track_allocation(
        void*              ptr,
        size_t             size,
        device_enum        device_type,
        int                device_index,
        const std::string& tag           = "",
        const char*        source_file   = __builtin_FILE(),
        int                source_line   = __builtin_LINE(),
        const char*        function_name = __builtin_FUNCTION()) = 0;

    /**
     * @brief Track a GPU memory deallocation
     * @param ptr Pointer to memory being deallocated
     * @return True if allocation was found and tracked
     */
    MEMORY_API virtual bool track_deallocation(void* ptr) = 0;

    /**
     * @brief Record memory access for usage tracking
     * @param ptr Pointer to accessed memory
     */
    MEMORY_API virtual void record_access(void* ptr) = 0;

    /**
     * @brief Get information about a specific allocation
     * @param ptr Pointer to query
     * @return Allocation information, or nullptr if not found
     */
    MEMORY_API virtual std::shared_ptr<gpu_allocation_info> get_allocation_info(
        void* ptr) const = 0;

    /**
     * @brief Get current resource usage statistics
     * @return Current statistics
     */
    MEMORY_API virtual unified_resource_stats get_statistics() const = 0;

    /**
     * @brief Get list of all active allocations
     * @return Vector of active allocation information
     */
    MEMORY_API virtual std::vector<std::shared_ptr<gpu_allocation_info>> get_active_allocations()
        const = 0;

    /**
     * @brief Detect potential memory leaks
     * @return Vector of allocations that may be leaks
     */
    MEMORY_API virtual std::vector<std::shared_ptr<gpu_allocation_info>> detect_leaks() const = 0;

    /**
     * @brief Get allocations by tag
     * @param tag Tag to search for
     * @return Vector of allocations with matching tag
     */
    MEMORY_API virtual std::vector<std::shared_ptr<gpu_allocation_info>> get_allocations_by_tag(
        const std::string& tag) const = 0;

    /**
     * @brief Get allocations by device
     * @param device_type Device type
     * @param device_index Device index
     * @return Vector of allocations on specified device
     */
    MEMORY_API virtual std::vector<std::shared_ptr<gpu_allocation_info>>
    get_allocations_by_device(device_enum device_type, int device_index) const = 0;

    /**
     * @brief Clear all tracking data
     */
    MEMORY_API virtual void clear_all_data() = 0;

    /**
     * @brief Generate detailed resource usage report
     * @param include_call_stacks Whether to include call stack information
     * @return Formatted report string
     */
    MEMORY_API virtual std::string generate_report(bool include_call_stacks = false) const = 0;

    /**
     * @brief Generate leak detection report
     * @return Formatted leak report string
     */
    MEMORY_API virtual std::string generate_leak_report() const = 0;

    /**
     * @brief Enable or disable tracking
     * @param enabled Whether tracking should be enabled
     */
    MEMORY_API virtual void set_tracking_enabled(bool enabled) = 0;

    /**
     * @brief Check if tracking is currently enabled
     * @return True if tracking is enabled
     */
    MEMORY_API virtual bool is_tracking_enabled() const = 0;

protected:
    gpu_resource_tracker() = default;
    MEMORY_DELETE_COPY_AND_MOVE(gpu_resource_tracker);
};

}  // namespace gpu
}  // namespace memory

/**
 * @brief Convenience macro for tracking GPU allocations with automatic source location
 */
#define MEMORY_TRACK_GPU_ALLOCATION(ptr, size, device_type, device_index, tag) \
    memory::gpu::gpu_resource_tracker::instance().track_allocation(            \
        ptr, size, device_type, device_index, tag, __FILE__, __LINE__, __FUNCTION__)

/**
 * @brief Convenience macro for tracking GPU deallocations
 */
#define MEMORY_TRACK_GPU_DEALLOCATION(ptr) \
    memory::gpu::gpu_resource_tracker::instance().track_deallocation(ptr)

/**
 * @brief Convenience macro for recording GPU memory access
 */
#define MEMORY_RECORD_GPU_ACCESS(ptr) \
    memory::gpu::gpu_resource_tracker::instance().record_access(ptr)
