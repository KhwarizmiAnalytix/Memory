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
 * Modifications for Quarisma:
 * - Adapted for Quarisma quantitative computing requirements
 * - Added high-performance GPU memory allocation optimizations
 * - Integrated CUDA-aware allocation strategies
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/memory_macros.h"
#include "logger.h"
#include "common/device.h"
#include "gpu/gpu_device_manager.h"
#include "gpu/gpu_memory_pool.h"
#include "gpu/gpu_memory_transfer.h"
#include "gpu/gpu_resource_tracker.h"
#include "profiler/unified_memory_stats.h"

#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>
#include "common/memory_export.h"
#endif

namespace memory
{
namespace gpu
{

/**
 * @brief GPU-specific logging verbosity levels for memory tracking operations.
 *
 * Controls the amount of detail logged during GPU memory allocation and
 * deallocation operations. Higher levels provide more detailed information
 * but may impact performance in GPU-intensive applications.
 */
enum class gpu_tracking_log_level : int
{
    SILENT      = 0,  ///< No logging output
    ERROR       = 1,  ///< Only log errors and critical CUDA issues
    WARNING     = 2,  ///< Log warnings, CUDA errors, and critical issues
    INFO        = 3,  ///< Log general information, warnings, and errors
    DEBUG_LEVEL = 4,  ///< Log detailed debugging information including CUDA calls
    TRACE       = 5   ///< Log all operations with maximum detail including memory transfers
};

/**
 * @brief CUDA-specific error information for comprehensive error tracking.
 *
 * Captures detailed CUDA error information including error codes, messages,
 * and context information for debugging GPU memory allocation issues.
 */
struct MEMORY_VISIBILITY cuda_error_info
{
    int         error_code{0};      ///< CUDA error code (cudaError_t)
    std::string error_message;      ///< Human-readable error message
    std::string cuda_function;      ///< CUDA function that failed
    int64_t     timestamp_us{0};    ///< Timestamp when error occurred
    size_t      attempted_size{0};  ///< Size of allocation that failed
    int         device_index{-1};   ///< Device index where error occurred

    /**
     * @brief Default constructor for container compatibility.
     */
    cuda_error_info() = default;

    /**
     * @brief Constructs CUDA error info from CUDA error.
     *
     * @param cuda_error CUDA error code
     * @param function_name Name of CUDA function that failed
     * @param size Size of allocation that failed
     * @param device Device index where error occurred
     */
#if MEMORY_HAS_CUDA
    cuda_error_info(
        cudaError_t cuda_error, const char* function_name, size_t size, int device) noexcept;
#endif
};

/**
 * @brief GPU memory bandwidth utilization metrics.
 *
 * Provides detailed analysis of GPU memory bandwidth usage patterns to help
 * optimize memory access patterns and identify performance bottlenecks in
 * quantitative computing applications.
 *
 * **Mathematical Foundation**:
 * - Bandwidth Utilization = (Actual Transfer Rate) / (Peak Memory Bandwidth)
 * - Effective Bandwidth = (Useful Data Transferred) / (Total Transfer Time)
 * - Memory Efficiency = (Coalesced Accesses) / (Total Memory Accesses)
 *
 * **Performance Impact**: O(1) for basic metrics, O(n) for detailed analysis
 */
struct MEMORY_VISIBILITY gpu_bandwidth_metrics
{
    double   peak_bandwidth_gbps{0.0};       ///< Peak theoretical bandwidth (GB/s)
    double   effective_bandwidth_gbps{0.0};  ///< Measured effective bandwidth (GB/s)
    double   utilization_percentage{0.0};    ///< Bandwidth utilization [0.0, 100.0]
    size_t   total_bytes_transferred{0};     ///< Total bytes transferred
    uint64_t total_transfer_time_us{0};      ///< Total transfer time (microseconds)
    size_t   coalesced_accesses{0};          ///< Number of coalesced memory accesses
    size_t   uncoalesced_accesses{0};        ///< Number of uncoalesced memory accesses
    double   memory_efficiency{0.0};         ///< Memory access efficiency [0.0, 1.0]

    /**
     * @brief Calculates comprehensive bandwidth analysis.
     *
     * @param device_info GPU device information
     * @param transfer_records Vector of memory transfer records
     * @return Computed bandwidth metrics
     *
     * **Complexity**: O(n) where n is number of transfer records
     * **Thread Safety**: Thread-safe for read-only operations
     */
    static gpu_bandwidth_metrics calculate(
        const gpu_device_info&                device_info,
        const std::vector<gpu_transfer_info>& transfer_records) noexcept;
};

/**
 * @brief Enhanced GPU allocation record with comprehensive metadata and CUDA-specific information.
 *
 * Extended allocation record that includes GPU-specific metadata for detailed
 * memory usage analysis, performance profiling, and CUDA debugging.
 */
struct MEMORY_VISIBILITY enhanced_gpu_alloc_record
{
    size_t          requested_bytes{0};              ///< Originally requested size
    size_t          allocated_bytes{0};              ///< Actual allocated size
    size_t          alignment{0};                    ///< Memory alignment requirement
    device_enum     device_type{device_enum::CUDA};  ///< Device type (CUDA, HIP, etc.)
    int             device_index{0};                 ///< Device index
    int64_t         allocation_id{0};                ///< Unique allocation identifier
    uint64_t        alloc_duration_us{0};            ///< Time taken for allocation (microseconds)
    uint64_t        dealloc_duration_us{0};          ///< Time taken for deallocation (microseconds)
    int64_t         alloc_timestamp_us{0};           ///< Timestamp when allocation occurred
    int64_t         dealloc_timestamp_us{0};         ///< Timestamp when deallocation occurred
    std::string     tag;                             ///< Optional allocation tag for categorization
    const char*     source_file{nullptr};            ///< Source file where allocation occurred
    int             source_line{0};                  ///< Source line number
    const char*     function_name{nullptr};          ///< Function name where allocation occurred
    void*           cuda_stream{nullptr};            ///< Associated CUDA stream (if any)
    size_t          memory_pool_id{0};               ///< Memory pool identifier
    cuda_error_info error_info;  ///< CUDA error information (if allocation failed)

    /**
     * @brief Default constructor for container compatibility.
     */
    enhanced_gpu_alloc_record() noexcept = default;

    /**
     * @brief Constructs enhanced GPU allocation record with comprehensive metadata.
     */
    enhanced_gpu_alloc_record(
        size_t      requested_size,
        size_t      actual_size,
        size_t      align,
        device_enum dev_type,
        int         dev_index,
        int64_t     alloc_time,
        int64_t     alloc_id,
        std::string allocation_tag = "",
        const char* file           = nullptr,
        int         line           = 0,
        const char* func           = nullptr,
        void*       stream         = nullptr) noexcept;
};

/**
 * @brief Advanced GPU memory allocation tracker with comprehensive CUDA debugging capabilities.
 *
 * Provides sophisticated GPU memory allocation tracking, debugging, and analysis
 * capabilities by wrapping GPU allocation functions. Offers detailed statistics,
 * allocation records, CUDA-specific metrics, and advanced debugging features
 * optimized for quantitative computing workloads.
 *
 * **Key Features**:
 * - Comprehensive GPU allocation tracking and statistics
 * - CUDA-specific error handling and debugging
 * - Memory bandwidth utilization analysis
 * - Stream-aware allocation tracking
 * - Device memory hierarchy monitoring
 * - Thread-safe operation with fine-grained locking
 * - Integration with Memory GPU resource management
 *
 * **Design Principles**:
 * - Wrapper pattern around existing GPU allocation functions
 * - Minimal performance overhead when tracking disabled
 * - Comprehensive CUDA debugging information when enabled
 * - Safe lifecycle management with RAII principles
 * - Integration with Memory device management framework
 *
 * **Use Cases**:
 * - GPU memory usage optimization for Monte Carlo simulations
 * - CUDA kernel memory access pattern analysis
 * - GPU memory leak detection and debugging
 * - Performance profiling for PDE solvers
 * - Resource usage monitoring in production GPU clusters
 *
 * **Thread Safety**: Fully thread-safe with internal mutex protection
 * **Performance**: Minimal overhead - delegates to underlying GPU allocators
 * **Memory Overhead**: Small fixed overhead plus tracking data structures
 * **CUDA Integration**: Full integration with CUDA runtime and events
 */
class MEMORY_VISIBILITY gpu_allocator_tracking
{
public:
    /**
     * @brief Constructs GPU tracking allocator with comprehensive configuration.
     *
     * @param device_type Target GPU device type (CUDA, HIP, etc.)
     * @param device_index Device index to track
     * @param enable_enhanced_tracking Whether to enable comprehensive analytics
     * @param enable_bandwidth_tracking Whether to track memory bandwidth metrics
     *
     * **Device Management**: Integrates with Memory device manager
     * **Enhanced Tracking**: Enables detailed analytics, timing, and source location tracking
     * **Bandwidth Tracking**: Monitors memory transfer performance and efficiency
     * **Thread Safety**: Constructor is not thread-safe
     * **Performance Impact**: Enhanced tracking adds minimal overhead (~5-10%)
     * **CUDA Integration**: Automatically detects and integrates with CUDA runtime
     */
    MEMORY_API explicit gpu_allocator_tracking(
        device_enum device_type               = device_enum::CUDA,
        int         device_index              = 0,
        bool        enable_enhanced_tracking  = true,
        bool        enable_bandwidth_tracking = false);

    /**
     * @brief Destructor ensuring proper cleanup of GPU resources.
     *
     * **CUDA Cleanup**: Ensures all CUDA events and streams are properly destroyed
     * **Resource Management**: Cleans up tracking data structures
     * **Thread Safety**: Destructor assumes no concurrent access
     */
    MEMORY_API ~gpu_allocator_tracking();

    // ========== Core GPU Allocation Interface ==========

    /**
     * @brief Allocates GPU memory with comprehensive tracking and statistics.
     *
     * Allocates GPU memory using the underlying GPU allocator while providing
     * comprehensive tracking, timing analysis, and CUDA-specific debugging
     * information.
     *
     * @tparam T Element type to allocate
     * @param count Number of elements to allocate
     * @param alignment Memory alignment requirement (default: 256ULL bytes for GPU coalescing)
     * @param pool Optional memory pool to use
     * @param tag Optional allocation tag for categorization
     * @param stream Optional CUDA stream for asynchronous operations
     * @param source_file Source file name (automatically filled by macro)
     * @param source_line Source line number (automatically filled by macro)
     * @param function_name Function name (automatically filled by macro)
     * @return Pointer to allocated GPU memory, or nullptr on failure
     *
     * **Performance**: Includes timing analysis with microsecond precision
     * **CUDA Integration**: Uses CUDA events for accurate GPU-side timing
     * **Error Handling**: Comprehensive CUDA error capture and reporting
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Memory Tracking**: Updates all relevant statistics and records
     *
     * **CUDA Memory Types Supported**:
     * - Device memory (cudaMalloc)
     * - Unified memory (cudaMallocManaged)
     * - Pinned memory (cudaMallocHost)
     * - Texture memory (via memory pools)
     */
    template <typename T>
    T* allocate(
        size_t                           count,
        size_t                           alignment     = 256ULL,
        std::shared_ptr<gpu_memory_pool> pool          = nullptr,
        const std::string&               tag           = "",
        void*                            stream        = nullptr,
        const char*                      source_file   = nullptr,
        int                              source_line   = 0,
        const char*                      function_name = nullptr);

    /**
     * @brief Deallocates GPU memory with comprehensive tracking and timing.
     *
     * Deallocates GPU memory while providing comprehensive tracking, timing
     * analysis, and CUDA-specific debugging information.
     *
     * @tparam T Element type being deallocated
     * @param ptr Pointer to GPU memory to deallocate
     * @param count Number of elements being deallocated
     * @param stream Optional CUDA stream for asynchronous operations
     *
     * **Performance**: Includes timing analysis with microsecond precision
     * **CUDA Integration**: Uses CUDA events for accurate GPU-side timing
     * **Error Handling**: Comprehensive CUDA error capture and reporting
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Memory Tracking**: Updates all relevant statistics and records
     */
    template <typename T>
    void deallocate(T* ptr, size_t count, void* stream = nullptr);

    /**
     * @brief Allocates raw GPU memory with comprehensive tracking.
     *
     * Low-level GPU memory allocation interface for cases where typed
     * allocation is not appropriate.
     *
     * @param bytes Number of bytes to allocate
     * @param alignment Memory alignment requirement
     * @param pool Optional memory pool to use
     * @param tag Optional allocation tag for categorization
     * @param stream Optional CUDA stream for asynchronous operations
     * @return Pointer to allocated GPU memory, or nullptr on failure
     */
    MEMORY_API void* allocate_raw(
        size_t                           bytes,
        size_t                           alignment = 256ULL,
        std::shared_ptr<gpu_memory_pool> pool      = nullptr,
        const std::string&               tag       = "",
        void*                            stream    = nullptr);

    /**
     * @brief Deallocates raw GPU memory with comprehensive tracking.
     *
     * @param ptr Pointer to GPU memory to deallocate
     * @param bytes Number of bytes being deallocated
     * @param stream Optional CUDA stream for asynchronous operations
     */
    MEMORY_API void deallocate_raw(void* ptr, size_t bytes, void* stream = nullptr);

    // ========== GPU-Specific Analytics and Metrics ==========

    /**
     * @brief Retrieves comprehensive GPU memory bandwidth analysis.
     *
     * Analyzes GPU memory transfer patterns to identify bandwidth utilization
     * issues and provide optimization recommendations for GPU-intensive
     * quantitative computing workloads.
     *
     * @return Detailed bandwidth metrics and analysis
     *
     * **Complexity**: O(n) where n is number of transfer records
     * **Thread Safety**: Thread-safe with shared lock for read operations
     * **Requirements**: Requires CUDA runtime for accurate bandwidth measurement
     *
     * **Use Cases**:
     * - GPU memory access pattern optimization
     * - CUDA kernel performance tuning
     * - Memory coalescing analysis
     * - GPU cluster resource planning
     */
    MEMORY_API gpu_bandwidth_metrics GetBandwidthMetrics() const;

    /**
     * @brief Retrieves detailed GPU performance timing statistics.
     *
     * Returns comprehensive timing analysis for GPU memory operations including
     * allocation, deallocation, and memory transfer times with CUDA event-based
     * precision for accurate GPU-side timing.
     *
     * @return Detailed timing statistics for GPU memory operations
     *
     * **Performance**: O(1) - returns cached atomic statistics
     * **Thread Safety**: Thread-safe atomic operations
     * **Precision**: Microsecond resolution using CUDA events
     *
     * **Metrics Included**:
     * - Average GPU allocation/deallocation times
     * - Min/max operation times with CUDA event precision
     * - Total operation counts and cumulative timing data
     * - CUDA synchronization overhead analysis
     */
    MEMORY_API atomic_timing_stats GetGPUTimingStats() const noexcept;

    /**
     * @brief Retrieves enhanced GPU allocation records with comprehensive metadata.
     *
     * Returns detailed GPU allocation history including CUDA-specific timing
     * information, device context, stream associations, and performance metrics
     * for advanced debugging and profiling analysis.
     *
     * @return Vector of enhanced GPU allocation records
     *
     * **Thread Safety**: Thread-safe with shared lock
     * **Performance**: O(n) where n is number of allocation records
     * **Memory**: Returns copy of records, may be memory-intensive for large histories
     *
     * **Enhanced GPU Data**:
     * - CUDA device and stream information
     * - GPU memory hierarchy details (device, unified, pinned)
     * - Memory bandwidth utilization per allocation
     * - CUDA error information for failed allocations
     */
    MEMORY_API std::vector<enhanced_gpu_alloc_record> GetEnhancedGPURecords() const;

    /**
     * @brief Configures logging verbosity for GPU tracking operations.
     *
     * Sets the level of detail for logging GPU memory allocation and deallocation
     * operations. Higher verbosity levels provide more CUDA-specific information
     * but may impact performance in GPU-intensive scenarios.
     *
     * @param level Desired GPU logging verbosity level
     *
     * **Performance Impact**: Higher levels may reduce GPU allocation performance
     * **Thread Safety**: Thread-safe atomic operation
     * **Default**: INFO level logging
     * **CUDA Integration**: TRACE level includes detailed CUDA API call logging
     */
    MEMORY_API void SetGPULoggingLevel(gpu_tracking_log_level level) noexcept;

    /**
     * @brief Gets current GPU logging verbosity level.
     *
     * @return Current GPU logging level setting
     *
     * **Performance**: O(1) atomic read
     * **Thread Safety**: Thread-safe atomic operation
     */
    MEMORY_API gpu_tracking_log_level GetGPULoggingLevel() const noexcept;

    /**
     * @brief Resets all GPU performance timing statistics.
     *
     * Clears all GPU timing data while preserving allocation records and
     * memory usage statistics. Useful for GPU performance benchmarking
     * and testing scenarios.
     *
     * **Thread Safety**: Thread-safe with exclusive lock
     * **Performance**: O(1) - simple atomic resets
     * **Preservation**: Keeps GPU allocation records and memory statistics
     * **CUDA Integration**: Resets CUDA event timing data
     */
    MEMORY_API void ResetGPUTimingStats() noexcept;

    /**
     * @brief Calculates current GPU memory efficiency metrics.
     *
     * Analyzes GPU memory allocation patterns to determine memory usage efficiency
     * including memory coalescing, allocation overhead, and GPU memory hierarchy
     * utilization ratios.
     *
     * @return Tuple of (coalescing_efficiency, memory_utilization, gpu_efficiency_score)
     *
     * **Return Values**:
     * - coalescing_efficiency: Memory coalescing ratio [0.0, 1.0]
     * - memory_utilization: GPU memory utilization ratio [0.0, 1.0]
     * - gpu_efficiency_score: Overall GPU memory efficiency [0.0, 1.0]
     *
     * **Thread Safety**: Thread-safe with shared lock
     * **Performance**: O(1) for cached metrics, O(n) for detailed analysis
     * **CUDA Integration**: Uses CUDA memory info APIs for accurate metrics
     */
    MEMORY_API std::tuple<double, double, double> GetGPUEfficiencyMetrics() const;

    /**
     * @brief Generates comprehensive GPU memory usage report.
     *
     * Creates detailed textual report of GPU memory usage patterns, CUDA-specific
     * performance statistics, bandwidth analysis, and optimization recommendations
     * for GPU-intensive quantitative computing applications.
     *
     * @param include_allocations Whether to include individual GPU allocation details
     * @param include_cuda_info Whether to include detailed CUDA runtime information
     * @return Formatted string report with comprehensive GPU analysis
     *
     * **Thread Safety**: Thread-safe with shared lock
     * **Performance**: O(n) where n is number of allocations if detailed
     * **Format**: Human-readable text with structured GPU-specific sections
     *
     * **Report Sections**:
     * - GPU device information and capabilities
     * - Memory usage summary across memory hierarchy
     * - CUDA-specific performance timing analysis
     * - Memory bandwidth utilization metrics
     * - GPU efficiency recommendations
     * - Optional detailed allocation history
     */
    MEMORY_API std::string GenerateGPUReport(
        bool include_allocations = false, bool include_cuda_info = true) const;

    /**
     * @brief Gets current GPU device information.
     *
     * @return GPU device information structure
     *
     * **Thread Safety**: Thread-safe read operation
     * **CUDA Integration**: Queries current CUDA device properties
     */
    MEMORY_API gpu_device_info GetDeviceInfo() const noexcept;

    /**
     * @brief Gets total GPU memory usage across all memory types.
     *
     * @return Tuple of (device_memory_bytes, unified_memory_bytes, pinned_memory_bytes)
     *
     * **Thread Safety**: Thread-safe with shared lock
     * **CUDA Integration**: Queries CUDA memory info APIs
     */
    MEMORY_API std::tuple<size_t, size_t, size_t> GetGPUMemoryUsage() const;

private:
    // ========== Core Configuration ==========

    device_enum     device_type_;   ///< Target GPU device type
    int             device_index_;  ///< Device index being tracked
    gpu_device_info device_info_;   ///< Cached device information

    // ========== Enhanced Analytics and Performance Tracking ==========

    mutable atomic_timing_stats gpu_timing_stats_;  ///< GPU timing statistics
    mutable std::vector<enhanced_gpu_alloc_record>
                                        gpu_records_;  ///< Enhanced GPU allocation records
    std::atomic<gpu_tracking_log_level> gpu_log_level_{
        gpu_tracking_log_level::INFO};  ///< GPU logging level

    // ========== Thread Safety ==========

    mutable std::shared_mutex gpu_shared_mu_;  ///< Shared mutex for GPU analytics
    mutable std::mutex        gpu_mu_;         ///< Exclusive mutex for GPU operations

    // ========== GPU-Specific Tracking ==========

    const bool           enhanced_tracking_enabled_;   ///< Enhanced tracking flag
    const bool           bandwidth_tracking_enabled_;  ///< Bandwidth tracking flag
    std::atomic<int64_t> next_gpu_allocation_id_{1};   ///< GPU allocation ID counter

    // ========== CUDA-Specific Members ==========

#if MEMORY_HAS_CUDA
    cudaEvent_t start_event_;                     ///< CUDA event for timing start
    cudaEvent_t end_event_;                       ///< CUDA event for timing end
    bool        cuda_events_initialized_{false};  ///< CUDA events initialization flag
#endif

    // ========== Memory Usage Tracking ==========

    std::atomic<size_t> total_device_memory_{0};   ///< Total device memory allocated
    std::atomic<size_t> total_unified_memory_{0};  ///< Total unified memory allocated
    std::atomic<size_t> total_pinned_memory_{0};   ///< Total pinned memory allocated
    std::atomic<size_t> peak_device_memory_{0};    ///< Peak device memory usage
    std::atomic<size_t> peak_unified_memory_{0};   ///< Peak unified memory usage
    std::atomic<size_t> peak_pinned_memory_{0};    ///< Peak pinned memory usage

    // ========== Private Helper Methods ==========

    void InitializeCUDAEvents();        ///< Initialize CUDA events for timing
    void CleanupCUDAEvents() noexcept;  ///< Cleanup CUDA events
    void UpdateMemoryUsage(size_t bytes, bool is_allocation);  ///< Update memory usage statistics
    void LogGPUOperation(
        const std::string& operation, const std::string& details) const;  ///< Log GPU operations
};

// ========== Template Method Implementations ==========

template <typename T>
T* gpu_allocator_tracking::allocate(
    size_t                           count,
    size_t                           alignment,
    std::shared_ptr<gpu_memory_pool> pool,
    const std::string&               tag,
    void*                            stream,
    MEMORY_UNUSED const char*      source_file,
    MEMORY_UNUSED int              source_line,
    MEMORY_UNUSED const char*      function_name)
{
    if (count == 0)
        return nullptr;

    const size_t bytes   = count * sizeof(T);
    void*        raw_ptr = allocate_raw(bytes, alignment, pool, tag, stream);
    return static_cast<T*>(raw_ptr);
}

template <typename T>
void gpu_allocator_tracking::deallocate(T* ptr, size_t count, void* stream)
{
    if (!ptr || count == 0)
        return;

    const size_t bytes = count * sizeof(T);
    deallocate_raw(ptr, bytes, stream);
}

}  // namespace gpu
}  // namespace memory

// ========== Convenience Macros for GPU Allocation Tracking ==========

/**
 * @brief Convenience macro for tracked GPU allocation with automatic source location
 */
#define MEMORY_GPU_ALLOCATE_TRACKED(tracker, type, count, alignment, pool, tag, stream) \
    (tracker).allocate<type>(count, alignment, pool, tag, stream, __FILE__, __LINE__, __FUNCTION__)

/**
 * @brief Convenience macro for tracked GPU deallocation
 */
#define MEMORY_GPU_DEALLOCATE_TRACKED(tracker, ptr, count, stream) \
    (tracker).deallocate(ptr, count, stream)
