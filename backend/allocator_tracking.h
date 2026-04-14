/*
 * Quarisma: High-Performance Computational Library
 *
 * Original work Copyright 2015 The TensorFlow Authors
 * Modified work Copyright 2025 Quarisma Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file contains code modified from TensorFlow (Apache 2.0 licensed)
 * and is part of Quarisma, licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * MODIFICATIONS FROM ORIGINAL:
 * - Adapted for Quarisma quantitative computing requirements
 * - Added high-performance memory allocation optimizations
 * - Integrated NUMA-aware allocation strategies
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
#include <tuple>
#include <vector>

#include "common/memory_macros.h"
#include "logger.h"
#include "cpu/allocator.h"
#include "profiler/unified_memory_stats.h"
#include "util/flat_hash.h"
#include "common/memory_export.h"

namespace memory
{
/**
 * @brief Comprehensive memory allocation tracking and debugging wrapper.
 *
 * allocator_tracking provides sophisticated memory allocation tracking and
 * debugging capabilities by wrapping any underlying Allocator. It maintains
 * detailed statistics, allocation records, and provides advanced debugging
 * features for memory usage analysis.
 *
 * **Key Features**:
 * - Comprehensive allocation tracking and statistics
 * - High watermark monitoring for peak memory usage
 * - Detailed allocation records with timestamps
 * - Reference counting for safe wrapper lifecycle management
 * - Optional local size tracking for non-tracking allocators
 * - Thread-safe operation with fine-grained locking
 *
 * **Design Principles**:
 * - Decorator pattern wrapping underlying allocator
 * - Minimal performance overhead when tracking disabled
 * - Comprehensive debugging information when enabled
 * - Safe lifecycle management with reference counting
 * - Integration with Memory execution framework
 *
 * **Use Cases**:
 * - Operation-specific memory usage tracking
 * - Memory leak detection and debugging
 * - Performance profiling and optimization
 * - Resource usage monitoring in production
 * - Development and testing environments
 *
 * **Thread Safety**: Fully thread-safe with internal mutex protection
 * **Performance**: Minimal overhead - delegates to underlying allocator
 * **Memory Overhead**: Small fixed overhead plus tracking data structures
 *
 * **Lifecycle Management**:
 * The allocator_tracking uses reference counting to manage its lifecycle safely.
 * It assumes that all allocations by an operation occur before the operation's
 * Compute method returns, establishing the high watermark. Deallocations can
 * occur long after operation completion, and the wrapper automatically deletes
 * itself when the last reference is released and all memory is deallocated.
 */

/**
 * @brief Record of a single memory allocation with timing information.
 *
 * Captures essential information about individual allocations including
 * size and timing for detailed memory usage analysis and debugging.
 *
 * **Use Cases**:
 * - Allocation pattern analysis
 * - Memory usage profiling
 * - Performance debugging
 * - Resource usage monitoring
 */
struct alloc_record
{
    /**
     * @brief Constructs allocation record with size and timing.
     *
     * @param a_bytes Size of allocation in bytes
     * @param a_micros Timestamp in microseconds when allocation occurred
     *
     * **Timing**: Microsecond precision for detailed analysis
     * **Size**: Actual allocated size (may differ from requested)
     */
    alloc_record(int64_t a_bytes, int64_t a_micros) noexcept
        : alloc_bytes(a_bytes), alloc_micros(a_micros)
    {
    }

    /**
     * @brief Default constructor creating zero-initialized record.
     *
     * **Use Cases**: Container initialization, placeholder records
     */
    alloc_record() noexcept : alloc_record(0, 0) {}

    /**
     * @brief Size of allocation in bytes.
     *
     * Records the actual allocated size, which may be larger than
     * the originally requested size due to alignment or allocator policies.
     */
    int64_t alloc_bytes;

    /**
     * @brief Timestamp when allocation occurred (microseconds).
     *
     * High-precision timestamp enabling detailed timing analysis
     * of allocation patterns and performance characteristics.
     */
    int64_t alloc_micros;
};

/**
 * @brief Logging verbosity levels for memory tracking operations.
 *
 * Controls the amount of detail logged during memory allocation and
 * deallocation operations. Higher levels provide more detailed information
 * but may impact performance.
 */
enum class tracking_log_level : int
{
    SILENT      = 0,  ///< No logging output
    ERROR       = 1,  ///< Only log errors and critical issues
    WARNING     = 2,  ///< Log warnings and errors
    INFO        = 3,  ///< Log general information, warnings, and errors
    DEBUG_LEVEL = 4,  ///< Log detailed debugging information
    TRACE       = 5   ///< Log all operations with maximum detail
};

/**
 * @brief Enhanced allocation record with comprehensive metadata and timing.
 *
 * Extended version of alloc_record that includes additional metadata for
 * detailed memory usage analysis, performance profiling, and debugging.
 */
struct enhanced_alloc_record : public alloc_record
{
    size_t      requested_bytes{0};      ///< Originally requested size
    size_t      alignment{0};            ///< Memory alignment requirement
    int64_t     allocation_id{0};        ///< Unique allocation identifier
    uint64_t    alloc_duration_us{0};    ///< Time taken for allocation (microseconds)
    uint64_t    dealloc_duration_us{0};  ///< Time taken for deallocation (microseconds)
    std::string tag;                     ///< Optional allocation tag for categorization
    const char* source_file{nullptr};    ///< Source file where allocation occurred
    int         source_line{0};          ///< Source line number
    const char* function_name{nullptr};  ///< Function name where allocation occurred

    /**
     * @brief Constructs enhanced allocation record with comprehensive metadata.
     *
     * @param requested_size Originally requested allocation size
     * @param actual_size Actual allocated size (may be larger due to alignment)
     * @param align Memory alignment requirement
     * @param alloc_time Timestamp when allocation occurred
     * @param alloc_id Unique allocation identifier
     * @param allocation_tag Optional tag for categorization
     * @param file Source file name (optional)
     * @param line Source line number (optional)
     * @param func Function name (optional)
     */
    enhanced_alloc_record(
        size_t             requested_size,
        size_t             actual_size,
        size_t             align,
        int64_t            alloc_time,
        int64_t            alloc_id,
        const std::string& allocation_tag = "",
        const char*        file           = nullptr,
        int                line           = 0,
        const char*        func           = nullptr) noexcept
        : alloc_record(static_cast<int64_t>(actual_size), alloc_time),
          requested_bytes(requested_size),
          alignment(align),
          allocation_id(alloc_id),
          tag(allocation_tag),
          source_file(file),
          source_line(line),
          function_name(func)
    {
    }

    /**
     * @brief Default constructor for container compatibility.
     */
    enhanced_alloc_record() noexcept = default;
};

/**
 * @brief Advanced memory allocation tracker with comprehensive debugging capabilities.
 *
 * Provides sophisticated memory allocation tracking, debugging, and analysis
 * capabilities by wrapping any underlying Allocator implementation. Offers
 * detailed statistics, allocation records, and advanced debugging features.
 */
class MEMORY_VISIBILITY allocator_tracking : public Allocator
{
public:
    /**
     * @brief Constructs tracking allocator wrapping underlying allocator.
     *
     * @param allocator Underlying allocator to wrap (not owned)
     * @param track_sizes Whether to track allocation sizes locally
     * @param enable_enhanced_tracking Whether to enable comprehensive analytics
     *
     * **Ownership**: Does not take ownership of underlying allocator
     * **Local Tracking**: When enabled, maintains size information even
     *                    if underlying allocator doesn't track sizes
     * **Enhanced Tracking**: Enables detailed analytics, timing, and source location tracking
     * **Thread Safety**: Constructor is not thread-safe
     * **Reference Counting**: Initializes with reference count of 1
     * **Performance Impact**: Enhanced tracking adds minimal overhead (~5-10%)
     */
    MEMORY_API explicit allocator_tracking(
        Allocator* allocator, bool track_sizes, bool enable_enhanced_tracking = true);

    /**
     * @brief Returns name of underlying allocator.
     *
     * @return Name string from wrapped allocator
     *
     * **Delegation**: Forwards to underlying allocator
     * **Thread Safety**: Thread-safe if underlying allocator is thread-safe
     * **Performance**: O(1) - simple delegation
     */
    std::string Name() const override { return allocator_->Name(); }

    /**
     * @brief Allocates memory with default allocation attributes.
     *
     * Convenience overload that uses default allocation_attributes.
     * Delegates to full allocate_raw method with comprehensive tracking.
     *
     * @param alignment Required alignment in bytes
     * @param num_bytes Size of memory block to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Performance**: Same as full allocate_raw method
     * **Tracking**: Full allocation tracking and statistics
     */
    void* allocate_raw(size_t alignment, size_t num_bytes) override
    {
        return allocate_raw(alignment, num_bytes, allocation_attributes{});
    }

    /**
     * @brief Allocates memory with comprehensive tracking and statistics.
     *
     * Core allocation method that provides full tracking capabilities
     * including statistics updates, allocation records, and debugging
     * information collection.
     *
     * @param alignment Required alignment in bytes
     * @param num_bytes Size of memory block to allocate
     * @param allocation_attr Allocation attributes and policies
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Algorithm**: Delegates to underlying allocator with tracking
     * **Performance**: O(1) + tracking overhead
     * **Thread Safety**: Thread-safe with internal mutex protection
     * **Statistics**: Updates all tracking metrics and records
     * **Reference Counting**: Increments reference count during allocation
     */
    MEMORY_API void* allocate_raw(
        size_t alignment, size_t num_bytes, const allocation_attributes& allocation_attr) override;

    /**
     * @brief Deallocates memory with tracking updates.
     *
     * Deallocates memory through underlying allocator while updating
     * tracking statistics and managing reference counting for safe
     * wrapper lifecycle management.
     *
     * @param ptr Pointer to memory to deallocate
     *
     * **Algorithm**: Updates tracking then delegates to underlying allocator
     * **Performance**: O(1) + tracking overhead
     * **Thread Safety**: Thread-safe with internal mutex protection
     * **Reference Counting**: Decrements reference count, may trigger cleanup
     * **Statistics**: Updates bytes in use and allocation records
     */
    MEMORY_API void deallocate_raw(void* ptr) override;

    /**
     * @brief Indicates comprehensive allocation size tracking capability.
     *
     * @return true - allocator_tracking always provides size tracking
     *
     * **Capability**: Always true regardless of underlying allocator
     * **Local Tracking**: Provides tracking even for non-tracking allocators
     * **Thread Safety**: Thread-safe (returns constant)
     */
    MEMORY_API bool tracks_allocation_sizes() const noexcept override;

    /**
     * @brief Returns originally requested size for allocation.
     *
     * @param ptr Pointer to allocated memory
     * @return Original requested size in bytes
     *
     * **Source**: Local tracking or underlying allocator
     * **Performance**: O(1) - hash table lookup or delegation
     * **Thread Safety**: Thread-safe with internal synchronization
     */
    MEMORY_API size_t RequestedSize(const void* ptr) const override;

    /**
     * @brief Returns actual allocated size for allocation.
     *
     * @param ptr Pointer to allocated memory
     * @return Actual allocated size in bytes
     *
     * **Source**: Local tracking or underlying allocator
     * **Performance**: O(1) - hash table lookup or delegation
     * **Thread Safety**: Thread-safe with internal synchronization
     */
    MEMORY_API size_t AllocatedSize(const void* ptr) const override;

    /**
     * @brief Returns unique allocation identifier.
     *
     * @param ptr Pointer to allocated memory
     * @return Unique allocation ID
     *
     * **Source**: Local tracking or underlying allocator
     * **Performance**: O(1) - hash table lookup or delegation
     * **Thread Safety**: Thread-safe with internal synchronization
     */
    MEMORY_API int64_t AllocationId(const void* ptr) const override;

    /**
     * @brief Retrieves comprehensive tracking statistics.
     *
     * @return Current allocation statistics and metrics
     *
     * **Statistics**: Includes tracking-specific metrics
     * **Performance**: O(1) - returns cached statistics
     * **Thread Safety**: Thread-safe with internal synchronization
     */
    MEMORY_API std::optional<allocator_stats> GetStats() const override;

    /**
     * @brief Resets tracking statistics while preserving allocations.
     *
     * @return true if statistics were reset successfully
     *
     * **Behavior**: Resets counters but preserves current allocations
     * **Performance**: O(1) - simple counter reset
     * **Thread Safety**: Thread-safe with internal synchronization
     */
    MEMORY_API bool ClearStats() override;

    /**
     * @brief Returns memory type of underlying allocator.
     *
     * @return Memory type from wrapped allocator
     *
     * **Delegation**: Forwards to underlying allocator
     * **Thread Safety**: Thread-safe if underlying allocator is thread-safe
     */
    allocator_memory_enum GetMemoryType() const noexcept override
    {
        return allocator_->GetMemoryType();
    }

    /**
     * @brief Retrieves comprehensive memory usage metrics.
     *
     * Returns detailed memory usage information including total allocated bytes,
     * high watermark (peak usage), and currently allocated bytes. The exact
     * interpretation depends on underlying allocator capabilities.
     *
     * @return Tuple of (total_bytes, high_watermark, current_bytes)
     *
     * **Return Values**:
     * - If underlying allocator tracks sizes:
     *   - total_bytes: Total bytes allocated through this wrapper
     *   - high_watermark: Peak bytes allocated simultaneously
     *   - current_bytes: Currently allocated bytes still alive
     * - If underlying allocator doesn't track sizes:
     *   - total_bytes: Total bytes requested through this wrapper
     *   - high_watermark: 0
     *   - current_bytes: 0
     *
     * **Performance**: O(1) - returns cached values
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Use Cases**: Memory usage monitoring, performance analysis
     */
    MEMORY_API std::tuple<size_t, size_t, size_t> GetSizes() const;

    /**
     * @brief Retrieves allocation records and releases reference.
     *
     * Returns complete allocation history and decrements the reference count.
     * After this call, only deallocate_raw calls are allowed, and the wrapper
     * will automatically delete itself when all allocations are deallocated.
     *
     * @return Vector of all allocation records collected
     *
     * **Lifecycle**: Marks wrapper for eventual self-destruction
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Performance**: O(n) where n is number of allocation records
     * **Use Cases**: Final statistics collection, operation completion
     *
     * **Important**: After this call:
     * - No new allocations are allowed
     * - Only deallocations of existing pointers are permitted
     * - Wrapper will self-destruct when reference count reaches zero
     */
    MEMORY_API std::vector<alloc_record> GetRecordsAndUnRef();

    /**
     * @brief Returns copy of current allocation records.
     *
     * Provides access to allocation history without affecting wrapper
     * lifecycle or reference counting. Safe to call multiple times.
     *
     * @return Vector copy of current allocation records
     *
     * **Lifecycle**: Does not affect wrapper lifecycle
     * **Thread Safety**: Thread-safe with internal synchronization
     * **Performance**: O(n) where n is number of allocation records
     * **Use Cases**: Intermediate monitoring, debugging, analysis
     */
    MEMORY_API std::vector<alloc_record> GetCurrentRecords();

    // ========== Enhanced Memory Analytics ==========

    /**
     * @brief Retrieves comprehensive memory fragmentation analysis.
     *
     * Analyzes current memory usage patterns to identify fragmentation issues
     * and provide optimization recommendations. Requires underlying allocator
     * to support detailed memory layout information.
     *
     * @return Detailed fragmentation metrics and analysis
     *
     * **Complexity**: O(n) where n is number of free blocks
     * **Thread Safety**: Thread-safe with shared lock for read operations
     * **Requirements**: Underlying allocator must support fragmentation analysis
     *
     * **Use Cases**:
     * - Memory usage optimization
     * - Allocation strategy tuning
     * - Performance debugging
     * - Resource planning
     */
    MEMORY_API memory_fragmentation_metrics GetFragmentationMetrics() const;

    /**
     * @brief Retrieves detailed performance timing statistics.
     *
     * Returns comprehensive timing analysis for allocation and deallocation
     * operations including average times, min/max values, and distribution
     * statistics for performance optimization.
     *
     * @return Detailed timing statistics for memory operations
     *
     * **Performance**: O(1) - returns cached atomic statistics
     * **Thread Safety**: Thread-safe atomic operations
     * **Precision**: Microsecond resolution timing
     *
     * **Metrics Included**:
     * - Average allocation/deallocation times
     * - Min/max operation times
     * - Total operation counts
     * - Cumulative timing data
     */
    MEMORY_API atomic_timing_stats GetTimingStats() const noexcept;

    /**
     * @brief Retrieves enhanced allocation records with comprehensive metadata.
     *
     * Returns detailed allocation history including timing information,
     * source location data, and performance metrics for advanced debugging
     * and profiling analysis.
     *
     * @return Vector of enhanced allocation records
     *
     * **Thread Safety**: Thread-safe with shared lock
     * **Performance**: O(n) where n is number of allocation records
     * **Memory**: Returns copy of records, may be memory-intensive
     *
     * **Enhanced Data**:
     * - Source file and line information
     * - Allocation timing details
     * - Memory alignment requirements
     * - Custom allocation tags
     */
    MEMORY_API std::vector<enhanced_alloc_record> GetEnhancedRecords() const;

    /**
     * @brief Configures logging verbosity for tracking operations.
     *
     * Sets the level of detail for logging memory allocation and deallocation
     * operations. Higher verbosity levels provide more information but may
     * impact performance in high-frequency allocation scenarios.
     *
     * @param level Desired logging verbosity level
     *
     * **Performance Impact**: Higher levels may reduce allocation performance
     * **Thread Safety**: Thread-safe atomic operation
     * **Default**: INFO level logging
     *
     * **Levels**:
     * - SILENT: No logging output
     * - ERROR: Only critical issues
     * - WARNING: Warnings and errors
     * - INFO: General information (default)
     * - DEBUG: Detailed debugging information
     * - TRACE: Maximum detail (performance impact)
     */
    MEMORY_API void SetLoggingLevel(tracking_log_level level) noexcept;

    /**
     * @brief Gets current logging verbosity level.
     *
     * @return Current logging level setting
     *
     * **Performance**: O(1) atomic read
     * **Thread Safety**: Thread-safe atomic operation
     */
    MEMORY_API tracking_log_level GetLoggingLevel() const noexcept;

    /**
     * @brief Resets all performance timing statistics.
     *
     * Clears all timing data while preserving allocation records and
     * memory usage statistics. Useful for performance benchmarking
     * and testing scenarios.
     *
     * **Thread Safety**: Thread-safe with exclusive lock
     * **Performance**: O(1) - simple atomic resets
     * **Preservation**: Keeps allocation records and memory statistics
     */
    MEMORY_API void ResetTimingStats() noexcept;

    /**
     * @brief Calculates current allocation efficiency metrics.
     *
     * Analyzes allocation patterns to determine memory usage efficiency
     * including internal fragmentation, allocation overhead, and
     * utilization ratios.
     *
     * @return Tuple of (utilization_ratio, overhead_ratio, efficiency_score)
     *
     * **Return Values**:
     * - utilization_ratio: Requested bytes / Allocated bytes [0.0, 1.0]
     * - overhead_ratio: Overhead bytes / Total bytes [0.0, 1.0]
     * - efficiency_score: Overall efficiency metric [0.0, 1.0]
     *
     * **Thread Safety**: Thread-safe with shared lock
     * **Performance**: O(1) for cached metrics, O(n) for detailed analysis
     */
    MEMORY_API std::tuple<double, double, double> GetEfficiencyMetrics() const;

    /**
     * @brief Generates comprehensive memory usage report.
     *
     * Creates detailed textual report of memory usage patterns, performance
     * statistics, and optimization recommendations for debugging and analysis.
     *
     * @param include_allocations Whether to include individual allocation details
     * @return Formatted string report with comprehensive analysis
     *
     * **Thread Safety**: Thread-safe with shared lock
     * **Performance**: O(n) where n is number of allocations if detailed
     * **Format**: Human-readable text with structured sections
     *
     * **Report Sections**:
     * - Memory usage summary
     * - Performance timing analysis
     * - Fragmentation metrics
     * - Efficiency recommendations
     * - Optional allocation details
     */
    MEMORY_API std::string GenerateReport(bool include_allocations = false) const;

protected:
    /**
     * @brief Protected destructor for reference-counted lifecycle management.
     *
     * Destructor is protected to prevent direct deletion. The wrapper uses
     * reference counting and automatically deletes itself when the reference
     * count reaches zero and all allocations have been deallocated.
     *
     * **Lifecycle**: Only called by UnRef() when reference count reaches zero
     * **Thread Safety**: Destructor assumes no concurrent access
     */
    ~allocator_tracking() override = default;

private:
    /**
     * @brief Decrements reference count and handles self-destruction.
     *
     * @return true if wrapper should continue to exist, false if destroyed
     *
     * **Reference Counting**: Decrements ref_ and destroys wrapper if zero
     * **Thread Safety**: Requires mutex protection (caller must hold mu_)
     * **Self-Destruction**: May delete this object before returning
     */
    bool UnRef() MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mu_);

    // ========== Core Components ==========

    /**
     * @brief Underlying allocator being wrapped.
     *
     * Pointer to the actual allocator that performs memory allocation
     * and deallocation. Not owned by this wrapper - caller retains ownership.
     *
     * **Ownership**: Not owned - caller responsible for lifetime management
     * **Thread Safety**: Assumed to be thread-safe by underlying implementation
     */
    Allocator* allocator_;

    /**
     * @brief Mutex protecting all mutable state.
     *
     * Provides thread-safe access to all tracking data structures and
     * statistics. Marked mutable to allow const methods to acquire locks.
     *
     * **Granularity**: Protects all mutable members
     * **Performance**: Fine-grained locking minimizes contention
     */
    mutable std::mutex mu_;

    // ========== Reference Counting and Lifecycle ==========

    /**
     * @brief Reference count for safe wrapper lifecycle management.
     *
     * Tracks outstanding allocations plus one reference for the executor.
     * When this reaches zero, the wrapper automatically deletes itself.
     *
     * **Initial Value**: 1 (for executor reference)
     * **Increment**: On each successful allocation
     * **Decrement**: On each deallocation and GetRecordsAndUnRef()
     * **Self-Destruction**: Wrapper deletes itself when count reaches zero
     */
    int ref_ MEMORY_GUARDED_BY(mu_){1};

    // ========== Memory Usage Tracking ==========

    /**
     * @brief Current outstanding allocated bytes.
     *
     * Tracks the current number of bytes allocated through this wrapper
     * that have not yet been deallocated. Set to 0 if underlying allocator
     * doesn't track allocation sizes.
     *
     * **Updates**: Incremented on allocation, decremented on deallocation
     * **Accuracy**: Reflects actual allocated sizes when available
     */
    size_t allocated_ MEMORY_GUARDED_BY(mu_){0};

    /**
     * @brief High watermark of peak memory usage.
     *
     * Records the maximum number of bytes that were simultaneously
     * allocated through this wrapper. Set to 0 if underlying allocator
     * doesn't track allocation sizes.
     *
     * **Updates**: Updated when allocated_ reaches new peak
     * **Use Cases**: Peak memory usage analysis, resource planning
     */
    size_t high_watermark_ MEMORY_GUARDED_BY(mu_){0};

    /**
     * @brief Total bytes allocated through this wrapper.
     *
     * Cumulative total of all bytes allocated through this wrapper.
     * If underlying allocator tracks sizes, this reflects actual allocated
     * bytes; otherwise, it reflects requested bytes.
     *
     * **Accumulation**: Never decreases, only increases with allocations
     * **Interpretation**: Actual vs requested bytes depends on underlying allocator
     */
    size_t total_bytes_ MEMORY_GUARDED_BY(mu_){0};

    // ========== Allocation Records and History ==========

    /**
     * @brief Complete history of allocation records.
     *
     * Maintains detailed records of all allocations including sizes
     * and timestamps for comprehensive memory usage analysis.
     *
     * **Growth**: Grows with each allocation
     * **Memory**: May consume significant memory for long-running operations
     * **Use Cases**: Debugging, profiling, memory usage analysis
     */
    std::vector<alloc_record> allocations_ MEMORY_GUARDED_BY(mu_);

    // ========== Local Size Tracking ==========

    /**
     * @brief Whether to maintain local size tracking.
     *
     * Immutable flag set during construction that determines whether
     * to maintain local allocation size information when the underlying
     * allocator doesn't provide size tracking.
     *
     * **Immutable**: Set during construction, never changes
     * **Performance**: Enables size tracking for non-tracking allocators
     */
    const bool track_sizes_locally_;

    /**
     * @brief Local allocation metadata for size tracking.
     *
     * Contains essential information about individual allocations
     * when local size tracking is enabled.
     */
    struct Chunk
    {
        size_t  requested_size{0};  ///< Originally requested size
        size_t  allocated_size{0};  ///< Actual allocated size
        int64_t allocation_id{0};   ///< Unique allocation identifier
    };

    /**
     * @brief Map of active allocations for local size tracking.
     *
     * Maintains mapping from memory pointers to allocation metadata
     * when local size tracking is enabled. Used to provide size
     * information for allocators that don't track sizes natively.
     *
     * **Key**: Memory pointer returned by allocator
     * **Value**: Chunk metadata with size and ID information
     * **Lifecycle**: Entries added on allocation, removed on deallocation
     */
    logging::logging_map<const void*, Chunk> in_use_ MEMORY_GUARDED_BY(mu_);

    /**
     * @brief Counter for generating unique allocation IDs.
     *
     * Provides unique identifiers for each allocation when local
     * tracking is enabled. Incremented for each new allocation.
     *
     * **Uniqueness**: Each allocation gets a different positive ID
     * **Thread Safety**: Protected by mu_ mutex
     */
    int64_t next_allocation_id_ MEMORY_GUARDED_BY(mu_){1};

    // ========== Enhanced Analytics and Performance Tracking ==========

    /**
     * @brief Performance timing statistics for allocation operations.
     *
     * Thread-safe atomic counters for tracking allocation and deallocation
     * performance with microsecond precision. Used for performance analysis
     * and optimization of memory allocation patterns.
     *
     * **Thread Safety**: All members are atomic for lock-free access
     * **Precision**: Microsecond resolution using steady_clock
     * **Performance**: Minimal overhead for timing collection
     */
    mutable atomic_timing_stats timing_stats_;

    /**
     * @brief Enhanced allocation records with comprehensive metadata.
     *
     * Maintains detailed history of allocations including source location,
     * timing information, and custom tags for advanced debugging and
     * profiling analysis.
     *
     * **Growth**: Grows with each allocation when detailed tracking enabled
     * **Memory**: May consume significant memory for long-running operations
     * **Thread Safety**: Protected by shared_mutex for concurrent read access
     */
    mutable std::vector<enhanced_alloc_record> enhanced_records_ MEMORY_GUARDED_BY(shared_mu_);

    /**
     * @brief Current logging verbosity level.
     *
     * Atomic flag controlling the level of detail in logging output for
     * memory allocation operations. Higher levels provide more information
     * but may impact performance.
     *
     * **Thread Safety**: Atomic for lock-free access
     * **Default**: INFO level logging
     * **Performance**: Checked on each allocation/deallocation
     */
    std::atomic<tracking_log_level> log_level_{tracking_log_level::INFO};

    /**
     * @brief Shared mutex for enhanced analytics operations.
     *
     * Provides efficient concurrent read access to enhanced analytics data
     * while maintaining exclusive write access for updates. Used for
     * operations that need to read large data structures without blocking
     * other readers.
     *
     * **Usage**: Shared locks for read operations, exclusive locks for writes
     * **Performance**: Allows multiple concurrent readers
     * **Scope**: Protects enhanced_records_ and fragmentation analysis
     */
    mutable std::shared_mutex shared_mu_;

    /**
     * @brief Flag indicating whether enhanced tracking is enabled.
     *
     * Immutable flag set during construction that determines whether to
     * collect enhanced allocation metadata including source location,
     * timing details, and comprehensive statistics.
     *
     * **Immutable**: Set during construction, never changes
     * **Performance**: Enables/disables enhanced tracking overhead
     * **Default**: Enabled for debug builds, disabled for release builds
     */
    const bool enhanced_tracking_enabled_;

    /**
     * @brief Cached fragmentation metrics for performance optimization.
     *
     * Maintains cached fragmentation analysis to avoid expensive recalculation
     * on every request. Updated periodically or when significant allocation
     * pattern changes are detected.
     *
     * **Caching**: Updated on allocation pattern changes
     * **Thread Safety**: Protected by shared_mu_
     * **Performance**: Avoids O(n) recalculation for frequent queries
     */
    mutable memory_fragmentation_metrics cached_fragmentation_ MEMORY_GUARDED_BY(shared_mu_);

    /**
     * @brief Timestamp of last fragmentation metrics update.
     *
     * Tracks when fragmentation metrics were last calculated to determine
     * when cache invalidation and recalculation is needed.
     *
     * **Units**: Microseconds since epoch
     * **Thread Safety**: Protected by shared_mu_
     * **Usage**: Cache invalidation and update scheduling
     */
    mutable std::atomic<int64_t> last_fragmentation_update_{0};
};

}  // namespace memory
