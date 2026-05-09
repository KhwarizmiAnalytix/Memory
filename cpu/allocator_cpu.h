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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "common/memory_macros.h"
#include "cpu/allocator.h"
#include "common/memory_export.h"

namespace memory
{

// EnableCPUAllocatorStats / DisableCPUAllocatorStats are declared in cpu/allocator.h.

/**
 * @brief Checks if CPU allocator statistics collection is enabled.
 *
 * @return true if statistics are being collected, false otherwise
 *
 * **Thread Safety**: Thread-safe atomic read
 * **Performance**: O(1) - simple atomic load
 */
//MEMORY_API bool CPUAllocatorStatsEnabled() noexcept;

/**
 * @brief High-performance CPU memory allocator with comprehensive monitoring.
 *
 * allocator_cpu provides efficient memory allocation for CPU-based computations
 * with optional statistics collection, memory pressure monitoring, and
 * integration with profiling systems.
 *
 * **Key Features**:
 * - Direct integration with optimized cpu_allocator backend
 * - Optional comprehensive statistics collection
 * - Memory pressure warnings and monitoring
 * - Profiling integration with TraceMe system
 * - Thread-safe operation with minimal contention
 *
 * **Performance Characteristics**:
 * - Allocation: O(1) - delegates to cpu::memory_allocator::allocate
 * - Deallocation: O(1) - delegates to cpu::memory_allocator::free
 * - Statistics overhead: ~1-2% when enabled, 0% when disabled
 * - Memory overhead: Minimal - only statistics when enabled
 *
 * **Thread Safety**: Fully thread-safe with fine-grained locking
 * **Memory Type**: Host pageable memory (standard system RAM)
 *
 * **Design Principles**:
 * - Minimal overhead when statistics disabled
 * - Comprehensive monitoring when statistics enabled
 * - Integration with Memory profiling ecosystem
 * - Robust warning system for memory pressure detection
 *
 * **Use Cases**:
 * - CPU-based numerical computations
 * - Temporary buffer allocation
 * - General-purpose memory management
 * - Development and debugging with statistics
 */
class MEMORY_VISIBILITY allocator_cpu : public Allocator
{
public:
    /**
     * @brief Constructs CPU allocator with default configuration.
     *
     * **Initialization**: Sets up warning counters and statistics
     * **Performance**: O(1) - minimal initialization overhead
     * **Thread Safety**: Constructor is not thread-safe
     */
    MEMORY_API allocator_cpu();

    /**
     * @brief Destructor with automatic cleanup.
     *
     * **Cleanup**: No explicit cleanup needed - uses RAII
     * **Thread Safety**: Destructor is not thread-safe
     */
    MEMORY_API ~allocator_cpu() override;

    /**
     * @brief Returns allocator name for identification and debugging.
     *
     * @return Static string "cpu" identifying this allocator type
     *
     * **Thread Safety**: Thread-safe - returns constant string
     * **Performance**: O(1) - returns static string
     * **Use Cases**: Logging, debugging, allocator identification
     */
    MEMORY_API std::string Name() const override;

    /**
     * @brief Allocates aligned memory block with comprehensive monitoring.
     *
     * Provides high-performance memory allocation with optional statistics
     * collection, memory pressure warnings, and profiling integration.
     *
     * @param alignment Required alignment in bytes (must be power of 2)
     * @param num_bytes Size of memory block to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     *
     * **Algorithm**: Delegates to optimized cpu::memory_allocator::allocate backend
     * **Performance**: O(1) - direct system allocator call + optional stats
     * **Thread Safety**: Thread-safe with fine-grained locking for statistics
     * **Memory Pressure**: Monitors and warns about large/total allocations
     *
     * **Statistics Collection** (when enabled):
     * - Updates allocation count and bytes in use
     * - Tracks peak memory usage
     * - Records largest single allocation
     * - Generates profiling traces
     *
     * **Warning System**:
     * - Large allocation warnings for individual allocations > 10% of RAM
     * - Total allocation warnings when total usage > 50% of RAM
     * - Rate-limited to prevent log spam
     *
     * **Error Handling**: Returns nullptr on allocation failure
     * **Alignment**: Guaranteed to meet or exceed requested alignment
     */
    MEMORY_API void* allocate_raw(size_t alignment, size_t num_bytes) override;

    /**
     * @brief Deallocates memory block with statistics tracking.
     *
     * Efficiently deallocates memory with optional statistics updates
     * and profiling integration. Provides the primary deallocation interface.
     *
     * @param ptr Pointer to memory block to deallocate (must not be nullptr)
     *
     * **Algorithm**: Delegates to cpu::memory_allocator::free backend
     * **Performance**: O(1) - direct system deallocator call + optional stats
     * **Thread Safety**: Thread-safe with fine-grained locking for statistics
     * **Error Handling**: Undefined behavior if ptr is invalid
     *
     * **Statistics Collection** (when enabled):
     * - Updates bytes in use counter
     * - Generates profiling deallocation trace
     * - Maintains accurate memory usage tracking
     *
     * **Profiling Integration**: Records deallocation events for analysis
     */
    MEMORY_API void deallocate_raw(void* ptr) override;

    /**
     * @brief Deallocates memory block with size and alignment hints.
     *
     * Extended deallocation interface that accepts size and alignment
     * parameters for potential optimization. Currently delegates to
     * standard deallocate_raw as cpu_allocator doesn't use these hints.
     *
     * @param ptr Pointer to memory block to deallocate
     * @param alignment Original alignment requirement (unused)
     * @param num_bytes Original allocation size (unused)
     *
     * **Algorithm**: Same as deallocate_raw(ptr) - hints are unused
     * **Performance**: O(1) - identical to single-parameter version
     * **Thread Safety**: Thread-safe with fine-grained locking
     * **Future**: May be optimized to use size/alignment hints
     */
    MEMORY_API void deallocate_raw(void* ptr, size_t alignment, size_t num_bytes) override;

    /**
     * @brief Retrieves current allocator statistics.
     *
     * @return Optional statistics object, or nullopt if statistics disabled
     *
     * **Availability**: Only available when statistics collection is enabled
     * **Performance**: O(1) - returns cached statistics under lock
     * **Thread Safety**: Thread-safe with mutex protection
     * **Consistency**: Returns atomic snapshot of current statistics
     *
     * **Statistics Include**:
     * - Total number of allocations
     * - Current bytes in use
     * - Peak bytes in use
     * - Largest single allocation size
     * - Reserved bytes (always 0 for CPU allocator)
     */
    MEMORY_API std::optional<allocator_stats> GetStats() const override;

    /**
     * @brief Resets statistics counters while preserving current state.
     *
     * @return true if statistics were reset, false if statistics disabled
     *
     * **Behavior**: Resets counters but preserves current bytes_in_use
     * **Performance**: O(1) - simple counter reset under lock
     * **Thread Safety**: Thread-safe with mutex protection
     *
     * **Reset Operations**:
     * - num_allocs → 0
     * - peak_bytes_in_use → current bytes_in_use
     * - largest_alloc_size → 0
     * - bytes_in_use remains unchanged
     */
    MEMORY_API bool ClearStats() override;

    /**
     * @brief Returns memory type managed by this allocator.
     *
     * @return allocator_memory_enum::HOST_PAGEABLE (standard system RAM)
     *
     * **Memory Type**: Host pageable memory (standard system RAM)
     * **Performance**: O(1) - returns constant
     * **Thread Safety**: Thread-safe (returns constant)
     * **Use Cases**: Memory type identification, NUMA awareness, optimization
     */
    MEMORY_API allocator_memory_enum GetMemoryType() const noexcept override;

    // Prevent copying to avoid complex state duplication
    allocator_cpu(const allocator_cpu&)            = delete;
    allocator_cpu& operator=(const allocator_cpu&) = delete;

private:
#if MEMORY_HAS_NATIVE_PROFILER
    /**
     * @brief Adds comprehensive profiling trace for memory operations.
     *
     * Integrates with Memory's profiling system to record detailed memory
     * operation traces including allocator state, operation context, and
     * memory usage statistics.
     *
     * @param traceme_name Operation name ("MemoryAllocation" or "MemoryDeallocation")
     * @param chunk_ptr Pointer to memory being traced
     * @param req_bytes Originally requested size (0 for deallocations)
     * @param alloc_bytes Actual allocated size
     *
     * **Performance**: Minimal overhead - only called when statistics enabled
     * **Thread Safety**: Thread-safe - captures consistent snapshot
     * **Integration**: Works with Memory profiling and debugging tools
     *
     * **Trace Information**:
     * - Allocator identification and current state
     * - Memory addresses and sizes
     * - Current operation context and metadata
     * - Peak usage and allocation statistics
     *
     * **Use Cases**:
     * - Memory usage profiling and analysis
     * - Allocation pattern debugging
     * - Performance optimization
     * - Memory leak detection
     */
    void AddTraceMe(
        std::string_view traceme_name,
        const void*      chunk_ptr,
        std::size_t      req_bytes,
        std::size_t      alloc_bytes);
#endif

    /**
     * @brief Mutex protecting statistics and warning counters.
     *
     * Provides thread-safe access to mutable allocator state including
     * statistics and warning counters. Marked mutable to allow const
     * methods to acquire locks for read operations.
     *
     * **Granularity**: Fine-grained - only protects statistics updates
     * **Contention**: Minimal - statistics updates are brief
     * **Performance**: Low overhead when statistics disabled
     */
    mutable std::mutex mu_;

    /**
     * @brief Comprehensive allocator statistics.
     *
     * Tracks detailed memory usage metrics including allocation counts,
     * current usage, peak usage, and largest allocation size.
     * Protected by mu_ for thread-safe access.
     *
     * **Updates**: Modified during allocation/deallocation when stats enabled
     * **Thread Safety**: Protected by mu_ mutex
     * **Persistence**: Maintained throughout allocator lifetime
     */
    allocator_stats stats_;

    /**
     * @brief Atomic counter for large allocation warnings.
     *
     * Tracks number of large allocation warnings emitted to prevent
     * log spam. Uses atomic operations for thread-safe rate limiting
     * without requiring mutex acquisition.
     *
     * **Thread Safety**: Atomic operations ensure thread-safe updates
     * **Rate Limiting**: Prevents excessive warning messages
     * **Performance**: Lock-free increment and comparison
     */
    std::atomic<int> single_allocation_warning_count_;

    /**
     * @brief Counter for total allocation warnings.
     *
     * Tracks number of total allocation warnings emitted. Protected
     * by mu_ since it's only accessed during statistics updates.
     *
     * **Thread Safety**: Protected by mu_ mutex
     * **Rate Limiting**: Prevents excessive total allocation warnings
     * **Scope**: Only incremented during statistics collection
     */
    int total_allocation_warning_count_;
};

}  // namespace memory
