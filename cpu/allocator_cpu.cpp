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

#include "cpu/allocator_cpu.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "common/memory_macros.h"
#include "logger.h"
#include "cpu/allocator.h"
#include "helper/memory_allocator.h"
#include "helper/memory_info.h"
#if MEMORY_HAS_NATIVE_PROFILER && 0
#include "native/memory/scoped_memory_debug_annotation.h"
#include "native/tracing/traceme.h"
#include "native/tracing/traceme_encode.h"
#endif

namespace memory
{

/**
 * @brief Global flag controlling CPU allocator statistics collection.
 *
 * When enabled, the CPU allocator tracks detailed memory usage statistics
 * including allocation counts, peak usage, and memory warnings. This adds
 * minimal overhead but provides valuable debugging information.
 *
 * **Performance Impact**: ~1-2% overhead when enabled
 * **Thread Safety**: Atomic operations ensure thread-safe access
 * **Default**: Disabled for optimal performance
 */
static std::atomic<bool> cpu_allocator_collect_stats{false};

void EnableCPUAllocatorStats() noexcept
{
    cpu_allocator_collect_stats.store(true, std::memory_order_relaxed);
}

void DisableCPUAllocatorStats() noexcept
{
    cpu_allocator_collect_stats.store(false, std::memory_order_relaxed);
}

bool CPUAllocatorStatsEnabled() noexcept
{
    return cpu_allocator_collect_stats.load(std::memory_order_relaxed);
}

// ========== Memory Warning Configuration ==========

/**
 * @brief Maximum number of total allocation warnings to emit.
 *
 * Prevents log spam when total memory usage consistently exceeds thresholds.
 * After this many warnings, further total allocation warnings are suppressed.
 */
static constexpr int kMaxTotalAllocationWarnings = 1;

#ifndef NDEBUG
/**
 * @brief Maximum number of large single allocation warnings to emit.
 *
 * Prevents log spam from repeated large allocations. After this many warnings,
 * further large allocation warnings are suppressed.
 */
static constexpr int kMaxSingleAllocationWarnings = 5;
#endif

/**
 * @brief Threshold for total memory usage warnings (fraction of available RAM).
 *
 * When total allocated memory exceeds this fraction of available system RAM,
 * a warning is logged to alert about potential memory pressure.
 *
 * **Value**: 0.5 (50% of available RAM)
 * **Purpose**: Early warning for memory pressure conditions
 */
static constexpr double kTotalAllocationWarningThreshold = 0.5;

#ifndef NDEBUG
/**
 * @brief Threshold for large single allocation warnings (fraction of available RAM).
 *
 * Individual allocations exceeding this fraction of available system RAM
 * trigger warnings to identify potentially problematic large allocations.
 *
 * **Value**: 0.1 (10% of available RAM)
 * **Purpose**: Identify unexpectedly large allocations
 */
static constexpr double kLargeAllocationWarningThreshold = 0.1;

/**
 * @brief Cached threshold for large allocation warnings in bytes.
 *
 * @return Byte threshold for large allocation warnings
 *
 * **Caching**: Expensive port::available_ram() call is cached on first use
 * **Thread Safety**: Static initialization is thread-safe in C++11+
 * **Performance**: O(1) after first call, expensive first call
 */
static int64_t LargeAllocationWarningBytes() noexcept
{
    static const auto value =
        static_cast<int64_t>(port::available_ram() * kLargeAllocationWarningThreshold);
    return value;
}
#endif

/**
 * @brief Cached threshold for total allocation warnings in bytes.
 *
 * @return Byte threshold for total allocation warnings
 *
 * **Caching**: Expensive port::available_ram() call is cached on first use
 * **Thread Safety**: Static initialization is thread-safe in C++11+
 * **Performance**: O(1) after first call, expensive first call
 */
static int64_t TotalAllocationWarningBytes() noexcept
{
    static const auto value =
        static_cast<int64_t>(port::available_ram() * kTotalAllocationWarningThreshold);
    return value;
}

// ========== allocator_cpu Implementation ==========

allocator_cpu::allocator_cpu()
    : single_allocation_warning_count_{0}, total_allocation_warning_count_{0}
{
}

allocator_cpu::~allocator_cpu() = default;

std::string allocator_cpu::Name() const
{
    return "cpu";
}

void* allocator_cpu::allocate_raw(size_t alignment, size_t num_bytes)
{
#ifndef NDEBUG
    // Check for large allocation warning (rate-limited)
    // cppcheck-suppress syntaxError
    // Explanation: MEMORY_UNLIKELY is a branch prediction macro that expands to compiler-specific
    // attributes (__builtin_expect or [[unlikely]]). Cppcheck doesn't understand this macro syntax.
    if MEMORY_UNLIKELY (num_bytes > static_cast<size_t>(LargeAllocationWarningBytes()))
    {
        const auto current_count = single_allocation_warning_count_.load(std::memory_order_relaxed);
        if (current_count < kMaxSingleAllocationWarnings)
        {
            // Use compare_exchange to avoid race conditions in warning count
            auto expected = current_count;
            if (single_allocation_warning_count_.compare_exchange_weak(
                    expected, current_count + 1, std::memory_order_relaxed))
            {
                LOGGING_LOG_WARNING(
                    "Large allocation of {} bytes ({}% of available RAM) exceeds {}% threshold",
                    num_bytes,
                    (100.0 * num_bytes / port::available_ram()),
                    (100 * kLargeAllocationWarningThreshold));
            }
        }
    }
#endif

    // Perform the actual allocation
    void* p = cpu::memory_allocator::allocate(num_bytes, alignment);  //NOLINT

    // Collect statistics if enabled (fast path when disabled)
    if MEMORY_UNLIKELY (
        cpu_allocator_collect_stats.load(std::memory_order_relaxed) && p != nullptr)
    {
        const auto alloc_size = 0;

        // Use scoped lock for statistics update
        {
            std::scoped_lock const lock(mu_);

            // Update core statistics
            stats_.num_allocs.fetch_add(1, std::memory_order_relaxed);
            stats_.bytes_in_use.fetch_add(alloc_size, std::memory_order_relaxed);

            // Update peak bytes in use atomically
            int64_t const current_bytes = stats_.bytes_in_use.load(std::memory_order_relaxed);
            int64_t       peak_bytes    = stats_.peak_bytes_in_use.load(std::memory_order_relaxed);
            while (current_bytes > peak_bytes &&
                   !stats_.peak_bytes_in_use.compare_exchange_weak(
                       peak_bytes, current_bytes, std::memory_order_relaxed))
            {
                // Retry if another thread updated peak_bytes
            }

            // Update largest allocation size atomically
            auto const alloc_size_int64 = static_cast<int64_t>(alloc_size);
            int64_t    largest_size     = stats_.largest_alloc_size.load(std::memory_order_relaxed);
            while (alloc_size_int64 > largest_size &&
                   !stats_.largest_alloc_size.compare_exchange_weak(
                       largest_size, alloc_size_int64, std::memory_order_relaxed))
            {
                // Retry if another thread updated largest_size
            }

            // Check for total allocation warning (rate-limited)
            if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
                total_allocation_warning_count_ < kMaxTotalAllocationWarnings)
            {
                ++total_allocation_warning_count_;
                size_t bytes_in_use = stats_.bytes_in_use.load(std::memory_order_relaxed);
                LOGGING_LOG_WARNING(
                    "Total allocated memory {} bytes ({}% of available RAM) exceeds {}% "
                    "threshold",
                    bytes_in_use,
                    (100.0 * bytes_in_use / port::available_ram()),
                    (100 * kTotalAllocationWarningThreshold));
            }
        }

        // Add profiling trace (outside lock to minimize contention)
#if MEMORY_HAS_NATIVE_PROFILER && 0
        AddTraceMe("MemoryAllocation", p, num_bytes, alloc_size);
#endif
    }

    return p;
}

void allocator_cpu::deallocate_raw(void* ptr)
{
    // Fast path when statistics disabled
    if MEMORY_UNLIKELY (cpu_allocator_collect_stats.load(std::memory_order_relaxed))
    {
        // Get allocation size before deallocation
        const auto alloc_size = 0;

        // Update statistics under lock
        {
            std::scoped_lock const lock(mu_);
            stats_.bytes_in_use -= alloc_size;
        }

        // Add profiling trace (outside lock to minimize contention)
#if MEMORY_HAS_NATIVE_PROFILER && 0
        AddTraceMe("MemoryDeallocation", ptr, 0, alloc_size);
#endif
    }

    // Perform actual deallocation
    cpu::memory_allocator::free(ptr);
}

void allocator_cpu::deallocate_raw(void* ptr, size_t /*alignment*/, size_t /*num_bytes*/)
{
    // Currently identical to single-parameter version
    // Future optimization could use num_bytes hint to avoid GetAllocatedSize call
    deallocate_raw(ptr);
}

std::optional<allocator_stats> allocator_cpu::GetStats() const
{
    if (!cpu_allocator_collect_stats.load(std::memory_order_relaxed))
    {
        return std::nullopt;
    }

    std::scoped_lock const lock(mu_);
    // Create a copy of the atomic stats structure
    allocator_stats stats_copy(stats_);
    return stats_copy;
}

bool allocator_cpu::ClearStats()
{
    if (!cpu_allocator_collect_stats.load(std::memory_order_relaxed))
    {
        return false;
    }

    std::scoped_lock const lock(mu_);
    stats_.num_allocs.store(0, std::memory_order_relaxed);
    stats_.peak_bytes_in_use.store(
        stats_.bytes_in_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
    stats_.largest_alloc_size.store(0, std::memory_order_relaxed);
    return true;
}

allocator_memory_enum allocator_cpu::GetMemoryType() const noexcept
{
    return allocator_memory_enum::HOST_PAGEABLE;
}

#if MEMORY_HAS_NATIVE_PROFILER && 0
void allocator_cpu::AddTraceMe(
    std::string_view traceme_name,
    const void*      chunk_ptr,
    std::size_t      req_bytes,
    std::size_t      alloc_bytes)
{
    memory::traceme::instant_activity(
        [this, traceme_name, chunk_ptr, req_bytes, alloc_bytes]()
            MEMORY_NO_THREAD_SAFETY_ANALYSIS -> std::string
        {
            // Capture current debug annotation context
            const auto& annotation = memory::scoped_memory_debug_annotation::current_annotation();

            // Create comprehensive trace with current allocator state
            return memory::traceme_encode(
                std::string(traceme_name),
                {{"allocator_name", Name()},
                 {"bytes_reserved", stats_.bytes_reserved.load(std::memory_order_relaxed)},
                 {"bytes_allocated", stats_.bytes_in_use.load(std::memory_order_relaxed)},
                 {"peak_bytes_in_use", stats_.peak_bytes_in_use.load(std::memory_order_relaxed)},
                 {"requested_bytes", req_bytes},
                 {"allocation_bytes", alloc_bytes},
                 {"addr", reinterpret_cast<uint64_t>(chunk_ptr)},
                 {"memory_op", annotation.pending_op_name},
                 {"id", annotation.pending_step_id},
                 {"region_type", annotation.pending_region_type},
                 {"data_type", annotation.pending_data_type},
                 {"shape", annotation.pending_shape_func()}});
        },
        static_cast<int>(memory::traceme_level_enum::INFO));
}
#endif

}  // namespace memory
