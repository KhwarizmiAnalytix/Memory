/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>

#include "common/memory_export.h"

namespace memory
{

/**
 * @brief Reports CPU alloc / free / OOM to the active profiler.
 *
 * Equivalent of c10::ProfiledCPUMemoryReporter. Wired from
 * `cpu::memory_allocator` only when `MEMORY_HAS_PROFILER=1`, and then only
 * after `profiler::memory_profiling_active()` (predicted-false). Without
 * Profiler the allocate/free hot path does not call this at all.
 *
 * `total_reserved` is always 0: the CPU path is a thin dispatch over
 * mimalloc / TBB / platform malloc, not a caching pool.
 */
class MEMORY_VISIBILITY profiled_cpu_memory_reporter
{
public:
    profiled_cpu_memory_reporter() = default;

    profiled_cpu_memory_reporter(const profiled_cpu_memory_reporter&)            = delete;
    profiled_cpu_memory_reporter& operator=(const profiled_cpu_memory_reporter&) = delete;
    profiled_cpu_memory_reporter(profiled_cpu_memory_reporter&&)                 = delete;
    profiled_cpu_memory_reporter& operator=(profiled_cpu_memory_reporter&&)      = delete;

    /// Record a successful allocation (`c10::ProfiledCPUMemoryReporter::New`).
    MEMORY_API void record_allocation(void* ptr, std::size_t nbytes);

    /// Record a deallocation (`Delete`). Unknown pointers (allocated before
    /// reporting started) are ignored after a rate-limited warning when
    /// @p emit_event is true. Passing false still removes tracked state, but
    /// skips the profiler event for allocations freed after profiling stops.
    MEMORY_API void record_deallocation(void* ptr, bool emit_event = true);

    /// Record a failed allocation (`OutOfMemory`).
    MEMORY_API void record_out_of_memory(std::size_t nbytes);

    /// Running total of live recorded bytes.
    MEMORY_API std::size_t allocated() const;

    /// Number of live blocks in the size table.
    MEMORY_API std::size_t tracked_blocks() const;

    /// Fast no-lock hint for whether a free may need stale profiler cleanup.
    MEMORY_API bool has_tracked_blocks() const noexcept;

    /// OOM records since construction or the last `reset()`.
    MEMORY_API std::size_t num_ooms() const;

    /// Drop the size table and counters. Does not free live allocations.
    MEMORY_API void reset();

private:
    mutable std::mutex                     mutex_;
    std::unordered_map<void*, std::size_t> size_table_;
    std::atomic<std::size_t>               tracked_blocks_fast_{0};
    std::size_t                            allocated_{0};
    std::size_t                            num_ooms_{0};
    std::size_t                            unknown_free_log_count_{0};
};

/// Process-wide reporter used by `cpu::memory_allocator` when Profiler is linked.
MEMORY_API profiled_cpu_memory_reporter& cpu_memory_reporter();

}  // namespace memory
