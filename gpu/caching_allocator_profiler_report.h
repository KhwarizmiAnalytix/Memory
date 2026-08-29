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

#include "common/memory_macros.h"

#if MEMORY_HAS_PROFILER

#include <cstdint>

#include "common/instrumentation.h"
#include "common/memory_export.h"
#include "profiler/unified_memory_stats.h"

namespace memory::gpu
{

/**
 * @brief Reports one caching-allocator allocate/deallocate to the active
 * session, mirroring CUDACachingAllocator's reportMemoryUsageToProfiler.
 *
 * @p alloc_size is the known block size (positive alloc, negative free) --
 * not a caller-supplied size, which deallocate ignores. Predicted-false
 * `memory_profiling_active()` so idle sessions skip the Kineto hop.
 *
 * @param device_type Raw profiler::device_enum value (CPU=0, CUDA=1, HIP=2,
 *        PrivateUse1=3).
 */
MEMORY_FORCE_INLINE void report_caching_allocator_event(
    void*   ptr,
    int64_t alloc_size,
    size_t  total_allocated,
    size_t  total_reserved,
    int     device_index,
    int16_t device_type)
{
    if MEMORY_UNLIKELY (profiler::memory_profiling_active())
    {
        profiler::report_memory_usage(
            ptr,
            alloc_size,
            total_allocated,
            total_reserved,
            device_type,
            static_cast<int16_t>(device_index));
    }
}

/**
 * @brief Reports a caching-allocator OOM (`c10::reportOutOfMemoryToProfiler`).
 */
MEMORY_FORCE_INLINE void report_caching_allocator_oom(
    int64_t alloc_size,
    size_t  total_allocated,
    size_t  total_reserved,
    int     device_index,
    int16_t device_type)
{
    if MEMORY_UNLIKELY (profiler::memory_profiling_active())
    {
        profiler::report_out_of_memory(
            alloc_size,
            total_allocated,
            total_reserved,
            device_type,
            static_cast<int16_t>(device_index));
    }
}

/**
 * @brief Test helper: diffs two `unified_cache_stats` snapshots into one
 * `report_caching_allocator_event`. Production allocators report the known
 * block size instead of scanning pools.
 */
MEMORY_API void report_caching_allocator_delta(
    void*                      ptr,
    const unified_cache_stats& before,
    const unified_cache_stats& after,
    int                        device_index,
    int16_t                    device_type);

}  // namespace memory::gpu

#endif  // MEMORY_HAS_PROFILER
