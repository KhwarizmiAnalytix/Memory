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

#include "common/memory_macros.h"

#if MEMORY_HAS_PROFILER

#include <cstdint>

#include "common/memory_export.h"
#include "profiler/unified_memory_stats.h"

namespace memory::gpu
{

/**
 * @brief Reports one caching-allocator allocate/deallocate event to the
 * active profiling session (profiler::report_memory_usage), shared by
 * cuda_caching_allocator.cpp and metal_caching_allocator.mm.
 *
 * Reports the real change in the allocator's tracked byte count (@p before
 * vs. @p after), rather than trusting a caller-supplied size: Impl::allocate()
 * rounds requests up to its block-size policy, and Impl::deallocate()'s own
 * `size` parameter is unused -- it looks up the real freed size internally --
 * so a caller-supplied size (e.g. allocator<T>::free()'s hardcoded 0 at
 * Library/Memory/allocator.h) would otherwise silently misreport.
 *
 * @param device_type Raw profiler::device_enum value (CPU=0, CUDA=1, HIP=2,
 *        PrivateUse1=3) -- passed as int16_t to avoid pulling in the internal
 *        bespoke/common/orchestration/observer.h header.
 */
MEMORY_API void report_caching_allocator_delta(
    void*                      ptr,
    const unified_cache_stats& before,
    const unified_cache_stats& after,
    int                        device_index,
    int16_t                    device_type);

}  // namespace memory::gpu

#endif  // MEMORY_HAS_PROFILER
