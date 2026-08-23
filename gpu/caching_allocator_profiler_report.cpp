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

#include "gpu/caching_allocator_profiler_report.h"

#if MEMORY_HAS_PROFILER

#include "common/instrumentation.h"

namespace memory::gpu
{

void report_caching_allocator_delta(
    void* ptr, const unified_cache_stats& before, const unified_cache_stats& after,
    int device_index, int16_t device_type)
{
    const auto delta = static_cast<int64_t>(after.bytes_allocated) -
                        static_cast<int64_t>(before.bytes_allocated);
    profiler::report_memory_usage(
        ptr, delta, after.bytes_allocated, after.bytes_reserved, device_type,
        static_cast<int16_t>(device_index));
}

}  // namespace memory::gpu

#endif  // MEMORY_HAS_PROFILER
