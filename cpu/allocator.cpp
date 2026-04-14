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

#include "cpu/allocator.h"

#include <cstddef>
#include <string>
#include <vector>

#include "fmt/format.h"
#include "backend/allocator_tracking.h"
#include "cpu/allocator_cpu.h"
#include "helper/process_state.h"

namespace memory
{
// If true, cpu allocator collects full stats.
static bool cpu_allocator_collect_full_stats = false;

void EnableCPUAllocatorFullStats()
{
    cpu_allocator_collect_full_stats = true;
}
bool CPUAllocatorFullStatsEnabled()
{
    return cpu_allocator_collect_full_stats;
}

std::string allocator_attributes::debug_string() const
{
    return fmt::format(
        "allocator_attributes(on_host={} nic_compatible={} gpu_compatible={})",
        on_host(),
        nic_compatible(),
        gpu_compatible());
}

Allocator* allocator_cpu_base()
{
    static Allocator* cpu_alloc = new allocator_cpu();

    if (cpu_allocator_collect_full_stats && !cpu_alloc->tracks_allocation_sizes())
    {
        cpu_alloc = new allocator_tracking(cpu_alloc, true);
    }
    return cpu_alloc;
}

Allocator* cpu_allocator(int numa_node)
{
    static auto* ps = process_state::singleton();

    if (ps != nullptr)
    {
        return ps->GetCPUAllocator(numa_node);
    }

    return allocator_cpu_base();
}

}  // namespace memory
