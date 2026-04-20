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

#include "helper/process_state.h"

#include <fmt/format.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "backend/allocator_bfc.h"
#include "backend/allocator_pool.h"
#include "backend/allocator_tracking.h"
#include "common/memory_macros.h"
#include "cpu/allocator.h"
#include "util/exception.h"

namespace memory
{

/*static*/ process_state* process_state::singleton()
{
    static auto* instance = new process_state;
    return instance;
}

process_state::process_state() : numa_enabled_(false), cpu_allocators_cached_(0) {}

std::string process_state::MemDesc::debug_string() const
{
    return fmt::format(
        "{}{}, dma: {}, nic: {}",
        (loc == CPU ? "CPU " : "GPU "),
        dev_index,
        gpu_registered,
        nic_registered);
}

process_state::MemDesc process_state::PtrType(const void* ptr)
{
    if (FLAGS_brain_gpu_record_mem_types)
    {
        auto iter = mem_desc_map_.find(ptr);
        if (iter != mem_desc_map_.end())
        {
            return iter->second;
        }
    }
    return {};
}

Allocator* process_state::GetCPUAllocator(int numa_node)
{
    if (!numa_enabled_ || numa_node == NUMANOAFFINITY)
    {
        numa_node = 0;
    }

    // Check if allocator for the numa node is in lock-free cache.
    if (numa_node < cpu_allocators_cached_.load(std::memory_order_acquire))
    {
        return cpu_allocators_cache_[numa_node];
    }

    std::scoped_lock const lock(mu_);
    while (cpu_allocators_.size() <= static_cast<size_t>(numa_node))
    {
        // If visitors have been defined we need an Allocator built from
        // a sub_allocator.  Prefer allocator_bfc, but fall back to allocator_pool
        // depending on env var setting.
        const bool alloc_visitors_defined =
            (!cpu_alloc_visitors_.empty() || !cpu_free_visitors_.empty());

        const bool use_allocator_bfc      = false;
        bool const use_allocator_tracking = false;
        //fixme:
        //MEMORY_UNUSED auto status = logging::utils::read_env_bool(
        //    "CPU_ALLOCATOR_USE_BFC", alloc_visitors_defined, &use_allocator_bfc);

        Allocator*     allocator = nullptr;
        sub_allocator* sub_allocator =
            (numa_enabled_ || alloc_visitors_defined || use_allocator_bfc)
                ? new basic_cpu_allocator(
                      numa_enabled_ ? numa_node : -1, cpu_alloc_visitors_, cpu_free_visitors_)
                : nullptr;
        if (use_allocator_bfc)
        {
            // TODO(reedwm): evaluate whether 64GB by default is the best choice.
            int64_t cpu_mem_limit_in_mb = -1;
            //fixme:
            //MEMORY_UNUSED auto const status2 = logging::utils::read_env_int64(
            //    "CPU_BFC_MEM_LIMIT_IN_MB", 1LL << 16 /*64GB max by default*/, &cpu_mem_limit_in_mb);
            int64_t const cpu_mem_limit = cpu_mem_limit_in_mb * (1LL << 20);
            MEMORY_CHECK_DEBUG(sub_allocator != nullptr);

            allocator_bfc::Options allocator_opts;
            allocator_opts.allow_growth = true;

            allocator = new allocator_bfc(
                std::unique_ptr<memory::sub_allocator>(sub_allocator),
                cpu_mem_limit,
                /*name=*/"bfc_cpu_allocator_for_gpu",
                allocator_opts);

            MEMORY_LOG_INFO(
                "Using allocator_bfc with memory limit of {} MB for process_state CPU allocator",
                cpu_mem_limit_in_mb);
        }
        else if (sub_allocator != nullptr)
        {
            allocator = new allocator_pool(
                /*pool_size_limit=*/100,
                /*auto_resize=*/true,
                std::unique_ptr<memory::sub_allocator>(sub_allocator),
                std::unique_ptr<round_up_interface>(new NoopRounder),
                "cpu_pool");

            MEMORY_LOG_INFO(
                "Using allocator_pool for process_state CPU allocator numa_enabled_={} "
                "numa_node={}",
                numa_enabled_,
                numa_node);
        }
        else
        {
            allocator = allocator_cpu_base();
        }

        if constexpr (use_allocator_tracking && !allocator->tracks_allocation_sizes())
        {
            allocator = new allocator_tracking(allocator, true);
        }

        cpu_allocators_.push_back(allocator);

        if (cpu_allocators_.size() < cpu_allocators_cache_.max_size())
        {
            cpu_allocators_cache_[cpu_allocators_.size() - 1] = allocator;
            cpu_allocators_cached_.fetch_add(1, std::memory_order_release);
        }
        if (sub_allocator == nullptr)
        {
            MEMORY_CHECK_DEBUG(cpu_alloc_visitors_.empty() && cpu_free_visitors_.empty());
        }
    }
    return cpu_allocators_[numa_node];
}

void process_state::AddCPUAllocVisitor(sub_allocator::Visitor visitor)
{
    MEMORY_LOG_INFO("AddCPUAllocVisitor");
    std::scoped_lock const lock(mu_);
    MEMORY_CHECK(
        cpu_allocators_.empty(),
        "AddCPUAllocVisitor must be called prior to first call to "
        "process_state::GetCPUAllocator");
    cpu_alloc_visitors_.push_back(std::move(visitor));
}

void process_state::AddCPUFreeVisitor(sub_allocator::Visitor visitor)
{
    std::scoped_lock const lock(mu_);
    MEMORY_CHECK(
        cpu_allocators_.empty(),
        "AddCPUFreeVisitor must be called prior to first call to "
        "process_state::GetCPUAllocator");

    cpu_free_visitors_.push_back(std::move(visitor));
}

void process_state::TestOnlyReset()
{
    std::scoped_lock const lock(mu_);
    // Don't delete this value because it's static.
    Allocator const* default_cpu_allocator = allocator_cpu_base();
    mem_desc_map_.clear();
    for (Allocator const* a : cpu_allocators_)
    {
        if (a != default_cpu_allocator)
        {
            delete a;
        }
    }
    cpu_allocators_.clear();
    for (Allocator const* a : cpu_al_)
    {
        delete a;
    }
    cpu_al_.clear();
}

}  // namespace memory
