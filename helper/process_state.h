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

#include <array>
#include <functional>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/memory_export.h"
#include "common/memory_macros.h"
#include "cpu/allocator.h"

#if 0
#include "util/flat_hash.h"

template <typename T>
using memory_set = flat_hash_set<T>;
template <typename K, typename V, typename H = std::hash<K>>
using memory_map = flat_hash_map<K, V, H>;
#else
#include <unordered_map>
#include <unordered_set>

template <typename T>
using memory_set = std::unordered_set<T>;
template <typename K, typename V>
using memory_map = std::unordered_map<K, V>;

#endif

namespace memory
{

class allocator_pool;

// Singleton that manages per-process state, e.g. allocation of
// shared resources.
class MEMORY_VISIBILITY process_state
{
public:
    MEMORY_API static process_state* singleton();

    // Descriptor for memory allocation attributes, used by optional
    // runtime correctness analysis logic.
    struct MemDesc
    {
        enum MemLoc
        {
            CPU,
            GPU
        };
        MemLoc loc;
        int    dev_index;
        bool   gpu_registered;
        bool   nic_registered;
        MemDesc() : loc(CPU), dev_index(0), gpu_registered(false), nic_registered(false) {}
        MEMORY_API std::string debug_string() const;
    };

    // If NUMA Allocators are desired, call this before calling any
    // Allocator accessor.
    void EnableNUMA() { numa_enabled_ = true; }

    // Returns what we know about the memory at ptr.
    // If we know nothing, it's called CPU 0 with no other attributes.
    MEMORY_API MemDesc PtrType(const void* ptr);

    // Returns the one cpu_allocator used for the given numa_node.
    // Treats numa_node == NUMANOAFFINITY as numa_node == 0.
    MEMORY_API Allocator* GetCPUAllocator(int numa_node);

    // Registers alloc visitor for the CPU allocator(s).
    // REQUIRES: must be called before GetCPUAllocator.
    MEMORY_API void AddCPUAllocVisitor(sub_allocator::Visitor v);

    // Registers free visitor for the CPU allocator(s).
    // REQUIRES: must be called before GetCPUAllocator.
    MEMORY_API void AddCPUFreeVisitor(sub_allocator::Visitor v);

    typedef memory_map<const void*, MemDesc> MDMap;

    // Helper method for unit tests to reset the process_state singleton by
    // cleaning up everything. Never use in production.
    MEMORY_API void TestOnlyReset();

protected:
    process_state();
    virtual ~process_state() {}
    friend class gpu_process_State;
    friend class pluggable_device_process_state;

    // If these flags need to be runtime configurable consider adding
    // them to ConfigProto.
    static constexpr bool FLAGS_brain_mem_reg_gpu_dma      = true;
    static constexpr bool FLAGS_brain_gpu_record_mem_types = false;

    static process_state* instance_;
    bool                  numa_enabled_;

    mutable std::mutex mu_;

    // Indexed by numa_node.  If we want numa-specific allocators AND a
    // non-specific allocator, maybe should index by numa_node+1.
    std::vector<Allocator*> cpu_allocators_                 MEMORY_GUARDED_BY(mu_);
    std::vector<sub_allocator::Visitor> cpu_alloc_visitors_ MEMORY_GUARDED_BY(mu_);
    std::vector<sub_allocator::Visitor> cpu_free_visitors_  MEMORY_GUARDED_BY(mu_);

    // A cache of cpu allocators indexed by a numa node. Used as a fast path to
    // get CPU allocator by numa node id without locking the mutex. We can't use
    // `cpu_allocators_` storage in the lock-free path because concurrent
    // operation can deallocate the vector storage.
    std::atomic<int>          cpu_allocators_cached_;
    std::array<Allocator*, 8> cpu_allocators_cache_;

    // Optional RecordingAllocators that wrap the corresponding
    // Allocators for runtime attribute use analysis.
    MDMap                           mem_desc_map_;
    std::vector<Allocator*> cpu_al_ MEMORY_GUARDED_BY(mu_);
};

namespace internal
{
class recording_allocator : public Allocator
{
public:
    recording_allocator(
        process_state::MDMap* mm, Allocator* a, process_state::MemDesc md, std::mutex* mu)
        : mm_(mm), a_(a), md_(md), mu_(mu)
    {
    }

    std::string Name() const override { return a_->Name(); }
    void*       allocate_raw(size_t alignment, size_t num_bytes) override
    {
        void*                       p = a_->allocate_raw(alignment, num_bytes);
        std::lock_guard<std::mutex> l(*mu_);
        (*mm_)[p] = md_;
        return p;
    }
    void deallocate_raw(void* p) override
    {
        std::lock_guard<std::mutex> l(*mu_);
        auto                        iter = mm_->find(p);
        mm_->erase(iter);
        a_->deallocate_raw(p);
    }
    bool tracks_allocation_sizes() const noexcept override { return a_->tracks_allocation_sizes(); }
    size_t RequestedSize(const void* p) const override { return a_->RequestedSize(p); }
    size_t AllocatedSize(const void* p) const override { return a_->AllocatedSize(p); }
    std::optional<allocator_stats> GetStats() const override { return a_->GetStats(); }
    bool                           ClearStats() override { return a_->ClearStats(); }

    allocator_memory_enum GetMemoryType() const noexcept override { return a_->GetMemoryType(); }

    process_state::MDMap*  mm_;  // not owned
    Allocator*             a_;   // not owned
    process_state::MemDesc md_;
    std::mutex*            mu_;
};
}  // namespace internal
}  // namespace memory
