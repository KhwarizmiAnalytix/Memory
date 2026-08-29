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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <vector>

#include "common/memory_export.h"

namespace memory::gpu
{

/// Default ring size for `record_memory_history`, matching PyTorch's
/// `_record_memory_history(max_entries=100000)`.
inline constexpr size_t kDefaultMemoryHistoryEntries = 100000;

/**
 * @brief One allocator action in the history ring (`c10` TraceEntry::Action).
 *
 * Stack frames (PyTorch GatheredContext) are not captured; the ring records
 * address, size, and running totals only.
 */
enum class gpu_memory_trace_action
{
    alloc,
    free_requested,
    free_completed,
    segment_alloc,
    segment_free,
    oom,
    snapshot
};

struct MEMORY_VISIBILITY gpu_memory_trace_entry
{
    gpu_memory_trace_action action{gpu_memory_trace_action::alloc};
    void*                   address{nullptr};
    size_t                  size{0};
    size_t                  total_allocated{0};
    size_t                  total_reserved{0};
    int64_t                 stream{0};
};

struct MEMORY_VISIBILITY gpu_memory_block_info
{
    void*  address{nullptr};
    size_t size{0};
    size_t requested_size{0};
    bool   allocated{false};
    bool   active{false};
};

struct MEMORY_VISIBILITY gpu_memory_segment_info
{
    void*                                address{nullptr};
    size_t                               total_size{0};
    size_t                               allocated_size{0};
    size_t                               active_size{0};
    size_t                               requested_size{0};
    bool                                 is_small{false};
    bool                                 is_expandable{false};
    int64_t                              stream{0};
    std::vector<gpu_memory_block_info>   blocks;
};

/**
 * @brief Segment map plus optional history ring (`c10` SnapshotInfo).
 *
 * Produced by `cuda_caching_allocator::snapshot` /
 * `metal_caching_allocator::snapshot`. History is empty unless
 * `record_memory_history(true)` was called.
 */
struct MEMORY_VISIBILITY gpu_memory_snapshot
{
    std::vector<gpu_memory_segment_info>  segments;
    std::vector<gpu_memory_trace_entry>   device_trace;
};

/**
 * @brief Bounded alloc/free/OOM ring, equivalent of CUDACachingAllocator's
 * recordHistory buffer. Callers must serialize (allocator mutex).
 */
class gpu_memory_history
{
public:
    void set_enabled(bool enabled, size_t max_entries)
    {
        enabled_ = enabled;
        if (max_entries != 0)
        {
            max_entries_ = max_entries;
        }
        if (!enabled_)
        {
            entries_.clear();
            return;
        }
        while (entries_.size() > max_entries_)
        {
            entries_.pop_front();
        }
    }

    bool enabled() const { return enabled_; }

    void record(
        gpu_memory_trace_action action,
        void*                   address,
        size_t                  size,
        size_t                  total_allocated,
        size_t                  total_reserved,
        int64_t                 stream)
    {
        if (!enabled_)
        {
            return;
        }
        if (entries_.size() >= max_entries_)
        {
            entries_.pop_front();
        }
        gpu_memory_trace_entry entry;
        entry.action          = action;
        entry.address         = address;
        entry.size            = size;
        entry.total_allocated = total_allocated;
        entry.total_reserved  = total_reserved;
        entry.stream          = stream;
        entries_.push_back(entry);
    }

    std::vector<gpu_memory_trace_entry> copy() const
    {
        return {entries_.begin(), entries_.end()};
    }

private:
    bool                               enabled_{false};
    size_t                             max_entries_{kDefaultMemoryHistoryEntries};
    std::deque<gpu_memory_trace_entry> entries_;
};

template <typename Stream>
inline int64_t stream_as_int(Stream stream)
{
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(stream));
}

inline void add_snapshot_block(
    std::map<uintptr_t, gpu_memory_segment_info>& segments,
    void*                                         segment_base,
    size_t                                        segment_size,
    bool                                          is_small,
    bool                                          is_expandable,
    int64_t                                       stream,
    void*                                         block_ptr,
    size_t                                        block_size,
    size_t                                        requested_size,
    bool                                          allocated,
    bool                                          active)
{
    gpu_memory_segment_info& seg = segments[reinterpret_cast<uintptr_t>(segment_base)];
    if (seg.address == nullptr)
    {
        seg.address       = segment_base;
        seg.total_size    = segment_size;
        seg.is_small      = is_small;
        seg.is_expandable = is_expandable;
        seg.stream        = stream;
    }
    gpu_memory_block_info block;
    block.address        = block_ptr;
    block.size           = block_size;
    block.requested_size = requested_size;
    block.allocated      = allocated;
    block.active         = active;
    seg.blocks.push_back(block);
    if (allocated)
    {
        seg.allocated_size += block_size;
        seg.requested_size += requested_size;
    }
    if (active)
    {
        seg.active_size += block_size;
    }
}

inline gpu_memory_snapshot finish_snapshot(
    std::map<uintptr_t, gpu_memory_segment_info>&& segments,
    std::vector<gpu_memory_trace_entry>            trace)
{
    gpu_memory_snapshot snap;
    snap.device_trace = std::move(trace);
    snap.segments.reserve(segments.size());
    for (auto& kv : segments)
    {
        gpu_memory_segment_info& seg = kv.second;
        std::sort(
            seg.blocks.begin(),
            seg.blocks.end(),
            [](const gpu_memory_block_info& a, const gpu_memory_block_info& b)
            {
                return reinterpret_cast<uintptr_t>(a.address) <
                       reinterpret_cast<uintptr_t>(b.address);
            });
        if (seg.total_size == 0)
        {
            for (const gpu_memory_block_info& block : seg.blocks)
            {
                seg.total_size += block.size;
            }
        }
        snap.segments.push_back(std::move(seg));
    }
    return snap;
}

}  // namespace memory::gpu
