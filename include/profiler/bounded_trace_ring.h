/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/memory_export.h"

namespace memory::gpu
{

// Fixed-capacity ring buffer for allocation trace entries.  All storage is
// preallocated at construction time so record() never heap-allocates.
//
// Template parameter Entry must be default-constructible and copy-assignable.
// The ring is NOT thread-safe: callers must hold the allocator lock.
//
// Design contract (from cpu_gpu_memory_review.md Order 4):
//   - No dynamic allocation during record() — storage reserved upfront.
//   - Dropped entries increment a loss counter visible at snapshot time.
//   - copy() produces a std::vector snapshot without clearing the ring.
//   - clear() resets both read and write positions and the loss count.
template <typename Entry>
class bounded_trace_ring
{
public:
    explicit bounded_trace_ring(size_t capacity = 0)
    {
        if (capacity != 0)
        {
            storage_.resize(capacity);
            capacity_ = capacity;
        }
    }

    void resize(size_t new_capacity)
    {
        if (new_capacity == capacity_)
        {
            return;
        }
        // Materialise current entries into a temporary, then re-fill.
        auto         current    = copy();
        size_t const saved_lost = entries_lost_;
        storage_.resize(new_capacity);
        capacity_     = new_capacity;
        head_         = 0;
        count_        = 0;
        entries_lost_ = 0;
        // Entries that don't fit in the new capacity are dropped; count them.
        size_t const start = (current.size() > new_capacity) ? (current.size() - new_capacity) : 0;
        entries_lost_ = saved_lost + start;
        for (size_t i = start; i < current.size(); ++i)
        {
            push(current[i]);
        }
    }

    // Record one entry.  If the ring is full the oldest entry is overwritten
    // and entries_lost_ increments.  No heap allocation occurs.
    void push(Entry const& e)
    {
        if (capacity_ == 0)
        {
            ++entries_lost_;
            return;
        }
        if (count_ == capacity_)
        {
            ++entries_lost_;
            storage_[head_] = e;
            head_            = (head_ + 1) % capacity_;
        }
        else
        {
            size_t const slot = (head_ + count_) % capacity_;
            storage_[slot]    = e;
            ++count_;
        }
    }

    // Snapshot all entries in insertion order (oldest first).
    std::vector<Entry> copy() const
    {
        std::vector<Entry> out;
        out.reserve(count_);
        for (size_t i = 0; i < count_; ++i)
        {
            out.push_back(storage_[(head_ + i) % capacity_]);
        }
        return out;
    }

    void clear() noexcept
    {
        head_         = 0;
        count_        = 0;
        entries_lost_ = 0;
    }

    size_t size()         const noexcept { return count_; }
    size_t capacity()     const noexcept { return capacity_; }
    size_t entries_lost() const noexcept { return entries_lost_; }
    bool   empty()        const noexcept { return count_ == 0; }

private:
    std::vector<Entry> storage_;
    size_t capacity_     {0};
    size_t head_         {0};  // index of the oldest entry
    size_t count_        {0};  // number of valid entries
    size_t entries_lost_ {0};  // entries overwritten due to ring full
};

}  // namespace memory::gpu
