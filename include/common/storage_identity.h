/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace memory
{

// A stable allocation identity.  Assigned once at allocate time; survives all
// derived views and address reuse so telemetry can pair alloc/free events even
// when the block address is recycled between trace entries.
struct storage_identity
{
    uint64_t alloc_id{0};   // 0 = invalid/unset; assigned by next_id()
    void*    base{nullptr}; // allocation base pointer (not a view's start)
    size_t   capacity{0};   // allocation byte capacity

    bool valid() const noexcept { return alloc_id != 0; }

    // Thread-safe monotonic ID source (starts at 1; 0 is always invalid).
    static uint64_t next_id() noexcept
    {
        static std::atomic<uint64_t> counter{1};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }

    bool operator==(storage_identity const& o) const noexcept
    {
        return alloc_id == o.alloc_id && base == o.base;
    }
    bool operator!=(storage_identity const& o) const noexcept { return !(*this == o); }
};

}  // namespace memory
