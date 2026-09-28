/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "common/memory_macros.h"
#include "gpu/caching_allocator.h"

namespace memory::gpu
{

// Thread-local cache of per-device caching_allocator* pointers.
//
// Problem: every call to allocate<T> on a GPU device calls
// caching_allocator_for_device(index), which acquires a global registry mutex
// even when the allocator for that device was already created.  On
// allocation-heavy workloads this becomes a contention point.
//
// Solution: a small thread_local array of (device_index, pointer) pairs.
// After the first call the pointer is cached per-thread; subsequent lookups
// for the same device are mutex-free.
//
// Capacity: kMaxCachedDevices covers the typical 1–8 GPU scenario.  If more
// distinct devices are accessed within one thread the oldest entry is evicted
// (LRU via a linear scan; the small size makes this cheaper than a hash map).
//
// Usage:
//   caching_allocator& alloc = device_handle_cache::get(device_index);
//   alloc.allocate(nbytes, stream);

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL

class device_handle_cache
{
public:
    static constexpr int kMaxCachedDevices = 8;
    static constexpr int kInvalidDevice    = -1;

    struct entry
    {
        int                device_index{kInvalidDevice};
        caching_allocator* ptr{nullptr};
    };

    struct tl_state
    {
        std::array<entry, kMaxCachedDevices> cache{};
        int                                  next_slot{0};
    };

    // Per-thread cache state.  inline thread_local (C++17) ensures a single
    // instance is shared between get() and invalidate().
    static inline thread_local tl_state tl{};

    // Lookup the caching_allocator for @p device_index.
    // Falls back to the global registry on a miss, then caches the result.
    MEMORY_FORCE_INLINE static caching_allocator& get(int device_index)
    {
        for (auto& e : tl.cache)
        {
            if (e.device_index == device_index && e.ptr != nullptr)
            {
                return *e.ptr;
            }
        }
        // Miss: go to the global registry (acquires its mutex once).
        caching_allocator& alloc = caching_allocator_for_device(device_index);
        // Evict oldest entry via round-robin.
        tl.cache[tl.next_slot] = {device_index, &alloc};
        tl.next_slot           = (tl.next_slot + 1) % kMaxCachedDevices;
        return alloc;
    }

    // Invalidate all cached entries for @p device_index.  Call after
    // destroying the allocator (e.g., between unit tests).
    MEMORY_FORCE_INLINE static void invalidate(int device_index)
    {
        for (auto& e : tl.cache)
        {
            if (e.device_index == device_index)
            {
                e = {};
            }
        }
    }

    // Clear the entire per-thread cache.
    MEMORY_FORCE_INLINE static void clear() { tl = {}; }
};

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL

}  // namespace memory::gpu
