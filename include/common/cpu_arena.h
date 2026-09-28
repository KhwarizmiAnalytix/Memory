/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

#include "common/memory_macros.h"
#include "helper/memory_allocator.h"

namespace memory
{

// Scoped bump arena for short-lived CPU temporaries sharing a lifetime.
//
// All allocations are served from a single contiguous block acquired at
// construction time (or lazily on first use when capacity is 0).  Individual
// free() calls are no-ops; the entire arena is reclaimed by reset() or
// destruction.  This makes it suitable for expression-temporary workloads
// where all temporaries are discarded together after an operator completes.
//
// Alignment contract: each allocation is aligned to at least @p default_align
// bytes (constructor parameter; default MEMORY_ALIGNMENT).  Larger per-call
// alignments up to 4096 bytes are supported via the aligned_alloc() overload.
//
// Thread-safety: NOT thread-safe.  Use one arena per thread or protect
// externally.  The arena is intentionally kept small and fast; adding a mutex
// would defeat its purpose.
//
// Usage example:
//   cpu_arena arena(4 * 1024 * 1024);   // 4 MiB backing
//   float* tmp = arena.alloc<float>(1024);
//   // ... use tmp ...
//   arena.reset();                       // reclaim all, keep backing
class MEMORY_VISIBILITY cpu_arena
{
public:
    static constexpr size_t kMaxAlignment = 4096;

    // Construct with pre-allocated backing of @p capacity bytes.
    // capacity == 0 defers the driver allocation until first use.
    explicit cpu_arena(
        size_t capacity      = 0,
        size_t default_align = MEMORY_ALIGNMENT)
        : capacity_(capacity), default_align_(default_align)
    {
        if (capacity_ != 0)
        {
            acquire(capacity_);
        }
    }

    ~cpu_arena()
    {
        release_backing();
    }

    cpu_arena(cpu_arena const&)            = delete;
    cpu_arena& operator=(cpu_arena const&) = delete;

    cpu_arena(cpu_arena&& rhs) noexcept
        : backing_(rhs.backing_), capacity_(rhs.capacity_), cursor_(rhs.cursor_),
          default_align_(rhs.default_align_)
    {
        rhs.backing_  = nullptr;
        rhs.capacity_ = 0;
        rhs.cursor_   = 0;
    }

    cpu_arena& operator=(cpu_arena&& rhs) noexcept
    {
        if (this == &rhs) return *this;
        release_backing();
        backing_      = rhs.backing_;
        capacity_     = rhs.capacity_;
        cursor_       = rhs.cursor_;
        default_align_ = rhs.default_align_;
        rhs.backing_  = nullptr;
        rhs.capacity_ = 0;
        rhs.cursor_   = 0;
        return *this;
    }

    // Allocate @p count elements of type T, aligned to max(alignof(T),
    // default_align_).  Throws std::bad_alloc if the arena is exhausted.
    template <typename T>
    T* alloc(size_t count)
    {
        if (sizeof(T) != 0 && count > std::numeric_limits<size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }
        size_t const align = (alignof(T) > default_align_) ? alignof(T) : default_align_;
        return static_cast<T*>(alloc_bytes(count * sizeof(T), align));
    }

    // Allocate @p bytes with explicit @p alignment.
    void* alloc_bytes(size_t bytes, size_t alignment = MEMORY_ALIGNMENT)
    {
        if (bytes == 0) return nullptr;
        alignment = (alignment < alignof(std::max_align_t)) ? alignof(std::max_align_t) : alignment;

        if (backing_ == nullptr)
        {
            // Lazy acquire: grow to at least the requested size.
            size_t initial = (bytes > capacity_) ? bytes * 2 : capacity_;
            if (initial == 0) initial = 64 * 1024;  // 64 KiB default
            acquire(initial);
        }

        // Overflow-safe alignment rounding.
        if (cursor_ > std::numeric_limits<size_t>::max() - (alignment - 1))
        {
            throw std::bad_alloc();
        }
        size_t const aligned_cursor = (cursor_ + alignment - 1) & ~(alignment - 1);
        // Overflow-safe bounds check: capacity_ - aligned_cursor avoids wrap.
        if (aligned_cursor > capacity_ || capacity_ - aligned_cursor < bytes)
        {
            throw std::bad_alloc();
        }
        cursor_ = aligned_cursor + bytes;
        return static_cast<char*>(backing_) + aligned_cursor;
    }

    // Return all arena memory to the bump pointer start; backing stays alive.
    void reset() noexcept { cursor_ = 0; }

    size_t used()      const noexcept { return cursor_; }
    size_t capacity()  const noexcept { return capacity_; }
    size_t available() const noexcept { return capacity_ - cursor_; }
    bool   empty()     const noexcept { return cursor_ == 0; }

private:
    void acquire(size_t cap)
    {
        backing_  = cpu::memory_allocator::allocate(cap, default_align_);
        capacity_ = cap;
        cursor_   = 0;
    }

    void release_backing() noexcept
    {
        if (backing_)
        {
            cpu::memory_allocator::free(backing_);
            backing_  = nullptr;
            capacity_ = 0;
            cursor_   = 0;
        }
    }

    void*  backing_{nullptr};
    size_t capacity_{0};
    size_t cursor_{0};
    size_t default_align_{MEMORY_ALIGNMENT};
};

}  // namespace memory
