/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

#include "common/memory_macros.h"
#include "helper/memory_allocator.h"

namespace memory
{

// Scoped bump arena for short-lived CPU temporaries sharing a lifetime.
//
// Allocations are served by bumping a cursor inside a chunk. Individual frees do
// not exist; everything is reclaimed together by reset() or destruction, which
// suits expression-temporary workloads (Eigen/PyTorch style) where all
// temporaries die after one operator completes.
//
// Contract (plan 6.1):
// - Alignment is a property of the returned ADDRESS, not of an offset: every
//   allocation is aligned to max(requested, alignof(max_align_t)) in memory,
//   whatever the backing address is. The alignment must be a power of two no
//   larger than kMaxAlignment (std::invalid_argument otherwise).
// - Capacity: `capacity` bytes are acquired up front (0 = lazily, 64 KiB or the
//   first request, whichever is larger). With `max_capacity == 0` the arena is
//   fixed: exhaustion throws std::bad_alloc. With `max_capacity > capacity` it
//   grows by adding chunks (never by moving memory, so earlier pointers stay
//   valid) until the chunks total `max_capacity`, then throws. Sizes are checked
//   for overflow; a failed allocation leaves the arena unchanged.
// - reset() invalidates every pointer handed out so far: it rewinds to the first
//   chunk (extra chunks are returned to the allocator), bumps generation(), and in
//   Debug builds poisons the first chunk with 0xDD so a stale read is visible.
//   Memory is reused from the same address after a reset.
// - Thread confinement: NOT thread-safe, by design (a mutex would defeat the
//   point). Use one arena per thread or serialize externally. Moving an arena
//   transfers its memory; the moved-from arena is empty and reusable.
//
// Usage:
//   cpu_arena arena(4 * 1024 * 1024);   // 4 MiB backing
//   float* tmp = arena.alloc<float>(1024);
//   arena.reset();                       // reclaim all, keep the first chunk
class MEMORY_VISIBILITY cpu_arena
{
public:
    static constexpr size_t kMaxAlignment    = 4096;
    static constexpr size_t kDefaultLazySize = 64U * 1024U;

    // @p capacity       bytes to acquire now (0 = lazily on first use)
    // @p default_align  alignment of the backing and of untyped allocations
    // @p max_capacity   total bytes across chunks (0 = fixed, no growth)
    explicit cpu_arena(
        size_t capacity      = 0,
        size_t default_align = MEMORY_ALIGNMENT,
        size_t max_capacity  = 0)
        : default_align_(default_align), max_capacity_(max_capacity)
    {
        if (!is_valid_alignment(default_align_) || default_align_ < sizeof(void*))
        {
            throw std::invalid_argument(
                "cpu_arena: default alignment must be a power of two in [sizeof(void*), 4096]");
        }
        if (max_capacity_ != 0 && max_capacity_ < capacity)
        {
            throw std::invalid_argument("cpu_arena: max_capacity is smaller than capacity");
        }
        data_offset_ = round_up(sizeof(chunk), default_align_);
        if (capacity != 0)
        {
            add_chunk(capacity);
        }
    }

    ~cpu_arena() { release_all(); }

    cpu_arena(cpu_arena const&)            = delete;
    cpu_arena& operator=(cpu_arena const&) = delete;

    cpu_arena(cpu_arena&& rhs) noexcept
        : head_(rhs.head_), current_(rhs.current_), cursor_(rhs.cursor_),
          closed_used_(rhs.closed_used_), total_capacity_(rhs.total_capacity_),
          default_align_(rhs.default_align_), max_capacity_(rhs.max_capacity_),
          data_offset_(rhs.data_offset_), generation_(rhs.generation_)
    {
        rhs.forget();
    }

    cpu_arena& operator=(cpu_arena&& rhs) noexcept
    {
        if (this == &rhs)
        {
            return *this;
        }
        release_all();
        head_           = rhs.head_;
        current_        = rhs.current_;
        cursor_         = rhs.cursor_;
        closed_used_    = rhs.closed_used_;
        total_capacity_ = rhs.total_capacity_;
        default_align_  = rhs.default_align_;
        max_capacity_   = rhs.max_capacity_;
        data_offset_    = rhs.data_offset_;
        generation_     = rhs.generation_;
        rhs.forget();
        return *this;
    }

    // Allocate @p count elements of type T, aligned to max(alignof(T), default
    // alignment). Throws std::bad_alloc if the arena is exhausted or the size
    // overflows.
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

    // Allocate @p bytes whose address is a multiple of @p alignment.
    // Throws std::invalid_argument for a bad alignment, std::bad_alloc when the
    // arena cannot serve the request (the arena is then unchanged).
    void* alloc_bytes(size_t bytes, size_t alignment = MEMORY_ALIGNMENT)
    {
        if (!is_valid_alignment(alignment))
        {
            throw std::invalid_argument("cpu_arena: alignment must be a power of two <= 4096");
        }
        if (bytes == 0)
        {
            return nullptr;
        }
        if (alignment < alignof(std::max_align_t))
        {
            alignment = alignof(std::max_align_t);
        }

        if (current_ == nullptr)
        {
            // Lazy first chunk: the usual size, or enough for this request.
            size_t const need = worst_case(bytes, alignment);
            add_chunk(need > kDefaultLazySize ? need : kDefaultLazySize);
        }
        if (void* p = bump(bytes, alignment))
        {
            return p;
        }

        // The current chunk is full: grow, if allowed.
        if (max_capacity_ == 0 || total_capacity_ >= max_capacity_)
        {
            throw std::bad_alloc();
        }
        size_t const need = worst_case(bytes, alignment);
        size_t const room = max_capacity_ - total_capacity_;
        if (need > room)
        {
            throw std::bad_alloc();
        }
        size_t grow = (current_->capacity > room / 2) ? room : current_->capacity * 2;
        if (grow < need)
        {
            grow = need;
        }
        add_chunk(grow);
        void* p = bump(bytes, alignment);  // fits by construction: grow >= need
        if (p == nullptr)
        {
            throw std::bad_alloc();
        }
        return p;
    }

    // Invalidate every allocation and rewind. The first chunk stays; the others
    // are freed. Pointers obtained before the reset must not be used again.
    void reset() noexcept
    {
        ++generation_;
        if (head_ == nullptr)
        {
            return;
        }
        chunk* extra = head_->next;
        head_->next  = nullptr;
        while (extra != nullptr)
        {
            chunk* next = extra->next;
            cpu::memory_allocator::free(extra);
            extra = next;
        }
        current_        = head_;
        cursor_         = 0;
        closed_used_    = 0;
        total_capacity_ = head_->capacity;
#ifndef NDEBUG
        std::memset(chunk_data(head_), 0xDD, head_->capacity);
#endif
    }

    // Bytes consumed, including alignment padding, across all chunks.
    size_t used() const noexcept { return closed_used_ + cursor_; }
    // Bytes acquired from the allocator across all chunks (headers excluded).
    size_t capacity() const noexcept { return total_capacity_; }
    // capacity() - used(): an upper bound; padding and a full current chunk can
    // make a request of this size fail in a growable arena's last chunk.
    size_t available() const noexcept { return total_capacity_ - used(); }
    bool   empty() const noexcept { return used() == 0; }
    // Incremented by every reset(): a pointer is valid only for the generation
    // it was obtained in.
    std::uint64_t generation() const noexcept { return generation_; }
    size_t        max_capacity() const noexcept { return max_capacity_; }

private:
    struct chunk
    {
        chunk* next;
        size_t capacity;  // usable bytes after the header
    };

    static constexpr bool is_valid_alignment(size_t a) noexcept
    {
        return a != 0 && (a & (a - 1)) == 0 && a <= kMaxAlignment;
    }
    static constexpr size_t round_up(size_t value, size_t align) noexcept
    {
        return (value + align - 1) & ~(align - 1);
    }

    // Bytes a chunk must offer to serve (bytes, alignment) wherever its base lies.
    static size_t worst_case(size_t bytes, size_t alignment)
    {
        if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1))
        {
            throw std::bad_alloc();
        }
        return bytes + alignment - 1;
    }

    char* chunk_data(chunk* c) const noexcept { return reinterpret_cast<char*>(c) + data_offset_; }

    // Serve from the current chunk, aligning the address. nullptr if it does not fit.
    void* bump(size_t bytes, size_t alignment) noexcept
    {
        char* const         base    = chunk_data(current_);
        std::uintptr_t const addr    = reinterpret_cast<std::uintptr_t>(base) + cursor_;
        std::uintptr_t const padding = (alignment - (addr & (alignment - 1))) & (alignment - 1);
        size_t const        offset  = cursor_ + static_cast<size_t>(padding);
        if (offset > current_->capacity || current_->capacity - offset < bytes)
        {
            return nullptr;
        }
        cursor_ = offset + bytes;
        return base + offset;
    }

    // Acquire a chunk of @p cap usable bytes and make it current. On failure the
    // arena is unchanged.
    void add_chunk(size_t cap)
    {
        if (cap > std::numeric_limits<size_t>::max() - data_offset_)
        {
            throw std::bad_alloc();
        }
        void* raw = cpu::memory_allocator::allocate(cap + data_offset_, default_align_);
        auto* c   = static_cast<chunk*>(raw);
        c->next     = nullptr;
        c->capacity = cap;
        if (current_ != nullptr)
        {
            closed_used_ += cursor_;
            current_->next = c;
        }
        else
        {
            head_ = c;
        }
        current_ = c;
        cursor_  = 0;
        total_capacity_ += cap;
    }

    void release_all() noexcept
    {
        chunk* c = head_;
        while (c != nullptr)
        {
            chunk* next = c->next;
            cpu::memory_allocator::free(c);
            c = next;
        }
        forget();
    }

    void forget() noexcept
    {
        head_           = nullptr;
        current_        = nullptr;
        cursor_         = 0;
        closed_used_    = 0;
        total_capacity_ = 0;
    }

    chunk*        head_{nullptr};
    chunk*        current_{nullptr};
    size_t        cursor_{0};          // bytes used in current_
    size_t        closed_used_{0};     // bytes used in the chunks before current_
    size_t        total_capacity_{0};
    size_t        default_align_{MEMORY_ALIGNMENT};
    size_t        max_capacity_{0};
    size_t        data_offset_{0};
    std::uint64_t generation_{0};
};

}  // namespace memory
