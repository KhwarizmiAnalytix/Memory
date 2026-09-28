/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/device.h"
#include "common/memory_macros.h"
#include "common/execution_context.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
#include "gpu/caching_allocator.h"
#endif

namespace memory::gpu
{

// Reusable GPU scratch slab for operator workspaces.
//
// A gpu_workspace holds a single GPU allocation of fixed capacity.  The
// allocation is NOT freed between operator calls: callers acquire() a slice,
// use it, and release() it.  When the full capacity is available the same
// driver allocation is reused, avoiding repeated round trips through the
// caching allocator and driver.
//
// Lifetime contract:
//   - One gpu_workspace per operator (or one per stream, shared across calls).
//   - acquire() returns a raw pointer into the slab; the caller must not
//     hold this pointer across a subsequent acquire() or reset() call.
//   - release() returns the slab to "fully available" state; any previous
//     acquire() pointers are invalidated.
//   - If capacity is exceeded, acquire() throws std::bad_alloc; the caller
//     should fall back to a fresh allocator<T>::allocate() call.
//
// Thread-safety: NOT thread-safe.  Use one workspace per stream / thread.
class MEMORY_VISIBILITY gpu_workspace
{
public:
    explicit gpu_workspace(
        size_t                capacity  = 0,
        execution_context     ctx       = execution_context::cpu()) noexcept
        : capacity_(capacity), ctx_(ctx)
    {}

    ~gpu_workspace() { release_backing(); }

    gpu_workspace(gpu_workspace const&)            = delete;
    gpu_workspace& operator=(gpu_workspace const&) = delete;

    gpu_workspace(gpu_workspace&& rhs) noexcept
        : backing_(rhs.backing_), capacity_(rhs.capacity_),
          cursor_(rhs.cursor_), ctx_(rhs.ctx_)
    {
        rhs.backing_  = nullptr;
        rhs.capacity_ = 0;
        rhs.cursor_   = 0;
    }

    gpu_workspace& operator=(gpu_workspace&& rhs) noexcept
    {
        if (this == &rhs) return *this;
        release_backing();
        backing_      = rhs.backing_;
        capacity_     = rhs.capacity_;
        cursor_       = rhs.cursor_;
        ctx_          = rhs.ctx_;
        rhs.backing_  = nullptr;
        rhs.capacity_ = 0;
        rhs.cursor_   = 0;
        return *this;
    }

    // Acquire @p bytes from the slab (256-byte aligned for GPU use).
    // Throws std::bad_alloc if the slab is exhausted.  Lazily allocates the
    // backing slab on first call.
    void* acquire(size_t bytes)
    {
        if (bytes == 0) return nullptr;
        constexpr size_t kAlign = 256;
        if (backing_ == nullptr)
        {
            ensure_backing(bytes);
        }
        // Overflow-safe alignment rounding.
        if (cursor_ > std::numeric_limits<size_t>::max() - (kAlign - 1))
        {
            throw std::bad_alloc();
        }
        size_t const aligned_cursor = (cursor_ + kAlign - 1) & ~(kAlign - 1);
        // Overflow-safe bounds check: capacity_ - aligned_cursor avoids wrap.
        if (aligned_cursor > capacity_ || capacity_ - aligned_cursor < bytes)
        {
            throw std::bad_alloc();
        }
        cursor_ = aligned_cursor + bytes;
        return static_cast<char*>(backing_) + aligned_cursor;
    }

    // Typed convenience: acquire count * sizeof(T) bytes.
    template <typename T>
    T* acquire(size_t count)
    {
        if (sizeof(T) != 0 && count > std::numeric_limits<size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(acquire(count * sizeof(T)));
    }

    // Release all acquired slices; the backing allocation is retained.
    void release() noexcept { cursor_ = 0; }

    // Free the backing allocation entirely.  A subsequent acquire() will
    // re-allocate.
    void reset() { release_backing(); }

    // Re-bind to a different execution context (e.g. a new stream each call).
    // Calling this while slices are acquired leads to record_stream mismatches;
    // always call release() before rebind().
    //
    // If the new context targets a different device than the current backing
    // allocation, the backing is freed here (on the original device) before the
    // context is updated.  A same-device rebind retains the cached backing slab.
    void rebind(execution_context ctx) noexcept
    {
        if (backing_ != nullptr && ctx.device_index != ctx_.device_index)
        {
            release_backing();  // free on original device before switching
        }
        ctx_ = ctx;
    }

    size_t            capacity() const noexcept { return capacity_; }
    size_t            used()     const noexcept { return cursor_; }
    bool              empty()    const noexcept { return cursor_ == 0; }
    execution_context ctx()      const noexcept { return ctx_; }

private:
    void ensure_backing(size_t min_bytes)
    {
        size_t const needed = (min_bytes > capacity_) ? min_bytes : capacity_;
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        if (ctx_.is_gpu())
        {
            backing_ =
                caching_allocator_for_device(ctx_.device_index).allocate(needed, ctx_.stream);
            capacity_ = needed;
            return;
        }
#endif
        (void)needed;
        throw std::invalid_argument(
            "gpu_workspace: no GPU backend compiled in — cannot allocate GPU workspace");
    }

    void release_backing() noexcept
    {
        if (!backing_) return;
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        if (ctx_.is_gpu())
        {
            try
            {
                caching_allocator_for_device(ctx_.device_index)
                    .deallocate(backing_, capacity_, ctx_.stream);
            }
            catch (...) {}
        }
#endif
        backing_  = nullptr;
        capacity_ = 0;
        cursor_   = 0;
    }

    void*             backing_{nullptr};
    size_t            capacity_{0};
    size_t            cursor_{0};
    execution_context ctx_{};
};

}  // namespace memory::gpu
