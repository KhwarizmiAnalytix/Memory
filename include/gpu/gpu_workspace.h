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

#include <include/util/exception.h>

#include "common/device.h"
#include "common/memory_macros.h"
#include "common/execution_context.h"
#include "common/transfer.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
#include "gpu/gpu_dispatch.h"
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
// Lifetime contract (plan 6.2):
//   - Live slices: MULTIPLE slices may be live at once. acquire() bumps a cursor
//     and returns a distinct, 256-byte aligned, non-overlapping slice each time;
//     every slice stays valid until release(), reset() or destruction, which end
//     all of them together. There is no per-slice release.
//   - Order: slices are used on the workspace's own stream, so same-stream reuse
//     after release() is ordered by the stream and needs no synchronization.
//   - release() only rewinds the cursor. It never synchronizes. It must not be
//     treated as a completion point for work on ANOTHER stream.
//   - rebind() to a different stream (same device) keeps the slab, so it FAILS
//     CLOSED: it throws std::runtime_error unless the previous stream is idle at
//     that moment (a non-blocking query), leaving the workspace unchanged. The
//     caller waits for the old stream (or a token) and retries. rebind() to a
//     different device frees the slab on the original device and needs no proof.
//     Both require release() first (logging::exception otherwise, in every build).
//   - reset() frees the slab, which returns to the caching allocator keyed on the
//     workspace's stream, so no other stream can be handed it. It requires
//     release() first (logging::exception otherwise): freeing under live slices
//     would let their owners keep using memory the cache may hand out again. The
//     destructor frees unconditionally (it cannot throw).
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
            throw std::overflow_error("gpu_workspace::acquire<T>: count * sizeof(T) overflows size_t");
        }
        return static_cast<T*>(acquire(count * sizeof(T)));
    }

    // Release all acquired slices; the backing allocation is retained.
    void release() noexcept { cursor_ = 0; }

    // Free the backing allocation entirely.  A subsequent acquire() will
    // re-allocate. Requires release() first: fails closed (logging::exception)
    // while slices are live.
    void reset()
    {
        LOGGING_CHECK(
            cursor_ == 0,
            "gpu_workspace::reset called while slices are still acquired (cursor > 0); "
            "call release() before reset()");
        release_backing();
    }

    // Re-bind to a different execution context (e.g. a new stream each call).
    // Requires release() first (LOGGING_CHECK that cursor_ == 0). A different
    // device frees the slab on the original device. A different stream on the same
    // device keeps the slab, so the previous stream must be idle now: the check is a
    // non-blocking query, and failing it (or being unable to ask) throws
    // std::runtime_error with the workspace unchanged. Same stream: no check.
    void rebind(execution_context ctx)
    {
        LOGGING_CHECK(
            cursor_ == 0,
            "gpu_workspace::rebind called while slices are still acquired (cursor > 0); "
            "call release() before rebind()");
        if (backing_ != nullptr)
        {
            if (ctx.device_index() != ctx_.device_index())
            {
                release_backing();  // free on original device before switching
            }
            else if (ctx.stream != ctx_.stream)
            {
                require_previous_stream_idle();
            }
        }
        ctx_ = ctx;
    }

    size_t            capacity() const noexcept { return capacity_; }
    size_t            used()     const noexcept { return cursor_; }
    bool              empty()    const noexcept { return cursor_ == 0; }
    execution_context ctx()      const noexcept { return ctx_; }

private:
    // Fail closed: the slab is about to be used from another stream, which is safe
    // only if work already submitted on the old one has finished.
    void require_previous_stream_idle() const
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (ctx_.is_gpu() &&
            detail::token_stream_query(ctx_.device_index(), ctx_.stream).status ==
                detail::driver_status::ok)
        {
            return;
        }
#endif
        if (ctx_.is_gpu())
        {
            throw std::runtime_error(
                "gpu_workspace::rebind: the previous stream is not known to be idle; "
                "wait for it (or its completion token) and retry");
        }
    }

    void ensure_backing(size_t min_bytes)
    {
        size_t const needed = (min_bytes > capacity_) ? min_bytes : capacity_;
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        if (ctx_.is_gpu())
        {
            backing_ = allocate_device_bytes(needed, ctx_.device_index(), ctx_.stream);
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
                free_device_bytes(backing_, ctx_.device_index(), ctx_.stream);
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
