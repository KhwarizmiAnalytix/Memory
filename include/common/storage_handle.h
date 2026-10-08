/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstddef>
#include <stdexcept>

#include "common/cleanup_diagnostic.h"
#include "common/deleter_fn.h"
#include "common/device.h"
#include "common/execution_context.h"
#include "common/memory_export.h"
#include "common/memory_macros.h"
#include "common/storage_identity.h"

namespace memory
{

// 48-byte owning raw-byte handle (c10::DataPtr analogue, plan §4.2).
// Move-only; no heap allocation in the handle itself.
//
// GPU allocations (plan §2.10, R1):
//   deleter_ = gpu_free_fn (looks up the allocation stream from the cache_block).
//   ctx_     = pointer to the gpu::caching_allocator, obtained once at allocate time.
//              gpu_free_fn uses ctx_ to reach the cache without a registry lookup
//              (0 registry lookups on free). Dropping a bare GPU handle frees it.
//
// CPU allocations:
//   deleter_ = cpu_free_fn (calls cpu::memory_allocator::free).
//   ctx_     = nullptr.
//
// Adopted memory:
//   deleter_ = caller-supplied function pointer.
//   ctx_     = caller-supplied context (state a callable needs lives there).
class MEMORY_VISIBILITY storage_handle
{
public:
    storage_handle() noexcept = default;

    storage_handle(
        void*         ptr,
        std::size_t   nbytes,
        deleter_fn    del,
        void*         ctx,
        device        dev,
        allocation_id id) noexcept
        : ptr_(ptr), nbytes_(nbytes), deleter_(del), ctx_(ctx), dev_(dev), id_(id)
    {
    }

    ~storage_handle() noexcept
    {
        if (ptr_ != nullptr && deleter_ != nullptr)
        {
            deleter_(ctx_, ptr_, nbytes_);
        }
    }

    storage_handle(storage_handle&& o) noexcept
        : ptr_(o.ptr_), nbytes_(o.nbytes_), deleter_(o.deleter_), ctx_(o.ctx_), dev_(o.dev_),
          id_(o.id_)
    {
        o.zero_out();
    }

    storage_handle& operator=(storage_handle&& o) noexcept
    {
        if (this != &o)
        {
            if (ptr_ != nullptr && deleter_ != nullptr)
            {
                deleter_(ctx_, ptr_, nbytes_);
            }
            ptr_     = o.ptr_;
            nbytes_  = o.nbytes_;
            deleter_ = o.deleter_;
            ctx_     = o.ctx_;
            dev_     = o.dev_;
            id_      = o.id_;
            o.zero_out();
        }
        return *this;
    }

    storage_handle(storage_handle const&)            = delete;
    storage_handle& operator=(storage_handle const&) = delete;

    void*         get() const noexcept { return ptr_; }
    std::size_t   nbytes() const noexcept { return nbytes_; }
    device        dev() const noexcept { return dev_; }
    allocation_id id() const noexcept { return id_; }
    bool          empty() const noexcept { return ptr_ == nullptr; }

    // Disarm: returns the raw pointer and zeros the handle so the destructor
    // becomes a no-op.  The caller is responsible for freeing the memory.
    void* release() noexcept
    {
        void* p = ptr_;
        zero_out();
        return p;
    }

private:
    MEMORY_FORCE_INLINE void zero_out() noexcept
    {
        ptr_     = nullptr;
        nbytes_  = 0;
        deleter_ = nullptr;
        ctx_     = nullptr;
        dev_     = device{};
        id_      = allocation_id{};
    }

    // 48-byte layout (64-bit platforms):
    //   [ 0.. 7] ptr_      (8 B, pointer)
    //   [ 8..15] nbytes_   (8 B, size_t)
    //   [16..23] deleter_  (8 B, fn ptr)
    //   [24..31] ctx_      (8 B, void*)
    //   [32..35] dev_      (4 B: device_enum 1B + pad 1B + int16_t 2B)
    //   [36..39] implicit padding (align allocation_id to 8)
    //   [40..47] id_       (8 B, uint64_t)
    void*         ptr_{nullptr};
    std::size_t   nbytes_{0};
    deleter_fn    deleter_{nullptr};
    void*         ctx_{nullptr};
    device        dev_{};
    allocation_id id_;
};

static_assert(sizeof(storage_handle) == 48, "storage_handle must be exactly 48 bytes (plan §4.2)");

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

// Allocate nbytes bytes on the device described by ctx, with the given
// alignment.  Returns an empty handle when nbytes == 0.
// Throws: std::bad_alloc (OOM), std::invalid_argument (bad alignment / device).
// GPU: calls caching_allocator_for_device() exactly once; wires gpu_free_fn as
// the deleter (looks up the allocation stream from the cache block and frees —
// plan §2.10, R1). 0 registry lookups at free; the handle is self-freeing.
MEMORY_API storage_handle
allocate_bytes(std::size_t nbytes, std::size_t alignment, execution_context ctx);

// Wrap foreign memory in a storage_handle with a mandatory explicit deleter.
// Throws std::invalid_argument if ptr, nbytes, or del is null/zero.
inline storage_handle adopt_bytes(
    void* ptr, std::size_t nbytes, device dev, deleter_fn del, void* del_ctx)
{
    if (!ptr || !nbytes || !del)
    {
        throw std::invalid_argument(
            "adopt_bytes: ptr, nbytes, and deleter must all be non-null/non-zero");
    }
    return storage_handle(ptr, nbytes, del, del_ctx, dev, next_allocation_id());
}

}  // namespace memory
