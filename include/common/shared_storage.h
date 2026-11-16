/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#include "common/deleter_fn.h"
#include "common/execution_context.h"
#include "common/memory_macros.h"
#include "common/storage_handle.h"
#include "common/storage_identity.h"

namespace memory
{

/// Shared owner of one `storage_handle` (c10::StorageImpl analogue, plan 4.2).
///
/// An intrusive atomic reference count plus the handle: the last release destroys
/// the handle, whose deleter runs exactly once (CPU free, GPU cache free, or the
/// caller's adopted deleter). There is no other free path. Instances live only on
/// the heap, created by the two-phase `allocate_block()` / `construct()` pair so a
/// caller can reserve the memory *before* it commits a handle to the object; if the
/// reservation throws, the caller still owns whatever it was about to transfer.
class MEMORY_VISIBILITY shared_storage
{
public:
    shared_storage(shared_storage const&)            = delete;
    shared_storage& operator=(shared_storage const&) = delete;

    /// Reserve uninitialized memory for one shared_storage. May throw bad_alloc.
    static void* allocate_block() { return ::operator new(sizeof(shared_storage)); }

    /// Construct in a block from allocate_block(), taking ownership of @p handle.
    /// Starts with one reference. Does not throw.
    static shared_storage* construct(
        void* block, storage_handle&& handle, execution_context ctx) noexcept
    {
        return new (block) shared_storage(std::move(handle), ctx);
    }

    /// Convenience: reserve and construct. On bad_alloc @p handle is untouched
    /// (still owned by the caller).
    static shared_storage* create(storage_handle&& handle, execution_context ctx)
    {
        void* block = allocate_block();
        return construct(block, std::move(handle), ctx);
    }

    // One allocation for control block and payload (std::make_shared /
    // c10::make_intrusive analogue): a CPU block laid out as
    // [shared_storage | padding to alignment | nbytes payload]. The payload is
    // uninitialized, `allocated`, with a fresh allocation id; base() returns its
    // first byte. The embedded handle has no deleter: release() frees the whole
    // block once. nbytes must be non-zero and the context a CPU one; throws
    // bad_alloc, or overflow_error when the block size overflows size_t.
    MEMORY_API static shared_storage* allocate_inline_cpu(
        std::size_t nbytes, std::size_t alignment, execution_context ctx);

    // Bytes from the block start to the payload for a given payload alignment.
    static constexpr std::size_t inline_header_bytes(std::size_t alignment) noexcept
    {
        std::size_t const align =
            alignment < alignof(shared_storage) ? alignof(shared_storage) : alignment;
        return (sizeof(shared_storage) + align - 1) / align * align;
    }

    // True when the payload lives in this control block's own allocation.
    bool is_inline() const noexcept { return inline_header_ != 0; }

    void add_ref() noexcept { refs_.fetch_add(1, std::memory_order_relaxed); }

    /// Drops one reference; the last one destroys the object and its handle.
    static void release(shared_storage* s) noexcept
    {
        if (s != nullptr && s->refs_.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            if (s->is_inline())
            {
                std::size_t const block_bytes = s->inline_header_ + s->handle_.nbytes();
                deleter_fn const  free_block  = s->block_free_;
                s->~shared_storage();  // handle has no deleter: nothing freed yet
                free_block(nullptr, static_cast<void*>(s), block_bytes);
                return;
            }
            s->~shared_storage();
            ::operator delete(static_cast<void*>(s));
        }
    }

    int32_t use_count() const noexcept { return refs_.load(std::memory_order_relaxed); }

    storage_handle const&    handle() const noexcept { return handle_; }
    void*                    base() const noexcept { return handle_.get(); }
    std::size_t              nbytes() const noexcept { return handle_.nbytes(); }
    execution_context const& ctx() const noexcept { return ctx_; }
    storage_identity const&  identity() const noexcept { return identity_; }
    storage_provenance       provenance() const noexcept { return handle_.provenance(); }

private:
    shared_storage(storage_handle&& handle, execution_context ctx) noexcept
        : handle_(std::move(handle)), ctx_(ctx)
    {
        identity_.alloc_id = handle_.id().value;
        identity_.base     = handle_.get();
        identity_.capacity = handle_.nbytes();
    }

    ~shared_storage() = default;

    std::atomic<int32_t> refs_{1};
    storage_handle       handle_;
    execution_context    ctx_;
    storage_identity     identity_{};
    // Inline blocks only: payload offset and the block's free function, set by
    // allocate_inline_cpu() so release() needs no out-of-line symbol.
    std::size_t inline_header_{0};  // 0: separate control block
    deleter_fn  block_free_{nullptr};
};

}  // namespace memory
