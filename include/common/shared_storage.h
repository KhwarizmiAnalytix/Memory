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

    void add_ref() noexcept { refs_.fetch_add(1, std::memory_order_relaxed); }

    /// Drops one reference; the last one destroys the object and its handle.
    static void release(shared_storage* s) noexcept
    {
        if (s != nullptr && s->refs_.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
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
};

}  // namespace memory
