/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/cleanup_diagnostic.h"
#include "common/deleter_fn.h"
#include "common/execution_context.h"
#include "common/memory_macros.h"
#include "common/shared_storage.h"
#include "common/storage_element.h"
#include "common/storage_handle.h"
#include "common/storage_identity.h"
#include "common/transfer.h"

namespace memory
{

// Forward declaration: definition is in data_ptr.h (which includes retained_ptr.h
// transitively).  The promotion constructor body is defined out-of-class in data_ptr.h
// after both types are complete.
template <typename T>
struct data_ptr;

// Shared-ownership typed buffer.  Multiple retained_ptr instances may reference
// the same underlying allocation through a reference-counted control block.
// Copy increments the count; move steals the handle; the last owner calls the
// deleter and destroys the control block.
//
// Unlike data_ptr (unique ownership, deep-copy on copy), retained_ptr:
//   - Allows multiple views/slices of the same allocation without copies.
//   - Survives async operations via copy_token which holds both endpoints.
//   - Carries a stable storage_identity through all slices.
//   - Adopts foreign memory via retain_ptr::adopt() with a caller-supplied
//     deleter; no platform-specific allocator is assumed.
//
// Slice semantics: slice(offset, count) returns a retained_ptr that references
// the same control block (and thus the same storage_identity::alloc_id) as the
// original, keeping the whole allocation alive as long as any slice exists.
template <typename T>
class retained_ptr
{
    MEMORY_STATIC_ASSERT_STORAGE_ELEMENT(T, ::memory::storage_max_alignment);

public:
    using element_type = T;

    // --- Construction / Adoption ---

    retained_ptr() noexcept = default;

    // Promote unique ownership to shared: transfers data_ptr<T> storage into a
    // new control block without reallocation.  Defined out-of-class in data_ptr.h
    // after both types are complete.
    explicit retained_ptr(data_ptr<T>&& dp);

    // Adopt foreign memory: the supplied deleter takes ownership of storage.
    // The returned retained_ptr owns a single reference; callers may then
    // copy/move it to distribute access.
    //
    // Ownership transfers only when this call returns a handle. On every failure
    // (it throws) the deleter has NOT run and the caller still owns @p data:
    //   - an empty deleter is rejected (adoption needs an explicit way to free);
    //   - a null base with a non-zero capacity, or a non-null base with zero
    //     capacity, is rejected (invalid_argument) rather than silently dropped;
    //   - capacity * sizeof(T) overflowing size_t is rejected (overflow_error);
    //   - bad_alloc from the control block leaves @p data with the caller.
    // A null base with zero capacity yields an empty (null) retained_ptr; the
    // deleter is discarded unused. The capacity is asserted by the caller, not
    // verified.
    static retained_ptr adopt(
        T*                data,
        size_t            capacity,
        execution_context ctx,
        deleter_fn        deleter,
        void*             deleter_ctx)
    {
        if (deleter == nullptr)
        {
            throw std::invalid_argument("retained_ptr::adopt: an explicit deleter is required");
        }
        if (!validate_adoption(data, capacity))
        {
            return {};
        }
        // Reserve the block before committing the handle: a bad_alloc here leaves
        // the caller owning the data (the deleter has not run).
        void* const block = shared_storage::allocate_block();
        return retained_ptr(
            shared_storage::construct(
                block,
                storage_handle(
                    data, capacity * sizeof(T), deleter, deleter_ctx, ctx.dev, next_allocation_id()),
                ctx),
            data,
            capacity);
    }

    // Convenience overload taking a callable. The callable lives in a heap thunk
    // owned by the storage until the last reference drops; prefer the function-
    // pointer overload above on hot paths (no allocation beyond the block). A
    // callable that throws is counted in cleanup_diagnostic, not propagated.
    static retained_ptr adopt(
        T*                data,
        size_t            capacity,
        execution_context ctx,
        std::function<void(T*, size_t, execution_context const&)> deleter)
    {
        if (!deleter)
        {
            throw std::invalid_argument("retained_ptr::adopt: an explicit deleter is required");
        }
        if (!validate_adoption(data, capacity))
        {
            return {};
        }
        auto thunk = std::make_unique<adopt_thunk>(adopt_thunk{std::move(deleter), ctx, capacity});
        void* const block = shared_storage::allocate_block();
        // Past this point nothing throws: the thunk is released into the handle.
        auto* const storage = shared_storage::construct(
            block,
            storage_handle(
                data,
                capacity * sizeof(T),
                &adopt_thunk_free,
                thunk.release(),
                ctx.dev,
                next_allocation_id()),
            ctx);
        return retained_ptr(storage, data, capacity);
    }

    // Wrap an allocation made by allocate_bytes() (or any storage_handle) in
    // shared ownership without reallocating. An empty handle yields an empty
    // retained_ptr. On bad_alloc the handle is untouched and still owned by the
    // caller.
    static retained_ptr from_storage(storage_handle&& handle, execution_context ctx)
    {
        if (handle.empty())
        {
            return {};
        }
        T* const     base     = static_cast<T*>(handle.get());
        size_t const capacity = handle.nbytes() / sizeof(T);
        void* const  block    = shared_storage::allocate_block();
        return retained_ptr(
            shared_storage::construct(block, std::move(handle), ctx), base, capacity);
    }

    // --- Copy / Move / Destructor ---

    retained_ptr(retained_ptr const& rhs) noexcept
        : cb_(rhs.cb_), data_(rhs.data_), size_(rhs.size_)
    {
        if (cb_)
        {
            cb_->add_ref();
        }
    }

    retained_ptr& operator=(retained_ptr const& rhs) noexcept
    {
        if (this == &rhs)
        {
            return *this;
        }
        release();
        cb_   = rhs.cb_;
        data_ = rhs.data_;
        size_ = rhs.size_;
        if (cb_)
        {
            cb_->add_ref();
        }
        return *this;
    }

    retained_ptr(retained_ptr&& rhs) noexcept
        : cb_(rhs.cb_), data_(rhs.data_), size_(rhs.size_)
    {
        rhs.cb_   = nullptr;
        rhs.data_ = nullptr;
        rhs.size_ = 0;
    }

    retained_ptr& operator=(retained_ptr&& rhs) noexcept
    {
        if (this == &rhs)
        {
            return *this;
        }
        release();
        cb_         = rhs.cb_;
        data_       = rhs.data_;
        size_       = rhs.size_;
        rhs.cb_     = nullptr;
        rhs.data_   = nullptr;
        rhs.size_   = 0;
        return *this;
    }

    ~retained_ptr() { release(); }

    // --- Slicing ---

    // Returns a retained_ptr viewing [offset, offset+count) of this buffer.
    // The result shares the same control block (alloc_id and base), keeping
    // the whole allocation alive while any slice exists.
    retained_ptr slice(size_t offset, size_t count) const noexcept
    {
        if (data_ == nullptr || offset >= size_)
        {
            return {};
        }
        size_t const n = (count < size_ - offset) ? count : (size_ - offset);
        if (cb_)
        {
            cb_->add_ref();
        }
        return retained_ptr(cb_, data_ + offset, n);
    }

    // --- Accessors ---

    T*     data()  const noexcept { return data_; }
    T*     get()   const noexcept { return data_; }
    T*     begin() const noexcept { return data_; }
    T*     end()   const noexcept { return data_ ? data_ + size_ : nullptr; }
    size_t size()  const noexcept { return size_; }
    bool   empty() const noexcept { return data_ == nullptr || size_ == 0; }
    explicit operator bool() const noexcept { return data_ != nullptr; }

    // Allocation base (always cb_->base, independent of slice offset).
    T* base() const noexcept { return cb_ ? static_cast<T*>(cb_->base()) : nullptr; }

    storage_identity const& identity() const noexcept
    {
        static constexpr storage_identity k{};
        return cb_ ? cb_->identity() : k;
    }

    execution_context const& ctx() const noexcept
    {
        static constexpr execution_context k{};
        return cb_ ? cb_->ctx() : k;
    }

    // Shared reference count (0 for null/empty).
    int32_t use_count() const noexcept
    {
        return cb_ ? cb_->use_count() : 0;
    }

private:
    // State for the callable overload of adopt().
    struct adopt_thunk
    {
        std::function<void(T*, size_t, execution_context const&)> fn;
        execution_context                                         ctx;
        size_t                                                    capacity;
    };

    static void adopt_thunk_free(void* thunk_ptr, void* base, std::size_t) noexcept
    {
        std::unique_ptr<adopt_thunk> thunk(static_cast<adopt_thunk*>(thunk_ptr));
        try
        {
            thunk->fn(static_cast<T*>(base), thunk->capacity, thunk->ctx);
        }
        catch (...)
        {
            cleanup_diagnostic::record_failure(cleanup_source::retained_ptr);
        }
    }

    // Shared argument checks for both adopt() overloads. Returns false for the one
    // empty adoption (null base, zero capacity); throws for every invalid pairing.
    static bool validate_adoption(T* data, size_t capacity)
    {
        if (data == nullptr)
        {
            if (capacity != 0)
            {
                throw std::invalid_argument(
                    "retained_ptr::adopt: null base pointer with a non-zero capacity");
            }
            return false;
        }
        if (capacity == 0)
        {
            throw std::invalid_argument(
                "retained_ptr::adopt: non-null base pointer with zero capacity; "
                "the caller keeps ownership");
        }
        if (capacity > std::numeric_limits<size_t>::max() / sizeof(T))
        {
            throw std::overflow_error("retained_ptr::adopt: capacity * sizeof(T) overflows size_t");
        }
        return true;
    }

    // Takes over one existing reference of the storage (does not add one).
    retained_ptr(shared_storage* cb, T* data, size_t size) noexcept
        : cb_(cb), data_(data), size_(size)
    {
    }

    void release() noexcept
    {
        shared_storage::release(cb_);  // last owner destroys the handle: one free path
        cb_   = nullptr;
        data_ = nullptr;
        size_ = 0;
    }

    shared_storage* cb_{nullptr};
    T*             data_{nullptr};  // view start (may be offset from cb_->base)
    size_t         size_{0};        // view element count
};

/// Allocate count uninitialized elements and return them under shared ownership
/// (c10 make_intrusive analogue). Zero count yields an empty pointer.
/// Throws overflow_error when count * sizeof(T) overflows.
template <typename T>
retained_ptr<T> make_retained(std::size_t count, execution_context ctx)
{
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
        throw std::overflow_error("make_retained: count * sizeof(T) overflows size_t");
    }
    detail::validate_context_stream(ctx);
    return retained_ptr<T>::from_storage(
        allocate_bytes(count * sizeof(T), storage_max_alignment, ctx), ctx);
}

}  // namespace memory
