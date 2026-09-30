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
#include <utility>

#include "common/execution_context.h"
#include "common/memory_macros.h"
#include "common/storage_identity.h"

namespace memory
{

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
public:
    using element_type = T;

    // Per-allocation control block; one heap allocation per adopt() call.
    struct control_block
    {
        std::atomic<int32_t> ref_count{1};
        T*                   data{nullptr};      // allocation base
        size_t               capacity{0};        // element count at base
        execution_context    ctx{};
        storage_identity     identity{};
        // Deleter: called with (base, capacity, ctx) when ref_count hits 0.
        std::function<void(T*, size_t, execution_context const&)> deleter;

        control_block()                              = default;
        control_block(control_block const&)          = delete;
        control_block& operator=(control_block const&) = delete;
    };

    // --- Construction / Adoption ---

    retained_ptr() noexcept = default;

    // Adopt foreign memory: the supplied deleter takes ownership of storage.
    // The returned retained_ptr owns a single reference; callers may then
    // copy/move it to distribute access.  data == nullptr or capacity == 0
    // yields an empty (null) retained_ptr.
    static retained_ptr adopt(
        T*              data,
        size_t          capacity,
        execution_context ctx,
        std::function<void(T*, size_t, execution_context const&)> deleter)
    {
        if (data == nullptr || capacity == 0)
        {
            return {};
        }
        control_block* cb  = new control_block();
        cb->data           = data;
        cb->capacity       = capacity;
        cb->ctx            = ctx;
        cb->deleter        = std::move(deleter);
        cb->identity.alloc_id = storage_identity::next_id();
        cb->identity.base     = static_cast<void*>(data);
        cb->identity.capacity = capacity * sizeof(T);
        return retained_ptr(cb, data, capacity);
    }

    // --- Copy / Move / Destructor ---

    retained_ptr(retained_ptr const& rhs) noexcept
        : cb_(rhs.cb_), data_(rhs.data_), size_(rhs.size_)
    {
        if (cb_)
        {
            cb_->ref_count.fetch_add(1, std::memory_order_relaxed);
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
            cb_->ref_count.fetch_add(1, std::memory_order_relaxed);
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
            cb_->ref_count.fetch_add(1, std::memory_order_relaxed);
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

    // Allocation base (always cb_->data, independent of slice offset).
    T* base() const noexcept { return cb_ ? cb_->data : nullptr; }

    storage_identity const& identity() const noexcept
    {
        static constexpr storage_identity k{};
        return cb_ ? cb_->identity : k;
    }

    execution_context const& ctx() const noexcept
    {
        static constexpr execution_context k{};
        return cb_ ? cb_->ctx : k;
    }

    // Shared reference count (0 for null/empty).
    int32_t use_count() const noexcept
    {
        return cb_ ? cb_->ref_count.load(std::memory_order_relaxed) : 0;
    }

private:
    retained_ptr(control_block* cb, T* data, size_t size) noexcept
        : cb_(cb), data_(data), size_(size)
    {
    }

    void release() noexcept
    {
        if (!cb_)
        {
            return;
        }
        if (cb_->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            if (cb_->deleter && cb_->data)
            {
                try
                {
                    cb_->deleter(cb_->data, cb_->capacity, cb_->ctx);
                }
                catch (...)  // NOLINT: intentionally swallow in destructor
                {
                }
            }
            delete cb_;
        }
        cb_   = nullptr;
        data_ = nullptr;
        size_ = 0;
    }

    control_block* cb_{nullptr};
    T*             data_{nullptr};  // view start (may be offset from cb_->data)
    size_t         size_{0};        // view element count
};

}  // namespace memory
