#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "allocator.h"
#include "common/data_view.h"
#include "common/device.h"
#include "common/execution_context.h"
#include "common/memory_macros.h"

// Trivial accessors (data/begin/end/size) are called from CUDA kernel argument
// structs via tensor's __host__ __device__ accessors.  Annotate them so Clang
// CUDA does not reject the call as a __host__-only reference.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define DATA_PTR_GPU_CALLABLE __host__ __device__
#else
#define DATA_PTR_GPU_CALLABLE
#endif

namespace memory
{
/**
 * Unique owning typed buffer. Copy always deep-clones; move transfers
 * ownership. data_view<T> is a non-owning window over a data_ptr buffer.
 */
template <typename value_t>
struct data_ptr
{
    using allocator_t = allocator<value_t>;
    using stream_t    = typename allocator_t::stream_t;

    MEMORY_FORCE_INLINE data_ptr() = default;

    // Allocate from execution context (preferred API)
    MEMORY_FORCE_INLINE data_ptr(size_t size, execution_context ctx)
        : size_(size), ctx_(ctx), aligned_(true)
    {
        if (size == 0)
        {
            return;
        }
        data_ = allocator_t::allocate(size, ctx);
    }

    // Allocate from separate device/stream parameters (backward compatible)
    MEMORY_FORCE_INLINE data_ptr(
        size_t size, device_enum type, int device_index = 0, stream_t stream = nullptr)
        : data_ptr(size, execution_context{type, device_index, stream})
    {
    }

    // Adopt by cloning: allocate owned storage and copy @p data into it.
    MEMORY_FORCE_INLINE data_ptr(
        value_t const* data,
        size_t         size,
        device_enum    type,
        int            device_index = 0,
        stream_t       stream       = nullptr)
        : data_ptr(size, type, device_index, stream)
    {
        if (data != nullptr && data_ != nullptr && size != 0)
        {
            allocator_t::copy(data, size, data_, type, type, device_index, device_index, stream);
        }
    }

    MEMORY_FORCE_INLINE data_ptr(
        value_t const* data,
        size_t         size,
        device_enum    from_type,
        device_enum    to_type,
        int            from_index = 0,
        int            to_index   = 0,
        stream_t       stream     = nullptr)
        : data_ptr(size, to_type, to_index, stream)
    {
        if (data != nullptr && data_ != nullptr && size != 0)
        {
            allocator_t::copy(data, size, data_, from_type, to_type, from_index, to_index, stream);
        }
    }

    MEMORY_FORCE_INLINE explicit data_ptr(data_view<value_t> const& view)
        : data_ptr(view.data(), view.size(), view.device(), view.device_index(), view.stream())
    {
    }

    MEMORY_FORCE_INLINE data_ptr(data_ptr const& rhs) : data_ptr(rhs.view()) {}

    MEMORY_FORCE_INLINE data_ptr& operator=(data_ptr const& rhs)
    {
        if (this == &rhs)
        {
            return *this;
        }
        data_ptr tmp(rhs);
        *this = std::move(tmp);
        return *this;
    }

    MEMORY_FORCE_INLINE data_ptr(data_ptr&& rhs) noexcept
        : data_(rhs.data_), size_(rhs.size_), ctx_(rhs.ctx_), aligned_(rhs.aligned_)
    {
        rhs.clear_handle();
    }

    MEMORY_FORCE_INLINE data_ptr& operator=(data_ptr&& rhs)
    {
        if (this == &rhs)
        {
            return *this;
        }
        release_owned();
        data_    = rhs.data_;
        size_    = rhs.size_;
        ctx_     = rhs.ctx_;
        aligned_ = rhs.aligned_;
        rhs.clear_handle();
        return *this;
    }

    // Destructors are implicitly noexcept: release_owned() reaching the GPU
    // caching allocator's deallocate()/insert_events_locked() can throw (an
    // ownership-check failure, or a CUDA/HIP driver error surfaced while
    // recording a cross-stream event), and an exception leaving an implicitly
    // noexcept function calls std::terminate immediately -- not only during
    // unwinding. There is no safe recovery from a driver error at this point,
    // so the buffer is abandoned (leaked) rather than crashing the process.
    MEMORY_FORCE_INLINE ~data_ptr()
    {
        try
        {
            release_owned();
        }
        catch (...)
        {
        }
    }

    MEMORY_FORCE_INLINE data_view<value_t> view() const noexcept
    {
        return data_view<value_t>(*this);
    }

    MEMORY_FORCE_INLINE data_view<value_t> view(size_t offset, size_t count) const noexcept
    {
        return data_view<value_t>(*this, offset, count);
    }

    // Handle constness: a const data_ptr does not freeze the buffer (same as std::span<T>).
    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* data() const { return data_; }
    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* get() const { return data_; }
    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* begin() const { return data(); }
    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* end() const { return data() + size_; }

    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE size_t size() const { return size_; }
    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE bool   is_aligned() const { return aligned_; }
    MEMORY_FORCE_INLINE int                          device_index() const { return ctx_.device_index; }
    MEMORY_FORCE_INLINE device_enum                  device() const { return ctx_.device_type; }
    MEMORY_FORCE_INLINE stream_t                     stream() const { return ctx_.stream; }
    MEMORY_FORCE_INLINE execution_context            context() const { return ctx_; }

    MEMORY_FORCE_INLINE void record_stream(stream_t stream) const
    {
        allocator_t::record_stream(data_, ctx_.device_type, ctx_.device_index, stream);
    }

    friend struct data_view<value_t>;

private:
    MEMORY_FORCE_INLINE void release_owned()
    {
        if (data_ != nullptr)
        {
            allocator_t::free(data_, ctx_.device_type, ctx_.device_index, 0, ctx_.stream);
            data_ = nullptr;
        }
    }

    MEMORY_FORCE_INLINE void clear_handle() noexcept
    {
        data_    = nullptr;
        size_    = 0;
        ctx_     = execution_context::cpu();
        aligned_ = false;
    }

    value_t*         data_{nullptr};
    size_t           size_{0};
    execution_context ctx_{execution_context::cpu()};
    bool             aligned_{false};
};

// Handle-based copy_async: both endpoints are data_ptr base allocations so
// record_stream is guaranteed to find them in the caching allocator (no
// interior or foreign pointer risk). sizeof(Src) must equal sizeof(Dst);
// element counts must match. The caller must keep both data_ptrs alive until
// token.wait() returns — the token does not retain them.
template <typename Src, typename Dst>
MEMORY_FORCE_INLINE copy_token copy_async(
    data_ptr<Src> const&                from,
    data_ptr<Dst>&                      to,
    typename allocator<Src>::stream_t   stream)
{
    static_assert(
        sizeof(Src) == sizeof(Dst),
        "copy_async: source and destination element sizes must match");
    if (from.size() != to.size())
    {
        throw std::invalid_argument("copy_async: element count mismatch between endpoints");
    }
    // Byte-level copy via allocator<uint8_t> so the two element types need not
    // be the same type — only the same size (checked above via static_assert).
    return allocator<uint8_t>::copy_async(
        reinterpret_cast<const uint8_t*>(from.data()),
        from.size() * sizeof(Src),
        reinterpret_cast<uint8_t*>(to.data()),
        stream,
        from.device(),
        to.device(),
        from.device_index(),
        to.device_index());
}

template <typename value_t>
MEMORY_FORCE_INLINE data_view<value_t>::data_view(data_ptr<value_t> const& owner) noexcept
    : data_view(owner.data_, owner.size_, owner.ctx_.device_type, owner.ctx_.device_index,
                owner.ctx_.stream, owner.data_)
{
}

template <typename value_t>
MEMORY_FORCE_INLINE data_view<value_t>::data_view(
    data_ptr<value_t> const& owner, size_t offset, size_t count) noexcept
    : data_view(data_view(owner).subview(offset, count))
{
}
}  // namespace memory
