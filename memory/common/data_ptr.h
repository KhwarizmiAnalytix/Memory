#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "allocator.h"
#include "common/data_view.h"
#include "common/device.h"
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

    MEMORY_FORCE_INLINE data_ptr(
        size_t size, device_enum type, int device_index = 0, stream_t stream = nullptr)
        : size_(size),
          type_(type),
          device_index_(device_index),
          stream_(stream),
          allocated_(false),
          aligned_(true)
    {
        if (size == 0)
        {
            return;
        }
        data_      = allocator_t::allocate(size, type, device_index, stream);
        allocated_ = true;
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
        : data_(rhs.data_),
          size_(rhs.size_),
          type_(rhs.type_),
          device_index_(rhs.device_index_),
          stream_(rhs.stream_),
          allocated_(rhs.allocated_),
          aligned_(rhs.aligned_)
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
        data_         = rhs.data_;
        size_         = rhs.size_;
        type_         = rhs.type_;
        device_index_ = rhs.device_index_;
        stream_       = rhs.stream_;
        allocated_    = rhs.allocated_;
        aligned_      = rhs.aligned_;
        rhs.clear_handle();
        return *this;
    }

    MEMORY_FORCE_INLINE ~data_ptr() { release_owned(); }

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
    MEMORY_FORCE_INLINE int                          device_index() const { return device_index_; }
    MEMORY_FORCE_INLINE device_enum                  device() const { return type_; }
    MEMORY_FORCE_INLINE stream_t                     stream() const { return stream_; }

    MEMORY_FORCE_INLINE void record_stream(stream_t stream) const
    {
        allocator_t::record_stream(data_, type_, device_index_, stream);
    }

    friend struct data_view<value_t>;

private:
    MEMORY_FORCE_INLINE void release_owned()
    {
        if (allocated_ && data_ != nullptr)
        {
            allocator_t::free(data_, type_, device_index_, 0, stream_);
            data_      = nullptr;
            allocated_ = false;
        }
    }

    MEMORY_FORCE_INLINE void clear_handle() noexcept
    {
        data_         = nullptr;
        size_         = 0;
        type_         = device_enum::CPU;
        device_index_ = 0;
        stream_       = nullptr;
        allocated_    = false;
        aligned_      = false;
    }

    value_t*    data_{nullptr};
    size_t      size_{0};
    device_enum type_{device_enum::CPU};
    int         device_index_{0};
    stream_t    stream_{nullptr};
    bool        allocated_{false};
    bool        aligned_{false};
};

template <typename value_t>
MEMORY_FORCE_INLINE data_view<value_t>::data_view(data_ptr<value_t> const& owner) noexcept
    : data_view(owner.data_, owner.size_, owner.type_, owner.device_index_, owner.stream_)
{
}

template <typename value_t>
MEMORY_FORCE_INLINE data_view<value_t>::data_view(
    data_ptr<value_t> const& owner, size_t offset, size_t count) noexcept
    : data_view(data_view(owner).subview(offset, count))
{
}
}  // namespace memory
