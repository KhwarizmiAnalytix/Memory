#pragma once

#include <cstddef>
#include <cstdint>

#include "allocator.h"
#include "common/device.h"
#include "common/memory_macros.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DATA_VIEW_GPU_CALLABLE __host__ __device__
#else
#define DATA_VIEW_GPU_CALLABLE
#endif

namespace memory
{
template <typename value_t>
struct data_ptr;

/**
 * Non-owning window over a data_ptr buffer. Copy/move alias the pointer; the
 * destructor does not free. Construct from a data_ptr (full view or a slice).
 * Foreign memory that is not owned by a data_ptr uses borrow().
 */
template <typename value_t>
struct data_view
{
    using allocator_t = allocator<value_t>;
    using stream_t    = typename allocator_t::stream_t;

    MEMORY_FORCE_INLINE data_view() = default;

    MEMORY_FORCE_INLINE data_view(data_ptr<value_t> const& owner) noexcept;
    MEMORY_FORCE_INLINE data_view(
        data_ptr<value_t> const& owner, size_t offset, size_t count) noexcept;

    MEMORY_FORCE_INLINE data_view subview(size_t offset, size_t count) const noexcept
    {
        if (data_ == nullptr || offset >= size_)
        {
            return data_view();
        }
        size_t const remaining = size_ - offset;
        size_t const n         = count < remaining ? count : remaining;
        return data_view(data_ + offset, n, type_, device_index_, stream_);
    }

    static DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE data_view borrow(
        value_t*    data,
        size_t      size,
        device_enum type,
        int         device_index = 0,
        stream_t    stream       = nullptr) noexcept
    {
        return data_view(data, size, type, device_index, stream);
    }

    // Handle constness: a const view does not freeze the buffer (same as std::span<T>).
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* data() const { return data_; }
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* get() const { return data_; }
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* begin() const { return data(); }
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* end() const { return data() + size_; }

    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE size_t size() const { return size_; }
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE bool   is_aligned() const { return aligned_; }
    MEMORY_FORCE_INLINE int                           device_index() const { return device_index_; }
    MEMORY_FORCE_INLINE device_enum                   device() const { return type_; }
    MEMORY_FORCE_INLINE stream_t                      stream() const { return stream_; }

    MEMORY_FORCE_INLINE void record_stream(stream_t stream) const
    {
        allocator_t::record_stream(data_, type_, device_index_, stream);
    }

private:
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE data_view(
        value_t* data, size_t size, device_enum type, int device_index, stream_t stream) noexcept
        : data_(data),
          size_(size),
          type_(type),
          device_index_(device_index),
          stream_(stream),
          aligned_(is_ptr_aligned(data))
    {
    }

    static DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE bool is_ptr_aligned(
        value_t const* ptr) noexcept
    {
        return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) % MEMORY_ALIGNMENT == 0);
    }

    value_t*    data_{nullptr};
    size_t      size_{0};
    device_enum type_{device_enum::CPU};
    int         device_index_{0};
    stream_t    stream_{nullptr};
    bool        aligned_{false};
};
}  // namespace memory
