#pragma once

#include <cstddef>
#include <cstdint>

#include "allocator.h"
#include "common/device.h"
#include "common/memory_macros.h"
#include "common/storage_identity.h"

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
 * Identity of the allocation a view windows into (plan 4.3, task 2.5): the
 * allocation's base pointer, its allocation_id and its device. A slice of a
 * slice keeps all three. Memory not owned by this library (borrow()) has an
 * invalid id: it has no provenance the allocators could look up.
 */
template <typename value_t>
struct storage_ref
{
    value_t*      base{nullptr};
    allocation_id id{};
    device        dev{};
};

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
        return data_view(data_ + offset, n, ref_, stream_);
    }

    static DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE data_view borrow(
        value_t*    data,
        size_t      size,
        device_enum type,
        int         device_index = 0,
        stream_t    stream       = nullptr) noexcept
    {
        return data_view(
            data,
            size,
            storage_ref<value_t>{
                data, allocation_id{}, memory::device{type, static_cast<std::int16_t>(device_index)}},
            stream);
    }

    // Handle constness: a const view does not freeze the buffer (same as std::span<T>).
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* data() const { return data_; }
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* get() const { return data_; }
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* begin() const { return data(); }
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* end() const { return data() + size_; }

    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE size_t size() const { return size_; }
    /** Allocation's own base pointer, independent of any offset this view was sliced to. */
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* base() const { return ref_.base; }
    /** Allocation id of the owner; invalid for default-constructed and borrowed views. */
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE allocation_id id() const { return ref_.id; }
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE storage_ref<value_t> const& storage() const
    {
        return ref_;
    }
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE bool     is_aligned() const { return aligned_; }
    MEMORY_FORCE_INLINE int         device_index() const { return ref_.dev.index; }
    MEMORY_FORCE_INLINE device_enum device() const { return ref_.dev.type; }
    MEMORY_FORCE_INLINE stream_t    stream() const { return stream_; }

    /**
     * @brief Record a cross-stream use of this view's underlying allocation.
     *
     * Forwards the allocation's own base pointer (not this view's, possibly
     * offset, `data()`), since the CUDA/HIP caching allocator tracks live
     * blocks by their exact base address. A sliced/windowed view passing its
     * own interior pointer would fail that lookup (or, worse, silently miss
     * the intended block) — `base_` is threaded through subview()/borrow() so
     * it always names the allocation the caching allocator actually knows
     * about, however many slices deep this view is.
     */
    MEMORY_FORCE_INLINE void record_stream(stream_t stream) const
    {
        allocator_t::record_stream(ref_.base, ref_.dev.type, ref_.dev.index, stream);
    }

private:
    DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE data_view(
        value_t* data, size_t size, storage_ref<value_t> const& ref, stream_t stream) noexcept
        : data_(data), size_(size), ref_(ref), stream_(stream), aligned_(is_ptr_aligned(data))
    {
    }

    static DATA_VIEW_GPU_CALLABLE MEMORY_FORCE_INLINE bool is_ptr_aligned(
        value_t const* ptr) noexcept
    {
        return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) % MEMORY_ALIGNMENT == 0);
    }

    value_t*           data_{nullptr};
    size_t             size_{0};
    storage_ref<value_t> ref_{};  // base/id/device of the allocation, however deep the slice
    stream_t           stream_{nullptr};
    bool               aligned_{false};
};
}  // namespace memory
