#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "allocator.h"
#include "common/cleanup_diagnostic.h"
#include "common/data_view.h"
#include "common/execution_context.h"
#include "common/memory_macros.h"
#include "common/storage_handle.h"

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

// Forward declaration so retained_ptr<T> can be befriended (definition is in
// retained_ptr.h, included transitively through allocator.h above).
template <typename T>
class retained_ptr;

/**
 * Unique owning typed buffer (P2: backed by storage_handle).
 *
 * Layout: storage_handle handle_ (48 B) + stream_handle_t stream_ (8 B) = 56 B.
 * stream_ is kept separate because it is needed for the user-facing API
 * (stream(), record_stream(), clone(), context()); the free path no longer
 * requires it — handle_'s destructor calls gpu_free_fn which looks up the
 * allocation stream from the cache block (plan §2.10, R1).
 */
template <typename value_t>
struct data_ptr
{
    using allocator_t = allocator<value_t>;
    using stream_t    = typename allocator_t::stream_t;

    MEMORY_FORCE_INLINE data_ptr() = default;

    // Allocate from execution context (preferred API).
    // Zero size: empty storage (no memory, invalid allocation_id).
    MEMORY_FORCE_INLINE data_ptr(size_t size, execution_context ctx)
        : stream_(ctx.stream)
    {
        handle_ = allocate_bytes(size * sizeof(value_t), allocator_t::alignment_bytes, ctx);
    }

    // Allocate from separate device/stream parameters (backward compatible).
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
        if (data != nullptr && !handle_.empty() && size != 0)
        {
            allocator_t::copy(data, size, this->data(), type, type,
                              device_index, device_index, stream);
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
        if (data != nullptr && !handle_.empty() && size != 0)
        {
            allocator_t::copy(data, size, this->data(), from_type, to_type,
                              from_index, to_index, stream);
        }
    }

    MEMORY_FORCE_INLINE explicit data_ptr(data_view<value_t> const& view)
        : data_ptr(view.data(), view.size(), view.device(), view.device_index(), view.stream())
    {
    }

    data_ptr(data_ptr const&) = delete;
    data_ptr& operator=(data_ptr const&) = delete;

    MEMORY_FORCE_INLINE data_ptr(data_ptr&& rhs) noexcept
        : handle_(std::move(rhs.handle_)), stream_(rhs.stream_)
    {
        rhs.stream_ = nullptr;
    }

    MEMORY_FORCE_INLINE data_ptr& operator=(data_ptr&& rhs) noexcept
    {
        if (this == &rhs)
        {
            return *this;
        }
        handle_ = std::move(rhs.handle_);
        stream_ = rhs.stream_;
        rhs.stream_ = nullptr;
        return *this;
    }

    // handle_'s destructor runs its deleter (cpu_free_fn or gpu_free_fn). A
    // deleter_fn is noexcept by type; gpu_free_fn catches ownership-check and
    // driver failures inside the cache and counts them in cleanup_diagnostic
    // (source gpu_cache), so no failure can leave this destructor (task 1.3).
    MEMORY_FORCE_INLINE ~data_ptr() = default;

    MEMORY_FORCE_INLINE data_view<value_t> view() const noexcept
    {
        return data_view<value_t>(*this);
    }

    MEMORY_FORCE_INLINE data_view<value_t> view(size_t offset, size_t count) const noexcept
    {
        return data_view<value_t>(*this, offset, count);
    }

    MEMORY_FORCE_INLINE data_ptr clone() const
    {
        if (handle_.empty()) return {};
        return data_ptr(data(), size(), handle_.dev().type,
                        static_cast<int>(handle_.dev().index), stream_);
    }

    // Accessors — all derived from handle_ and stream_.
    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* data()  const
    {
        return static_cast<value_t*>(handle_.get());
    }
    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* get()   const { return data(); }
    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* begin() const { return data(); }
    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE value_t* end()   const
    {
        return data() + size();
    }

    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE size_t size() const
    {
        return handle_.empty() ? 0 : handle_.nbytes() / sizeof(value_t);
    }
    // True when the data pointer meets the allocator's alignment; an empty
    // handle (default-constructed, moved-from or zero-size) has nothing to misalign.
    DATA_PTR_GPU_CALLABLE MEMORY_FORCE_INLINE bool is_aligned() const
    {
        return handle_.empty() ||
               (reinterpret_cast<std::uintptr_t>(handle_.get()) &
                (allocator_t::alignment_bytes - 1)) == 0;
    }

    MEMORY_FORCE_INLINE int          device_index() const
    {
        return static_cast<int>(handle_.dev().index);
    }
    MEMORY_FORCE_INLINE device_enum  device()  const { return handle_.dev().type;  }
    MEMORY_FORCE_INLINE stream_t     stream()  const { return static_cast<stream_t>(stream_); }
    MEMORY_FORCE_INLINE execution_context context() const
    {
        return execution_context{handle_.dev().type,
                                 static_cast<int>(handle_.dev().index),
                                 stream_};
    }

    MEMORY_FORCE_INLINE allocation_id id() const { return handle_.id(); }

    MEMORY_FORCE_INLINE void record_stream(stream_t stream) const
    {
        allocator_t::record_stream(data(), handle_.dev().type,
                                   static_cast<int>(handle_.dev().index), stream);
    }

    // Grant retained_ptr<T> access to handle_ and stream_ for the promotion
    // constructor (defined in data_ptr.h after retained_ptr<T> is complete).
    template <typename U>
    friend class retained_ptr;
    friend struct data_view<value_t>;

private:
    storage_handle  handle_{};
    stream_handle_t stream_{nullptr};
};

// ---------------------------------------------------------------------------
// Free function: handle-based copy_async
// ---------------------------------------------------------------------------

// Both endpoints are data_ptr base allocations so record_stream is guaranteed
// to find them in the caching allocator (no interior or foreign pointer risk).
// sizeof(Src) must equal sizeof(Dst); element counts must match.  The caller
// must keep both data_ptrs alive until token.wait() returns.
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

// ---------------------------------------------------------------------------
// Out-of-class definitions: data_view constructors from data_ptr
// ---------------------------------------------------------------------------

template <typename value_t>
MEMORY_FORCE_INLINE data_view<value_t>::data_view(data_ptr<value_t> const& owner) noexcept
    : data_view(owner.data(), owner.size(), owner.device(), owner.device_index(),
                owner.stream(), owner.data())
{
}

template <typename value_t>
MEMORY_FORCE_INLINE data_view<value_t>::data_view(
    data_ptr<value_t> const& owner, size_t offset, size_t count) noexcept
    : data_view(data_view(owner).subview(offset, count))
{
}

// ---------------------------------------------------------------------------
// Out-of-class definition: retained_ptr<T>(data_ptr<T>&&) promotion ctor
// ---------------------------------------------------------------------------
// Both retained_ptr<T> (included through allocator.h → retained_ptr.h) and
// data_ptr<T> are complete at this point, so the body can reference both.

template <typename T>
retained_ptr<T>::retained_ptr(data_ptr<T>&& dp)
{
    if (dp.handle_.empty())
    {
        return;
    }
    auto* cb       = new control_block();
    cb->base       = static_cast<T*>(dp.handle_.get());
    cb->capacity   = dp.handle_.nbytes() / sizeof(T);
    cb->nbytes     = dp.handle_.nbytes();
    cb->stream     = dp.stream_;
    cb->ctx        = execution_context{dp.handle_.dev().type,
                                       static_cast<int>(dp.handle_.dev().index),
                                       dp.stream_};
    cb->identity.alloc_id = dp.handle_.id().value;
    cb->identity.base     = dp.handle_.get();
    cb->identity.capacity = dp.handle_.nbytes();
    // Extract the raw deleter and cache context from the storage_handle before
    // disarming it (plan §2.10, R1).
    // GPU: fn_del_ctx = cache ptr, fn_del = gpu_free_fn (looks up the stream).
    // CPU: fn_del_ctx = nullptr,   fn_del = cpu_free_fn.
    cb->fn_del_ctx = dp.handle_.ctx_raw();
    cb->fn_del     = dp.handle_.fn_deleter();
    (void)dp.handle_.release();   // disarm: ownership transferred to control_block
    dp.stream_     = nullptr;
    cb_            = cb;
    data_          = cb->base;
    size_          = cb->capacity;
}

}  // namespace memory
