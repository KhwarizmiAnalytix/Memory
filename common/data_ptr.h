#pragma once

#include <cstddef>
#include <cstdint>

#include "allocator.h"
#include "common/device.h"
#include "common/memory_macros.h"

namespace memory
{
template <typename value_t, bool clone>
struct data_ptr
{
    using allocator_t = allocator<value_t>;

    MEMORY_FORCE_INLINE data_ptr() = default;

    MEMORY_FORCE_INLINE data_ptr(size_t size, device_enum type)
        : data_(allocator_t::allocate(size, type)),
          size_(size),
          type_(type),
          allocated_(true),
          aligned_(true)  // allocator always satisfies MEMORY_ALIGNMENT
    {
    }

    MEMORY_FORCE_INLINE data_ptr(value_t* data, size_t size, device_enum type)
        : size_(size), type_(type), allocated_(clone)
    {
        if constexpr (clone)
        {
            data_ = allocator_t::allocate(size, type), aligned_ = true;
            allocator_t::copy(data, size_, data_, type, type_);
        }
        else
        {
            data_    = data;
            aligned_ = is_ptr_aligned(data);
        }
    }

    MEMORY_FORCE_INLINE data_ptr(
        value_t* data, size_t size, device_enum from_type, device_enum to_type)
        : size_(size), type_(to_type), allocated_(clone)
    {
        if constexpr (clone)
        {
            data_    = allocator_t::allocate(size, to_type);
            aligned_ = true;
            allocator_t::copy(data, size_, data_, from_type, to_type);
        }
        else
        {
            data_    = data;
            aligned_ = is_ptr_aligned(data);
        }
    }

    // Copy — behaviour controlled by clone template parameter
    MEMORY_FORCE_INLINE data_ptr(data_ptr const& rhs)
        : size_(rhs.size_), type_(rhs.type_), allocated_(clone)
    {
        if constexpr (clone)
        {
            data_    = allocator_t::allocate(size_, type_);
            aligned_ = true;
            allocator_t::copy(rhs.data_, size_, data_, rhs.type_, type_);
        }
        else
        {
            data_    = rhs.data_;
            aligned_ = rhs.aligned_;
        }
    }

    MEMORY_FORCE_INLINE data_ptr& operator=(data_ptr const& rhs)
    {
        if (this == &rhs)
        {
            return *this;
        }

        size_      = rhs.size_;
        type_      = rhs.type_;
        allocated_ = clone;

        if constexpr (clone)
        {
            data_    = allocator_t::allocate(size_, type_);
            aligned_ = true;
            allocator_t::copy(rhs.data_, size_, data_, rhs.type_, type_);
        }
        else
        {
            data_    = rhs.data_;
            aligned_ = rhs.aligned_;
        }

        return *this;
    }

    // Move — always transfers ownership, independent of clone
    MEMORY_FORCE_INLINE data_ptr(data_ptr&& rhs) noexcept
        : data_(std::move(rhs.data_)),
          size_(std::move(rhs.size_)),
          type_(std::move(rhs.type_)),
          allocated_(std::move(rhs.allocated_)),
          aligned_(std::move(rhs.aligned_))
    {
        rhs.data_      = nullptr;
        rhs.allocated_ = false;
        rhs.aligned_   = false;
    }

    MEMORY_FORCE_INLINE data_ptr& operator=(data_ptr&& rhs) noexcept
    {
        if (this == &rhs)
        {
            return *this;
        }

        data_      = std::move(rhs.data_);
        size_      = std::move(rhs.size_);
        type_      = std::move(rhs.type_);
        allocated_ = std::move(rhs.allocated_);
        aligned_   = std::move(rhs.aligned_);

        rhs.data_      = nullptr;
        rhs.allocated_ = false;
        rhs.aligned_   = false;

        return *this;
    }

    MEMORY_FORCE_INLINE ~data_ptr()
    {
        if (allocated_ && data_ != nullptr)
        {
            allocator_t::free(data_);
            data_ = nullptr;
        }
    }

    MEMORY_FORCE_INLINE void copy(data_ptr const& rhs)
    {
        if (data_ == nullptr)
        {
            data_      = allocator_t::allocate(rhs.size_, rhs.type_);
            type_      = rhs.type_;
            size_      = rhs.size_;
            allocated_ = true;
            aligned_   = true;
        }
        if (data_ != rhs.data_)
        {
            allocator_t::copy(rhs.data_, rhs.size_, data_, rhs.type_, type_);
        }
    }

    MEMORY_FORCE_INLINE const value_t* data() const { return data_; }
    MEMORY_FORCE_INLINE const value_t* get() const { return data_; }
    MEMORY_FORCE_INLINE const value_t* begin() const { return data(); }
    MEMORY_FORCE_INLINE const value_t* end() const { return data() + size_; }

    MEMORY_FORCE_INLINE value_t* data() { return data_; }
    MEMORY_FORCE_INLINE value_t* get() { return data_; }
    MEMORY_FORCE_INLINE value_t* begin() { return data(); }
    MEMORY_FORCE_INLINE value_t* end() { return data() + size_; }

    MEMORY_FORCE_INLINE size_t size() const { return size_; }
    MEMORY_FORCE_INLINE bool   is_aligned() const { return aligned_; }

    value_t*    data_{nullptr};
    size_t      size_{0};
    device_enum type_{device_enum::CPU};
    bool        allocated_{false};
    bool        aligned_{false};

private:
    static MEMORY_FORCE_INLINE bool is_ptr_aligned(value_t const* ptr) noexcept
    {
        return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) % MEMORY_ALIGNMENT == 0);
    }
};
}  // namespace memory
