/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#pragma once

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "helper/pinned_memory_allocator.h"

namespace memory
{
/// Move-only, uninitialized host storage for trivially copyable transfer data.
/// Destruction defers recycling until all registered streams complete.
template <typename T>
class pinned_buffer
{
    static_assert(
        std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>,
        "pinned_buffer requires trivially copyable transfer data");
    static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>);
    static_assert(
        alignof(T) <= cpu::pinned_memory_allocator::alignment,
        "pinned_buffer supports alignment up to 64 bytes");

public:
    using stream_type = cpu::pinned_memory_allocator::stream_type;

    pinned_buffer() noexcept = default;
    explicit pinned_buffer(std::size_t count, int device = 0)
    {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::length_error("pinned_buffer size overflows");
        }
        if (count != 0)
        {
            allocator_ = &cpu::pinned_allocator_for_device(device);
            data_      = static_cast<T*>(allocator_->allocate(count * sizeof(T)));
            size_      = count;
        }
    }
    ~pinned_buffer() { reset(); }
    pinned_buffer(const pinned_buffer&)            = delete;
    pinned_buffer& operator=(const pinned_buffer&) = delete;
    pinned_buffer(pinned_buffer&& other) noexcept { swap(other); }
    pinned_buffer& operator=(pinned_buffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            swap(other);
        }
        return *this;
    }

    T*          data() noexcept { return data_; }
    const T*    data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }

    void record_stream(stream_type stream) const
    {
        if (allocator_)
            allocator_->record_stream(data_, stream);
    }
    void copy_to_device_async(T* destination, stream_type stream = nullptr) const
    {
        if (allocator_)
            allocator_->copy_to_device_async(destination, data_, size_bytes(), stream);
    }
    void copy_from_device_async(const T* source, stream_type stream = nullptr)
    {
        if (allocator_)
            allocator_->copy_from_device_async(data_, source, size_bytes(), stream);
    }
    /// A false result indicates a runtime failure; storage is quarantined safely.
    bool reset() noexcept
    {
        const bool result = !allocator_ || allocator_->deallocate(data_);
        allocator_        = nullptr;
        data_             = nullptr;
        size_             = 0;
        return result;
    }
    void swap(pinned_buffer& other) noexcept
    {
        std::swap(allocator_, other.allocator_);
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

private:
    cpu::pinned_memory_allocator* allocator_{nullptr};
    T*                            data_{nullptr};
    std::size_t                   size_{0};
};
}  // namespace memory
