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

#include "common/cleanup_diagnostic.h"
#include "common/execution_context.h"
#include "common/retained_ptr.h"
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
    ~pinned_buffer() { release_counted(); }
    pinned_buffer(const pinned_buffer&)            = delete;
    pinned_buffer& operator=(const pinned_buffer&) = delete;
    pinned_buffer(pinned_buffer&& other) noexcept { swap(other); }
    pinned_buffer& operator=(pinned_buffer&& other) noexcept
    {
        if (this != &other)
        {
            release_counted();
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
    /// Hand the storage to a shared `retained_ptr<T>` (plan 6.3) so a retained copy
    /// (`allocator<T>::copy_async_retained`) keeps it alive until the transfer
    /// completes, however many user handles are dropped meanwhile. The last owner
    /// returns the block to its pinned pool; a failed return is counted in
    /// `cleanup_diagnostic`. The endpoint is a host (CPU-context) endpoint. On a
    /// throw (bad_alloc from the control block) this buffer still owns the storage.
    retained_ptr<T> into_retained() &&
    {
        if (data_ == nullptr)
        {
            return {};
        }
        retained_ptr<T> shared = retained_ptr<T>::adopt(
            data_, size_, execution_context::cpu(), &free_to_pool, allocator_);
        allocator_ = nullptr;  // ownership moved only now that adopt() succeeded
        data_      = nullptr;
        size_      = 0;
        return shared;
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
    static void free_to_pool(void* pool, void* ptr, std::size_t) noexcept
    {
        auto* allocator = static_cast<cpu::pinned_memory_allocator*>(pool);
        if (allocator != nullptr && !allocator->deallocate(ptr))
        {
            cleanup_diagnostic::record_failure(cleanup_source::pinned_buffer);
        }
    }

    // Destructor/move-assign release: reset() cannot throw, but a false result
    // means the storage was quarantined; a swallowed failure is not a successful
    // cleanup (plan §5.2), so it is counted.
    void release_counted() noexcept
    {
        if (!reset())
        {
            cleanup_diagnostic::record_failure(cleanup_source::pinned_buffer);
        }
    }

    cpu::pinned_memory_allocator* allocator_{nullptr};
    T*                            data_{nullptr};
    std::size_t                   size_{0};
};
}  // namespace memory
