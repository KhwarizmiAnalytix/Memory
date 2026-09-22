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

#include "helper/pinned_memory_allocator.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/device_guard.h"
#endif

#if MEMORY_HAS_PROFILER
#include "gpu/caching_allocator_profiler_report.h"
#endif

namespace memory::cpu
{
#if (MEMORY_HAS_CUDA || MEMORY_HAS_HIP) && MEMORY_HAS_PROFILER
// profiler::device_enum has no dedicated "pinned host" value (CPU=0, CUDA=1,
// HIP=2, PrivateUse1=3); pinned buffers are host-resident, so they report as
// CPU, same as PyTorch's CachingHostAllocator.
constexpr int16_t kPinnedDeviceType = 0;
#endif

struct pinned_memory_allocator::Impl
{
    explicit Impl(int device, std::size_t limit) : device_(device), limit_(limit)
    {
        if (device < 0)
            throw std::invalid_argument("Negative pinned allocator device");
    }

    int                 device_;
    std::size_t         limit_;
    mutable std::mutex  mutex_;
    pinned_memory_stats stats_;

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
    struct stream_use
    {
        cudaStream_t stream;
        cudaEvent_t  event;
        bool         used;
    };
    enum class state
    {
        live,
        pending,
        cached
    };
    struct block
    {
        void*                   raw{nullptr};
        void*                   ptr{nullptr};
        std::size_t             capacity{0};
        std::size_t             requested{0};
        std::size_t             bucket{0};
        state                   status{state::live};
        bool                    quarantined{false};
        block*                  next{nullptr};
        std::vector<stream_use> streams;
    };

    // Intrusive lists avoid allocating metadata when a buffer is destroyed.
    std::array<block*, std::numeric_limits<std::size_t>::digits> cached_{};
    block*                                                       pending_{nullptr};
    std::map<std::uintptr_t, std::unique_ptr<block>>             blocks_;

    ~Impl()
    {
        // No exceptions escape teardown. If completion cannot be established,
        // leave the pinned backing to the runtime rather than free in-flight data.
        try
        {
            gpu::device_guard guard(device_);
            for (auto& entry : blocks_)
            {
                auto& b = *entry.second;
                if (b.status == state::live)
                    record_events(b);
                if (!b.quarantined && complete(b, true))
                {
                    (void)cudaFreeHost(b.raw);
                }
                for (auto& use : b.streams)
                    (void)cudaEventDestroy(use.event);
            }
        }
        catch (...)  // NOLINT(bugprone-empty-catch)
        {
            // Runtime shutdown/device failure: retain unsafe backing rather than
            // risk freeing memory whose completion could not be established.
        }
    }

    static std::size_t capacity_for(std::size_t bytes, std::size_t& bucket)
    {
        std::size_t capacity = 512;
        bucket               = 9;
        while (capacity < bytes)
        {
            if (capacity > std::numeric_limits<std::size_t>::max() / 2)
                throw std::length_error("Pinned allocation size overflows");
            capacity *= 2;
            ++bucket;
        }
        if (capacity > std::numeric_limits<std::size_t>::max() - (alignment - 1))
            throw std::length_error("Pinned allocation alignment overflows");
        return capacity;
    }

    block& live_block(const void* ptr, std::size_t bytes = 0)
    {
        const auto address = reinterpret_cast<std::uintptr_t>(ptr);
        auto       it      = blocks_.upper_bound(address);
        if (it == blocks_.begin())
            throw std::invalid_argument("Foreign pinned pointer");
        auto&      b      = *std::prev(it)->second;
        const auto offset = address - reinterpret_cast<std::uintptr_t>(b.ptr);
        if (b.status != state::live || offset >= b.requested || bytes > b.requested - offset)
            throw std::invalid_argument(
                "Pinned pointer is not live or transfer exceeds its extent");
        return b;
    }

    void use_stream(block& b, cudaStream_t stream)
    {
        for (auto& use : b.streams)
            if (use.used && use.stream == stream)
                return;
        for (auto& use : b.streams)
        {
            if (!use.used)
            {
                use.stream = stream;
                use.used   = true;
                return;
            }
        }
        // Allocate metadata before creating the runtime event or submitting work.
        b.streams.reserve(b.streams.size() + 1);
        cudaEvent_t event = nullptr;
        gpu::throw_on_cuda_error(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags");
        b.streams.push_back({stream, event, true});
    }

    void record_events(block& b) noexcept
    {
        for (const auto& use : b.streams)
        {
            if (use.used && cudaEventRecord(use.event, use.stream) != cudaSuccess)
            {
                b.quarantined = true;
                ++stats_.num_errors;
            }
        }
    }

    bool complete(block& b, bool wait) noexcept
    {
        if (b.quarantined)
            return false;
        for (auto& use : b.streams)
        {
            if (!use.used)
                continue;
            const auto result = wait ? cudaEventSynchronize(use.event) : cudaEventQuery(use.event);
            if (result == cudaErrorNotReady)
            {
                // Not sticky, but cuda_caching_allocator.cpp's identical
                // cudaEventQuery(...) == cudaErrorNotReady path clears it via
                // cudaGetLastError() so it can't be misread by a later, unrelated
                // cudaGetLastError() call; match that convention here.
                (void)cudaGetLastError();
                return false;
            }
            if (result != cudaSuccess)
            {
                b.quarantined = true;
                ++stats_.num_errors;
                return false;
            }
            use.used = false;
        }
        return true;
    }

#if MEMORY_HAS_PROFILER
    void report_event(void* ptr, std::int64_t nbytes) noexcept
    {
        gpu::report_caching_allocator_event(
            ptr, nbytes, stats_.bytes_allocated, stats_.bytes_reserved, device_, kPinnedDeviceType);
    }
#endif

    void cache(block& b) noexcept
    {
        b.status          = state::cached;
        b.next            = cached_[b.bucket];
        cached_[b.bucket] = &b;
        stats_.bytes_cached += b.capacity;
    }

    void process_pending(bool wait) noexcept
    {
        auto** link = &pending_;
        while (*link)
        {
            auto& b = **link;
            if (!complete(b, wait))
            {
                link = &b.next;
                continue;
            }
            *link = b.next;
            stats_.bytes_pending -= b.capacity;
            cache(b);
        }
    }

    void trim(std::size_t limit) noexcept
    {
        for (std::size_t i = cached_.size(); i-- > 0 && stats_.bytes_cached > limit;)
        {
            while (cached_[i] && stats_.bytes_cached > limit)
            {
                auto* b    = cached_[i];
                cached_[i] = b->next;
                stats_.bytes_cached -= b->capacity;
                if (cudaFreeHost(b->raw) != cudaSuccess)
                {
                    ++stats_.num_errors;
                    b->quarantined = true;
                    b->status      = state::pending;
                    b->next        = pending_;
                    pending_       = b;
                    stats_.bytes_pending += b->capacity;
                    continue;
                }
                stats_.bytes_reserved -= b->capacity + alignment - 1;
                ++stats_.driver_frees;
                for (auto& use : b->streams)
                    (void)cudaEventDestroy(use.event);
                blocks_.erase(reinterpret_cast<std::uintptr_t>(b->ptr));
            }
        }
    }

    void* allocate(std::size_t bytes)
    {
        if (bytes == 0)
            return nullptr;
        std::size_t       bucket   = 0;
        const auto        capacity = capacity_for(bytes, bucket);
        std::scoped_lock  lock(mutex_);
        gpu::device_guard guard(device_);
        process_pending(false);
        trim(limit_);
        block* b = cached_[bucket];
        if (b)
        {
            cached_[bucket] = b->next;
            stats_.bytes_cached -= b->capacity;
            ++stats_.cache_hits;
        }
        else
        {
            ++stats_.cache_misses;
            auto owned = std::make_unique<block>();
            auto result =
                cudaHostAlloc(&owned->raw, capacity + alignment - 1, cudaHostAllocPortable);
            if (result == cudaErrorMemoryAllocation)
            {
                // Release ready cache only; do not wait for unrelated transfers.
                trim(0);
                result =
                    cudaHostAlloc(&owned->raw, capacity + alignment - 1, cudaHostAllocPortable);
            }
            if (result == cudaErrorMemoryAllocation)
            {
                ++stats_.num_ooms;
#if MEMORY_HAS_PROFILER
                gpu::report_caching_allocator_oom(
                    static_cast<std::int64_t>(bytes),
                    stats_.bytes_allocated,
                    stats_.bytes_reserved,
                    device_,
                    kPinnedDeviceType);
#endif
                throw std::bad_alloc();
            }
            gpu::throw_on_cuda_error(result, "cudaHostAlloc");
            owned->capacity = capacity;
            owned->bucket   = bucket;
            auto space      = capacity + alignment - 1;
            owned->ptr      = owned->raw;
            (void)std::align(alignment, capacity, owned->ptr, space);
            b = owned.get();
            // Allocate the map node before transferring ownership, so cleanup is
            // well-defined if the host metadata allocation itself fails.
            try
            {
                auto inserted = blocks_.try_emplace(reinterpret_cast<std::uintptr_t>(b->ptr));
                inserted.first->second = std::move(owned);
            }
            catch (...)
            {
                (void)cudaFreeHost(owned->raw);
                throw;
            }
            ++stats_.driver_allocations;
            stats_.bytes_reserved += capacity + alignment - 1;
            stats_.peak_bytes_reserved =
                std::max(stats_.peak_bytes_reserved, stats_.bytes_reserved);
        }
        b->status    = state::live;
        b->requested = bytes;
        b->next      = nullptr;
        stats_.bytes_allocated += b->capacity;
        stats_.bytes_requested += bytes;
#if MEMORY_HAS_PROFILER
        report_event(b->ptr, static_cast<std::int64_t>(b->capacity));
#endif
        return b->ptr;
    }

    bool deallocate(void* ptr) noexcept
    {
        if (!ptr)
            return true;
        std::scoped_lock lock(mutex_);
        auto             it = blocks_.find(reinterpret_cast<std::uintptr_t>(ptr));
        if (it == blocks_.end() || it->second->status != state::live)
            return false;
        auto& b = *it->second;
        stats_.bytes_allocated -= b.capacity;
        stats_.bytes_requested -= b.requested;
        stats_.bytes_pending += b.capacity;
#if MEMORY_HAS_PROFILER
        report_event(ptr, -static_cast<std::int64_t>(b.capacity));
#endif
        b.status = state::pending;
        b.next   = pending_;
        pending_ = &b;
        try
        {
            gpu::device_guard guard(device_);
            record_events(b);
            process_pending(false);
            trim(limit_);
        }
        catch (...)
        {
            b.quarantined = true;
            ++stats_.num_errors;
        }
        // quarantined is monotonic (only ever set, never cleared), so this
        // reports whether *this* allocation failed -- not whether some
        // unrelated block already sitting in pending_ also failed its own
        // completion check during the process_pending() sweep just above.
        // The previous stats_.num_errors before/after comparison conflated
        // the two: deallocate(ptr) could report false for a ptr that itself
        // transitioned cleanly, only because another pending block's event
        // query happened to fail in the same sweep.
        return !b.quarantined;
    }

    void record_stream(const void* ptr, stream_type stream)
    {
        if (!ptr)
            return;
        std::scoped_lock  lock(mutex_);
        gpu::device_guard guard(device_);
        use_stream(live_block(ptr), stream);
    }

    void copy(
        void*       destination,
        const void* source,
        std::size_t bytes,
        stream_type stream,
        bool        to_device)
    {
        if (bytes == 0)
            return;
        if (!destination || !source)
            throw std::invalid_argument("Null transfer endpoint");
        std::scoped_lock  lock(mutex_);
        gpu::device_guard guard(device_);
        auto&             b = live_block(to_device ? source : destination, bytes);
        use_stream(b,
                   stream);  // No untracked transfer if metadata allocation fails.
        gpu::throw_on_cuda_error(
            cudaMemcpyAsync(
                destination,
                source,
                bytes,
                to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
                stream),
            "Pinned cudaMemcpyAsync");
    }

    void collect(bool wait, std::size_t limit)
    {
        gpu::device_guard guard(device_);
        process_pending(wait);
        trim(limit);
    }
#else
    void* allocate(std::size_t bytes)
    {
        if (bytes == 0)
            return nullptr;
        throw std::runtime_error("Pinned host allocation requires the CUDA or HIP backend");
    }
    bool deallocate(void* ptr) noexcept { return ptr == nullptr; }
    void record_stream(const void* ptr, stream_type)
    {
        if (ptr)
            throw std::runtime_error("Pinned host streams require CUDA or HIP");
    }
    void copy(void*, const void*, std::size_t bytes, stream_type, bool)
    {
        if (bytes)
            throw std::runtime_error("Pinned host transfers require CUDA or HIP");
    }
    void collect(bool, std::size_t) {}
#endif
};

pinned_memory_allocator::pinned_memory_allocator(int device, std::size_t max_cached_bytes)
    : impl_(std::make_unique<Impl>(device, max_cached_bytes))
{
}
pinned_memory_allocator::~pinned_memory_allocator() = default;
void* pinned_memory_allocator::allocate(std::size_t bytes)
{
    return impl_->allocate(bytes);
}
bool pinned_memory_allocator::deallocate(void* ptr) noexcept
{
    return impl_->deallocate(ptr);
}
void pinned_memory_allocator::record_stream(const void* ptr, stream_type stream)
{
    impl_->record_stream(ptr, stream);
}
void pinned_memory_allocator::copy_to_device_async(
    void* destination, const void* source, std::size_t bytes, stream_type stream)
{
    impl_->copy(destination, source, bytes, stream, true);
}
void pinned_memory_allocator::copy_from_device_async(
    void* destination, const void* source, std::size_t bytes, stream_type stream)
{
    impl_->copy(destination, source, bytes, stream, false);
}
void pinned_memory_allocator::poll()
{
    std::scoped_lock lock(impl_->mutex_);
    impl_->collect(false, impl_->limit_);
}
void pinned_memory_allocator::empty_cache()
{
    std::scoped_lock lock(impl_->mutex_);
    impl_->collect(true, 0);
}
void pinned_memory_allocator::set_max_cached_bytes(std::size_t bytes)
{
    std::scoped_lock lock(impl_->mutex_);
    impl_->limit_ = bytes;
    impl_->collect(false, bytes);
}
std::size_t pinned_memory_allocator::max_cached_bytes() const
{
    std::scoped_lock lock(impl_->mutex_);
    return impl_->limit_;
}
pinned_memory_stats pinned_memory_allocator::stats() const
{
    std::scoped_lock lock(impl_->mutex_);
    return impl_->stats_;
}
int pinned_memory_allocator::device() const noexcept
{
    return impl_->device_;
}
pinned_memory_allocator& pinned_allocator_for_device(int device)
{
    static std::mutex                                              mutex;
    static std::map<int, std::unique_ptr<pinned_memory_allocator>> allocators;
    std::scoped_lock                                               lock(mutex);
    auto&                                                          allocator = allocators[device];
    if (!allocator)
        allocator = std::make_unique<pinned_memory_allocator>(device);
    return *allocator;
}
}  // namespace memory::cpu
