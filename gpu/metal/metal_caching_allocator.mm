/*
 * Quarisma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of Quarisma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#include "gpu/metal/metal_caching_allocator.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/memory_containers.h"
#include "common/memory_macros.h"
#include "util/memory_exception.h"

namespace memory
{
namespace gpu
{
namespace
{

// Size constants mirroring cuda_caching_allocator / PyTorch CUDACachingAllocator.
constexpr size_t kMinBlockSize  = 512;
constexpr size_t kSmallSize     = 1048576;
constexpr size_t kSmallBuffer   = 2097152;
constexpr size_t kMinLargeAlloc = 10485760;
constexpr size_t kLargeBuffer   = 20971520;
constexpr size_t kRoundLarge    = 2097152;

size_t round_request_size(size_t size)
{
    if (size < kMinBlockSize)
    {
        return kMinBlockSize;
    }
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
}

size_t segment_size_for(size_t size)
{
    if (size <= kSmallSize)
    {
        return kSmallBuffer;
    }
    if (size < kMinLargeAlloc)
    {
        return kLargeBuffer;
    }
    return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
}

id<MTLDevice> system_metal_device()
{
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}

struct block_pool;

// Subrange of one shared-storage MTLBuffer segment. Split/coalesce via prev/next;
// each block retains the segment buffer under ARC (released when the last block
// referencing it is deleted).
struct cache_block
{
    cache_block(void* p, size_t s, void* st, block_pool* pl, id<MTLBuffer> buf)
        : ptr(p), size(s), stream(st), pool(pl), buffer(buf)
    {
    }

    bool is_split() const { return prev != nullptr || next != nullptr; }

    size_t byte_offset() const
    {
        if (buffer == nil || ptr == nullptr)
        {
            return 0;
        }
        return static_cast<size_t>(
            static_cast<char*>(ptr) - static_cast<char*>([buffer contents]));
    }

    void*        ptr;
    size_t       size;
    size_t       requested_size{0};
    void*        stream;
    block_pool*  pool;
    id<MTLBuffer> buffer;
    bool         allocated{false};
    cache_block* prev{nullptr};
    cache_block* next{nullptr};
    int64_t      registration_counter{-1};
};

struct cache_block_comparator
{
    bool operator()(const cache_block* a, const cache_block* b) const
    {
        if (a->stream != b->stream)
        {
            return reinterpret_cast<uintptr_t>(a->stream) < reinterpret_cast<uintptr_t>(b->stream);
        }
        if (a->size != b->size)
        {
            return a->size < b->size;
        }
        if (a->registration_counter != b->registration_counter)
        {
            return a->registration_counter < b->registration_counter;
        }
        return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
    }
};

struct block_pool
{
    explicit block_pool(bool small) : is_small(small) {}

    std::set<cache_block*, cache_block_comparator> blocks;
    const bool                                     is_small;
};

}  // namespace

struct metal_caching_allocator::Impl
{
    Impl(int device, size_t max_cached_bytes) : device_(device), max_cached_bytes_(max_cached_bytes)
    {
        MEMORY_CHECK(device == 0, "Metal caching allocator only supports device index 0");
        MEMORY_CHECK(
            system_metal_device() != nil,
            "Metal caching allocator requires a Metal-capable device");
    }

    ~Impl()
    {
        std::scoped_lock const lock(mutex_);
        release_all_blocks_noexcept();
    }

    void* allocate(size_t size, void* stream)
    {
        MEMORY_CHECK(size > 0, "metal_caching_allocator cannot allocate zero bytes");
        (void)stream;  // v1: single default-stream pool

        std::scoped_lock const lock(mutex_);

        size_t const rounded    = round_request_size(size);
        block_pool&  pool       = rounded <= kSmallSize ? small_blocks_ : large_blocks_;
        size_t const alloc_size = segment_size_for(rounded);
        void* const  pool_stream = nullptr;

        cache_block* block = get_free_block_locked(pool, pool_stream, rounded);
        if (block == nullptr && trigger_free_memory_callbacks_locked())
        {
            block = get_free_block_locked(pool, pool_stream, rounded);
        }
        if (block != nullptr)
        {
            stats_.cache_hits++;
        }
        else
        {
            stats_.cache_misses++;
            block = alloc_segment_locked(pool, pool_stream, alloc_size, false);
            if (block == nullptr)
            {
                release_cached_blocks_locked();
                block = alloc_segment_locked(pool, pool_stream, alloc_size, true);
                if (block == nullptr)
                {
                    stats_.num_ooms++;
                    throw std::bad_alloc();
                }
            }
        }

        void* ptr = alloc_found_block_locked(block, rounded, size);
        trim_cache_locked();
        return ptr;
    }

    void deallocate(void* ptr, size_t /*size*/, void* /*stream*/)
    {
        if (ptr == nullptr)
        {
            return;
        }

        std::scoped_lock const lock(mutex_);

        auto it = allocated_blocks_.find(ptr);
        MEMORY_CHECK(
            it != allocated_blocks_.end(),
            "metal_caching_allocator does not own the provided pointer");

        cache_block* block = it->second;
        MEMORY_CHECK(block->allocated, "metal_caching_allocator detected a double free");

        allocated_blocks_.erase(it);
        block->allocated = false;
        stats_.successful_frees++;
        stats_.bytes_allocated -= block->size;

        free_block_locked(block);
        trim_cache_locked();
    }

    void record_stream(void* /*ptr*/, void* /*stream*/)
    {
        // No-op: Metal dispatch is synchronous (waitUntilCompleted); no deferred reuse.
    }

    void add_free_memory_callback(metal_caching_allocator::free_memory_callback callback)
    {
        std::scoped_lock const lock(mutex_);
        free_memory_callbacks_.push_back(std::move(callback));
    }

    void clear_free_memory_callbacks()
    {
        std::scoped_lock const lock(mutex_);
        free_memory_callbacks_.clear();
    }

    void empty_cache()
    {
        std::scoped_lock const lock(mutex_);
        release_cached_blocks_locked();
    }

    void set_max_cached_bytes(size_t bytes)
    {
        std::scoped_lock const lock(mutex_);
        max_cached_bytes_ = bytes;
        trim_cache_locked();
    }

    size_t max_cached_bytes() const
    {
        std::scoped_lock const lock(mutex_);
        return max_cached_bytes_;
    }

    unified_cache_stats stats() const
    {
        std::scoped_lock const lock(mutex_);
        unified_cache_stats    copy(stats_);
        copy.bytes_cached.store(bytes_cached_, std::memory_order_relaxed);
        copy.peak_bytes_cached.store(peak_bytes_cached_, std::memory_order_relaxed);
        copy.cache_blocks.store(
            small_blocks_.blocks.size() + large_blocks_.blocks.size(), std::memory_order_relaxed);
        size_t split_bytes = 0;
        for (const block_pool* pool : {&small_blocks_, &large_blocks_})
        {
            for (const cache_block* block : pool->blocks)
            {
                if (block->is_split())
                {
                    split_bytes += block->size;
                }
            }
        }
        copy.inactive_split_bytes.store(split_bytes, std::memory_order_relaxed);
        return copy;
    }

    int device() const { return device_; }

    bool resolve_live_allocation(void* ptr, void** handle_out, size_t* offset_out) const
    {
        if (ptr == nullptr || handle_out == nullptr || offset_out == nullptr)
        {
            return false;
        }

        std::scoped_lock const lock(mutex_);
        auto                   it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end())
        {
            return false;
        }
        cache_block* block = it->second;
        if (block->buffer == nil)
        {
            return false;
        }
        *handle_out = (__bridge void*)block->buffer;
        *offset_out = block->byte_offset();
        return true;
    }

private:
    bool trigger_free_memory_callbacks_locked()
    {
        bool freed_memory = false;
        for (const auto& callback : free_memory_callbacks_)
        {
            freed_memory |= callback();
        }
        return freed_memory;
    }

    cache_block* get_free_block_locked(block_pool& pool, void* stream, size_t size)
    {
        cache_block key(nullptr, size, stream, &pool, nil);
        auto        it = pool.blocks.lower_bound(&key);
        if (it == pool.blocks.end() || (*it)->stream != stream)
        {
            return nullptr;
        }
        cache_block* block = *it;
        pool.blocks.erase(it);
        bytes_cached_ -= block->size;
        return block;
    }

    cache_block* alloc_segment_locked(block_pool& pool, void* stream, size_t alloc_size, bool is_retry)
    {
        if (is_retry)
        {
            stats_.num_alloc_retries++;
        }

        id<MTLDevice> dev = system_metal_device();
        if (dev == nil)
        {
            return nullptr;
        }

        // Allocate metadata before the driver call so a throwing new cannot leak
        // a successfully created MTLBuffer.
        auto block = std::make_unique<cache_block>(nullptr, alloc_size, stream, &pool, nil);

        id<MTLBuffer> buffer =
            [dev newBufferWithLength:alloc_size options:MTLResourceStorageModeShared];
        if (buffer == nil)
        {
            return nullptr;
        }

        block->buffer = buffer;
        block->ptr    = [buffer contents];
        block->registration_counter =
            registration_counter_global_.fetch_add(1, std::memory_order_relaxed) + 1;
        stats_.driver_allocations++;
        stats_.bytes_reserved += alloc_size;
        return block.release();
    }

    static bool should_split(const cache_block* block, size_t size)
    {
        size_t const remaining = block->size - size;
        if (block->pool->is_small)
        {
            return remaining >= kMinBlockSize;
        }
        return remaining > kSmallSize;
    }

    void* alloc_found_block_locked(cache_block* block, size_t rounded, size_t orig_size)
    {
        if (should_split(block, rounded))
        {
            cache_block* remaining = block;
            block = new cache_block(
                remaining->ptr, rounded, remaining->stream, remaining->pool, remaining->buffer);
            block->registration_counter = remaining->registration_counter;
            block->prev                 = remaining->prev;
            if (block->prev != nullptr)
            {
                block->prev->next = block;
            }
            block->next     = remaining;
            remaining->prev = block;
            remaining->ptr  = static_cast<char*>(remaining->ptr) + rounded;
            remaining->size -= rounded;
            remaining->pool->blocks.insert(remaining);
            bytes_cached_ += remaining->size;
        }

        block->allocated      = true;
        block->requested_size = orig_size;
        allocated_blocks_.emplace(block->ptr, block);
        stats_.successful_allocations++;
        stats_.bytes_allocated += block->size;
        return block->ptr;
    }

    void free_block_locked(cache_block* block)
    {
        size_t const freed_size = block->size;
        try_merge_locked(block, block->prev);
        try_merge_locked(block, block->next);

        block->pool->blocks.insert(block);
        bytes_cached_ += freed_size;
        peak_bytes_cached_ = std::max(peak_bytes_cached_, bytes_cached_);
    }

    void try_merge_locked(cache_block* dst, cache_block* src)
    {
        if (src == nullptr || src->allocated)
        {
            return;
        }
        if (dst->prev == src)  // [src dst]
        {
            dst->ptr  = src->ptr;
            dst->prev = src->prev;
            if (dst->prev != nullptr)
            {
                dst->prev->next = dst;
            }
        }
        else  // [dst src]
        {
            dst->next = src->next;
            if (dst->next != nullptr)
            {
                dst->next->prev = dst;
            }
        }
        dst->size += src->size;
        dst->pool->blocks.erase(src);
        delete src;
    }

    void release_segment_locked(cache_block* block)
    {
        // Only whole segments (never split) can be returned to Metal.
        stats_.driver_frees++;
        stats_.cache_evictions++;
        stats_.bytes_reserved -= block->size;
        block->buffer = nil;  // ARC releases the MTLBuffer
        delete block;
    }

    void release_pool_blocks_locked(block_pool& pool)
    {
        auto it = pool.blocks.begin();
        while (it != pool.blocks.end())
        {
            cache_block* block = *it;
            ++it;
            if (!block->is_split())
            {
                bytes_cached_ -= block->size;
                pool.blocks.erase(block);
                release_segment_locked(block);
            }
        }
    }

    void release_cached_blocks_locked()
    {
        release_pool_blocks_locked(small_blocks_);
        release_pool_blocks_locked(large_blocks_);
    }

    void trim_cache_locked()
    {
        if (max_cached_bytes_ == std::numeric_limits<size_t>::max())
        {
            return;
        }
        while (bytes_cached_ > max_cached_bytes_)
        {
            cache_block* victim = nullptr;
            for (block_pool* pool : {&small_blocks_, &large_blocks_})
            {
                for (cache_block* block : pool->blocks)
                {
                    if (!block->is_split() && (victim == nullptr || block->size > victim->size))
                    {
                        victim = block;
                    }
                }
            }
            if (victim == nullptr)
            {
                break;
            }
            bytes_cached_ -= victim->size;
            victim->pool->blocks.erase(victim);
            release_segment_locked(victim);
        }
    }

    void release_all_blocks_noexcept()
    {
        std::set<cache_block*> all_blocks;
        auto                   collect = [&](cache_block* block)
        {
            all_blocks.insert(block);
            cache_block* head = block;
            while (head->prev != nullptr)
            {
                head = head->prev;
            }
            cache_block* cur = head;
            while (cur != nullptr)
            {
                all_blocks.insert(cur);
                cur = cur->next;
            }
        };

        for (block_pool* pool : {&small_blocks_, &large_blocks_})
        {
            for (cache_block* block : pool->blocks)
            {
                collect(block);
            }
            pool->blocks.clear();
        }
        for (auto& entry : allocated_blocks_)
        {
            collect(entry.second);
        }
        allocated_blocks_.clear();

        for (cache_block* block : all_blocks)
        {
            block->buffer = nil;
            delete block;
        }
        bytes_cached_      = 0;
        peak_bytes_cached_ = 0;
    }

    int    device_;
    size_t max_cached_bytes_;
    size_t bytes_cached_{0};
    size_t peak_bytes_cached_{0};

    mutable std::recursive_mutex mutex_;
    block_pool                   small_blocks_{true};
    block_pool                   large_blocks_{false};
    memory_map<void*, cache_block*> allocated_blocks_;
    std::vector<metal_caching_allocator::free_memory_callback> free_memory_callbacks_;
    std::atomic<int64_t>                                       registration_counter_global_{0};
    unified_cache_stats                                        stats_;
};

metal_caching_allocator::metal_caching_allocator(int device, size_t max_cached_bytes)
    : impl_(std::make_unique<Impl>(device, max_cached_bytes))
{
}

metal_caching_allocator::~metal_caching_allocator() = default;

metal_caching_allocator::metal_caching_allocator(metal_caching_allocator&&) noexcept = default;

metal_caching_allocator& metal_caching_allocator::operator=(metal_caching_allocator&&) noexcept =
    default;

void* metal_caching_allocator::allocate(size_t size, stream_type stream)
{
    if MEMORY_UNLIKELY (size == 0)
    {
        return nullptr;
    }
    return impl_->allocate(size, stream);
}

void metal_caching_allocator::deallocate(void* ptr, size_t size, stream_type stream)
{
    impl_->deallocate(ptr, size, stream);
}

void metal_caching_allocator::record_stream(void* ptr, stream_type stream)
{
    impl_->record_stream(ptr, stream);
}

void metal_caching_allocator::add_free_memory_callback(free_memory_callback callback)
{
    impl_->add_free_memory_callback(std::move(callback));
}

void metal_caching_allocator::clear_free_memory_callbacks()
{
    impl_->clear_free_memory_callbacks();
}

void metal_caching_allocator::empty_cache()
{
    impl_->empty_cache();
}

void metal_caching_allocator::set_max_cached_bytes(size_t bytes)
{
    impl_->set_max_cached_bytes(bytes);
}

size_t metal_caching_allocator::max_cached_bytes() const
{
    return impl_->max_cached_bytes();
}

unified_cache_stats metal_caching_allocator::stats() const
{
    return impl_->stats();
}

int metal_caching_allocator::device() const
{
    return impl_->device();
}

bool metal_caching_allocator::resolve_live_allocation(
    void* ptr, void** handle_out, size_t* offset_out) const
{
    return impl_->resolve_live_allocation(ptr, handle_out, offset_out);
}

metal_caching_allocator& metal_caching_allocator_for_device(int device_index)
{
    static std::mutex                                                        registry_mutex;
    static std::unordered_map<int, std::unique_ptr<metal_caching_allocator>> registry;

    std::scoped_lock const lock(registry_mutex);
    auto&                  entry = registry[device_index];
    if (entry == nullptr)
    {
        entry = std::make_unique<metal_caching_allocator>(device_index);
    }
    return *entry;
}

}  // namespace gpu
}  // namespace memory
