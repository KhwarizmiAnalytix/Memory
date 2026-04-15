/*
 * Quarisma: High-Performance Computational Library
 *
 * Original work Copyright 2015 The TensorFlow Authors
 * Modified work Copyright 2025 Quarisma Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file contains code modified from TensorFlow (Apache 2.0 licensed)
 * and is part of Quarisma, licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * MODIFICATIONS FROM ORIGINAL:
 * - Adapted for Quarisma quantitative computing requirements
 * - Added high-performance memory allocation optimizations
 * - Integrated NUMA-aware allocation strategies
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#include "backend/allocator_pool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "cpu/allocator.h"

#if !defined(_WIN32) && !defined(_MSC_VER)
#include <strings.h>
#include <sys/mman.h>  // for munmap
#endif

#include <map>
#include <utility>

//#include "logger/logger.h"
#include "helper/memory_allocator.h"
#if MEMORY_HAS_NATIVE_PROFILER && 0
#include "native/tracing/traceme.h"
#endif
#include "util/exception.h"

namespace memory
{
allocator_pool::allocator_pool(
    size_t                                 pool_size_limit,
    bool                                   auto_resize,
    std::unique_ptr<memory::sub_allocator> allocator,
    std::unique_ptr<round_up_interface>    size_rounder,
    std::string                            name)
    : name_(std::move(name)),
      has_size_limit_(pool_size_limit > 0),
      auto_resize_(auto_resize),
      pool_size_limit_(pool_size_limit),
      allocator_(std::move(allocator)),
      size_rounder_(std::move(size_rounder))
{
    if (auto_resize)
    {
        MEMORY_CHECK(pool_size_limit > 0, "size limit must be > 0 if auto_resize is true.");
    }
}

allocator_pool::~allocator_pool()
{
    Clear();
}

namespace
{
// Pools contain Chunks allocated from the underlying Allocator.
// Chunk alignment is always on kPoolAlignment boundaries.  Each Chunk
// begins with a descriptor (ChunkPrefix) that gives its size and a
// pointer to itself.  The pointer returned to the user is just past
// the ChunkPrefix.  If the user asks for a larger alignment, we will
// increase the size of the chunk, then adjust the returned user
// pointer and also re-write the ChunkPrefix.chunk_ptr value
// immediately before it.  This way the Chunk address and size can be
// recovered from the returned user pointer, regardless of alignment.
// Note that this dereferencing of the pointers means that we cannot
// handle GPU memory, only CPU memory.
struct ChunkPrefix
{
    size_t num_bytes;
    void*  chunk_ptr;
};

// kPoolAlignment cannot be less than the size of ChunkPrefix.
const int kPoolAlignment = sizeof(ChunkPrefix);

void* PrepareChunk(void* chunk, size_t alignment, size_t num_bytes)
{
    if (chunk == nullptr)
    {
        return nullptr;
    }
    auto* cp       = reinterpret_cast<ChunkPrefix*>(chunk);
    cp->num_bytes  = num_bytes;
    cp->chunk_ptr  = chunk;
    void* user_ptr = reinterpret_cast<void*>(cp + 1);
    if (alignment > kPoolAlignment)
    {
        // Move user_ptr forward to the first satisfying offset, and write
        // chunk_ptr just before it.
        size_t const aligned_ptr = reinterpret_cast<size_t>(user_ptr) + alignment;
        user_ptr                 = reinterpret_cast<void*>(aligned_ptr & ~(alignment - 1));
        (reinterpret_cast<ChunkPrefix*>(user_ptr) - 1)->chunk_ptr = chunk;
    }
    // Safety check that user_ptr is always past the ChunkPrefix.
    MEMORY_CHECK(user_ptr >= reinterpret_cast<ChunkPrefix*>(chunk) + 1);
    return user_ptr;
}

ChunkPrefix* FindPrefix(void* user_ptr)
{
    ChunkPrefix const* cp = reinterpret_cast<ChunkPrefix*>(user_ptr) - 1;
    return reinterpret_cast<ChunkPrefix*>(cp->chunk_ptr);
}
}  // namespace

void* allocator_pool::allocate_raw(size_t alignment, size_t num_bytes)
{
    MEMORY_CHECK(
        static_cast<std::ptrdiff_t>(num_bytes) > 0, "Cannot allocate {} bytes.", num_bytes);

    // If alignment is larger than kPoolAlignment, increase num_bytes so that we
    // are guaranteed to be able to return an aligned ptr by advancing user_ptr
    // without overrunning the end of the chunk.
    if (alignment > kPoolAlignment)
    {
        num_bytes += alignment;
    }
    num_bytes += sizeof(ChunkPrefix);
    num_bytes     = size_rounder_->RoundUp(num_bytes);
    PtrRecord* pr = nullptr;
    if (has_size_limit_)
    {
        {
            std::scoped_lock const lock(mutex_);
            auto                   iter = pool_.find(num_bytes);
            if (iter == pool_.end())
            {
                allocated_count_++;
                // Deliberately fall out of lock scope before
                // calling the allocator.  No further modification
                // to the pool will be performed.
            }
            else
            {
                get_from_pool_count_++;
                pr = iter->second;
                RemoveFromList(pr);
                pool_.erase(iter);
                // Fall out of lock scope and do the result without the lock held.
            }
        }
    }
    if (pr != nullptr)
    {
        void* r = pr->ptr;
        delete pr;
        return PrepareChunk(r, alignment, num_bytes);
    }

    size_t bytes_received;
    void*  ptr = allocator_->Alloc(kPoolAlignment, num_bytes, &bytes_received);
    return PrepareChunk(ptr, alignment, bytes_received);
}

void allocator_pool::deallocate_raw(void* ptr)
{
    if (ptr == nullptr)
    {
        return;
    }
    ChunkPrefix* cp = FindPrefix(ptr);
    MEMORY_CHECK(static_cast<void*>(cp) <= static_cast<void*>(ptr));
    if (!has_size_limit_ && !auto_resize_)
    {
        allocator_->Free(cp, cp->num_bytes);
    }
    else
    {
        std::scoped_lock const lock(mutex_);

        // Check for double-free: search if this pointer is already in the pool
        if (std::any_of(
                pool_.begin(),
                pool_.end(),
                [cp](const auto& iter) { return iter.second->ptr == cp; }))
        {
            // Double-free detected, handle gracefully by ignoring
            return;
        }

        ++put_count_;
        while (pool_.size() >= pool_size_limit_)
        {
            EvictOne();
        }
        auto* pr      = new PtrRecord;
        pr->num_bytes = cp->num_bytes;
        pr->ptr       = cp;
        AddToList(pr);
        pool_.insert(std::make_pair(cp->num_bytes, pr));
    }
}

void allocator_pool::Clear()
{
    if (has_size_limit_)
    {
        std::scoped_lock const lock(mutex_);
        for (auto iter : pool_)
        {
            PtrRecord const* pr = iter.second;
            allocator_->Free(pr->ptr, pr->num_bytes);
            delete pr;
        }
        pool_.clear();
        get_from_pool_count_ = 0;
        put_count_           = 0;
        allocated_count_     = 0;
        evicted_count_       = 0;
        lru_head_            = nullptr;
        lru_tail_            = nullptr;
    }
}

void allocator_pool::RemoveFromList(PtrRecord* ptr)
{
    if (ptr->prev == nullptr)
    {
        MEMORY_CHECK(lru_head_ == ptr);
        lru_head_ = ptr->next;
    }
    else
    {
        ptr->prev->next = ptr->next;
    }
    if (ptr->next == nullptr)
    {
        MEMORY_CHECK(lru_tail_ == ptr);
        lru_tail_ = ptr->prev;
    }
    else
    {
        ptr->next->prev = ptr->prev;
    }
}

void allocator_pool::AddToList(PtrRecord* pr)
{
    pr->prev = nullptr;
    if (lru_head_ == nullptr)
    {
        MEMORY_CHECK(lru_tail_ == nullptr);
        lru_tail_ = pr;
        pr->next  = nullptr;
    }
    else
    {
        pr->next       = lru_head_;
        pr->next->prev = pr;
    }
    lru_head_ = pr;
}

void allocator_pool::EvictOne()
{
    MEMORY_CHECK(lru_tail_ != nullptr);
    PtrRecord* prec = lru_tail_;
    RemoveFromList(prec);
    auto range = pool_.equal_range(prec->num_bytes);
    auto iter =
        std::find_if(range.first, range.second, [prec](const auto& p) { return p.second == prec; });
    MEMORY_CHECK(iter != range.second);
    pool_.erase(iter);
    allocator_->Free(prec->ptr, prec->num_bytes);
    delete prec;
    ++evicted_count_;
    // Auto-resizing, and warning messages.
    static const double kTolerable      = 2e-3;
    static const int    kCheckInterval  = 1000;
    static const double kIncreaseFactor = 1.1;
    static const int    kMinPoolSize    = 100;
    if (0 == evicted_count_ % kCheckInterval)
    {
        const double  eviction_rate       = evicted_count_ / static_cast<double>(put_count_);
        const int64_t alloc_request_count = allocated_count_ + get_from_pool_count_;
        const double  alloc_rate          = (alloc_request_count == 0)
                                                ? 0.0
                                                : allocated_count_ / static_cast<double>(alloc_request_count);
        // Can turn on for debugging purposes.
        const bool kShouldLog = false;
        if (kShouldLog)
        {
            MEMORY_LOG_INFO(
                "allocator_pool: After {} get requests, put_count={} evicted_count={} "
                "eviction_rate={} and unsatisfied allocation rate={}",
                alloc_request_count,
                put_count_,
                evicted_count_,
                eviction_rate,
                alloc_rate);
        }
        if (auto_resize_ && (eviction_rate > kTolerable) && (alloc_rate > kTolerable))
        {
            size_t new_size_limit = (pool_size_limit_ < kMinPoolSize)
                                        ? kMinPoolSize
                                        : (kIncreaseFactor * pool_size_limit_);
            if (kShouldLog)
            {
                MEMORY_LOG_INFO(
                    "Raising pool_size_limit_ from {} to {}", pool_size_limit_, new_size_limit);
            }
            pool_size_limit_ = new_size_limit;
            // Reset all the counters so that ratios are relative to new sizes
            // at next test interval.
            put_count_           = 0;
            allocated_count_     = 0;
            evicted_count_       = 0;
            get_from_pool_count_ = 0;
        }
    }
}

basic_cpu_allocator::~basic_cpu_allocator() = default;

void* basic_cpu_allocator::Alloc(size_t alignment, size_t num_bytes, size_t* bytes_received)
{
#if MEMORY_HAS_NATIVE_PROFILER && 0
    memory::traceme const traceme("basic_cpu_allocator::Alloc");
#endif

    void* ptr       = nullptr;
    *bytes_received = num_bytes;
    if (num_bytes > 0)
    {
        ptr = memory::cpu::memory_allocator::allocate(num_bytes, static_cast<int>(alignment));

        VisitAlloc(ptr, numa_node_, num_bytes);
    }
    return ptr;
}

void basic_cpu_allocator::Free(void* ptr, size_t num_bytes)
{
#if MEMORY_HAS_NATIVE_PROFILER && 0
    memory::traceme const traceme("basic_cpu_allocator::Free");
#endif

    if (num_bytes > 0)
    {
        VisitFree(ptr, numa_node_, num_bytes);
        memory::cpu::memory_allocator::free(ptr);
    }
}
}  // namespace memory
