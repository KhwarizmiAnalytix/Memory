#include "gpu/cuda_caching_allocator.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/memory_containers.h"
#include "common/memory_macros.h"
#include "gpu/caching_allocator_config.h"
#include "util/memory_exception.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/gpu_runtime.h"
#endif

#if MEMORY_HAS_PROFILER
#include "common/instrumentation.h"
#include "gpu/caching_allocator_profiler_report.h"
#endif

namespace memory
{
namespace gpu
{
namespace
{

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
inline void throw_on_cuda_error(cudaError_t result, const char* what)
{
    if (result != cudaSuccess)
    {
        std::string const message = std::string(what) + ": " + cudaGetErrorString(result);
        // Log error (simplified for build compatibility)
        throw std::runtime_error(message);
    }
}
#else
inline void throw_on_cuda_error(int result, const char* what)
{
    if (result != 0)
    {
        std::string message = std::string(what) + ": CUDA/HIP not available";
        // Log error (simplified for build compatibility)
        throw std::runtime_error(message);
    }
}
#endif

class DeviceGuard
{
public:
    explicit DeviceGuard(int device)
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        int current = 0;
        throw_on_cuda_error(cudaGetDevice(&current), "cudaGetDevice");
        prev_ = current;
        if (current != device)
        {
            throw_on_cuda_error(cudaSetDevice(device), "cudaSetDevice");
            changed_ = true;
        }
#else
        (void)device;
#endif
    }

    DeviceGuard(const DeviceGuard&)            = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    ~DeviceGuard()
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (changed_)
        {
            cudaSetDevice(prev_);
        }
#endif
    }

private:
    int  prev_{0};
    bool changed_{false};
};

}  // namespace

// cuda_caching_allocator is the CUDA/HIP caching layer (PyTorch-style segmented
// caching with per-stream pools, block split/merge, and event-deferred cross-stream
// reclamation). Metal uses metal_caching_allocator instead.
// Callers reach it through caching_allocator_for_device() (the process-wide
// per-device registry backing allocator<T>'s CUDA/HIP path), or via the
// cuda_caching_allocator_template<T> wrapper. The #else stub below exists purely so
// this translation unit still compiles in Metal builds; constructing the allocator
// there throws at runtime.
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
namespace
{

using caching_config::kMinBlockSize;
using caching_config::kSmallSize;
using caching_config::kSmallBuffer;
using caching_config::round_request_size;
using caching_config::segment_size_for;

struct block_pool;

// A cache_block is a subrange of a segment (one cudaMalloc). Blocks are split on
// reuse and coalesced on free via the intrusive prev/next links; metadata is
// raw-allocated because ownership transfers between the free pools, the active
// map, and merge operations, mirroring the upstream implementation.
struct cache_block
{
    cache_block(void* p, size_t s, cudaStream_t st, block_pool* pl)
        : ptr(p), size(s), stream(st), pool(pl)
    {
    }

    bool is_split() const { return prev != nullptr || next != nullptr; }

    void*                  ptr;
    size_t                 size;
    size_t                 requested_size{0};
    cudaStream_t           stream;
    block_pool*            pool;
    bool                   allocated{false};
    cache_block*           prev{nullptr};
    cache_block*           next{nullptr};
    int                    event_count{0};
    std::set<cudaStream_t> stream_uses;
    // Segment creation order; equal-size free blocks recycle FIFO (upstream
    // registration_counter). Search keys keep the -1 default so lower_bound
    // finds the oldest matching block.
    int64_t registration_counter{-1};
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

// Free blocks of one size class, ordered by (stream, size, ptr): blocks are only
// ever reused on the stream they were allocated on.
struct block_pool
{
    explicit block_pool(bool small) : is_small(small) {}

    std::set<cache_block*, cache_block_comparator> blocks;
    const bool                                     is_small;
};

}  // namespace

struct cuda_caching_allocator::Impl
{
    Impl(int device, size_t max_cached_bytes) : device_(device), max_cached_bytes_(max_cached_bytes)
    {
        // Validate device
        int device_count = 0;
        throw_on_cuda_error(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        MEMORY_CHECK(  //NOLINT
            device >= 0 && device < device_count,
            "Invalid CUDA device index: {} (available: 0-{})",
            device,
            device_count - 1);
    }

    ~Impl()
    {
        std::scoped_lock const lock(mutex_);
        release_all_blocks_noexcept();
    }

    void* allocate(size_t size, cudaStream_t stream)
    {
        MEMORY_CHECK(size > 0, "cuda_caching_allocator cannot allocate zero bytes");

        std::scoped_lock const lock(mutex_);
        process_events_locked();

        size_t const rounded    = round_request_size(size);
        block_pool&  pool       = rounded <= kSmallSize ? small_blocks_ : large_blocks_;
        size_t const alloc_size = segment_size_for(rounded);

        cache_block* block = get_free_block_locked(pool, stream, rounded);
        if (block == nullptr && trigger_free_memory_callbacks_locked())
        {
            // A callback freed device memory; retry the cache before the driver,
            // matching the upstream retry chain.
            block = get_free_block_locked(pool, stream, rounded);
        }
        if (block != nullptr)
        {
            stats_.cache_hits++;
        }
        else
        {
            stats_.cache_misses++;
            block = alloc_segment_locked(pool, stream, alloc_size, false);
            if (block == nullptr)
            {
                // OOM chain: flush the entire cache (synchronize pending events and
                // release every releasable cached segment) and retry once before
                // failing, matching the upstream retry behavior.
                release_cached_blocks_locked();
                block = alloc_segment_locked(pool, stream, alloc_size, true);
                if (block == nullptr)
                {
                    stats_.num_ooms++;
                    throw std::bad_alloc();
                }
            }
        }

        void* ptr = alloc_found_block_locked(block, rounded, size);
        // process_events_locked above may have grown the cache past the cap
        trim_cache_locked();
        return ptr;
    }

    void deallocate(void* ptr, size_t /*size*/, cudaStream_t stream)
    {
        if (ptr == nullptr)
        {
            return;
        }

        std::scoped_lock const lock(mutex_);
        process_events_locked();

        auto it = allocated_blocks_.find(ptr);
        MEMORY_CHECK(
            it != allocated_blocks_.end(),
            "cuda_caching_allocator does not own the provided pointer");

        cache_block* block = it->second;
        MEMORY_CHECK(block->allocated, "cuda_caching_allocator detected a double free");

        allocated_blocks_.erase(it);
        block->allocated = false;
        stats_.successful_frees++;
        stats_.bytes_allocated -= block->size;

        // The stream hint maps to recordStream semantics: freeing after use on a
        // stream other than the allocation stream counts as a cross-stream use.
        if (stream != nullptr && stream != block->stream)
        {
            block->stream_uses.insert(stream);
        }

        if (!block->stream_uses.empty())
        {
            insert_events_locked(block);
        }
        else
        {
            free_block_locked(block);
        }

        trim_cache_locked();
    }

    void add_free_memory_callback(cuda_caching_allocator::free_memory_callback callback)
    {
        std::scoped_lock const lock(mutex_);
        free_memory_callbacks_.push_back(std::move(callback));
    }

    void clear_free_memory_callbacks()
    {
        std::scoped_lock const lock(mutex_);
        free_memory_callbacks_.clear();
    }

    void record_stream(void* ptr, cudaStream_t stream)
    {
        if (ptr == nullptr || stream == nullptr)
        {
            return;
        }

        std::scoped_lock const lock(mutex_);
        auto                   it = allocated_blocks_.find(ptr);
        MEMORY_CHECK(
            it != allocated_blocks_.end(),
            "cuda_caching_allocator::record_stream on a pointer that is not a live allocation");

        cache_block* block = it->second;
        if (stream == block->stream)
        {
            // Uses on the allocation stream need no synchronization (upstream rule)
            return;
        }
        block->stream_uses.insert(stream);
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

private:
    bool trigger_free_memory_callbacks_locked()
    {
        // All callbacks run (no short-circuit), matching upstream; each reports
        // whether it freed memory.
        bool freed_memory = false;
        for (const auto& callback : free_memory_callbacks_)
        {
            freed_memory |= callback();
        }
        return freed_memory;
    }

    cache_block* get_free_block_locked(block_pool& pool, cudaStream_t stream, size_t size)
    {
        cache_block key(nullptr, size, stream, &pool);
        auto        it = pool.blocks.lower_bound(&key);
        // Free pools are stream-scoped: a block belonging to another stream is
        // never reused (upstream get_free_block rule).
        if (it == pool.blocks.end() || (*it)->stream != stream)
        {
            return nullptr;
        }
        cache_block* block = *it;
        pool.blocks.erase(it);
        bytes_cached_ -= block->size;
        return block;
    }

    cache_block* alloc_segment_locked(
        block_pool& pool, cudaStream_t stream, size_t alloc_size, bool is_retry)
    {
        if (is_retry)
        {
            stats_.num_alloc_retries++;
        }
        // Metadata is allocated before the driver call so a throwing new cannot
        // leak a successfully cudaMalloc'd segment.
        auto              block = std::make_unique<cache_block>(nullptr, alloc_size, stream, &pool);
        DeviceGuard const guard(device_);
        void*             ptr = nullptr;
        cudaError_t const err = cudaMalloc(&ptr, alloc_size);
        if (err != cudaSuccess)
        {
            // Forgive and clear CUDA's internal error state, matching upstream;
            // only an out-of-memory error falls through to the cache-flush retry.
            (void)cudaGetLastError();
            if (err != cudaErrorMemoryAllocation)
            {
                throw_on_cuda_error(err, "cudaMalloc");
            }
            return nullptr;
        }
        block->ptr = ptr;
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
        // Upstream additionally requires the request to be below max_split_size,
        // which defaults to SIZE_MAX and is always true here.
        return remaining > kSmallSize;
    }

    void* alloc_found_block_locked(cache_block* block, size_t rounded, size_t orig_size)
    {
        if (should_split(block, rounded))
        {
            cache_block* remaining = block;
            block = new cache_block(remaining->ptr, rounded, remaining->stream, remaining->pool);
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

        // Merging only relabels sizes already counted in the pool; the net new
        // cached bytes are the freed block's own (pre-merge) size.
        block->pool->blocks.insert(block);
        bytes_cached_ += freed_size;
        peak_bytes_cached_ = std::max(peak_bytes_cached_, bytes_cached_);
    }

    void try_merge_locked(cache_block* dst, cache_block* src)
    {
        if (src == nullptr || src->allocated || src->event_count > 0 || !src->stream_uses.empty())
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

    void insert_events_locked(cache_block* block)
    {
        DeviceGuard const      guard(device_);
        std::set<cudaStream_t> streams;
        streams.swap(block->stream_uses);
        // Boundary/interop path: a CUDA error here must not orphan the block
        // between the pools and the event queues.
        try
        {
            for (cudaStream_t stream : streams)
            {
                cudaEvent_t event = acquire_event_locked();
                throw_on_cuda_error(cudaEventRecord(event, stream), "cudaEventRecord");
                cuda_events_[stream].emplace_back(event, block);
                block->event_count++;
            }
        }
        catch (...)
        {
            // Events already queued will recycle the block when they complete;
            // with nothing recorded, return it to its pool immediately.
            if (block->event_count == 0)
            {
                free_block_locked(block);
            }
            throw;
        }
    }

    cudaEvent_t acquire_event_locked()
    {
        if (!event_pool_.empty())
        {
            cudaEvent_t event = event_pool_.back();
            event_pool_.pop_back();
            return event;
        }
        cudaEvent_t event = nullptr;
        throw_on_cuda_error(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags");
        return event;
    }

    void recycle_event_locked(cudaEvent_t event) { event_pool_.push_back(event); }

    void process_events_locked()
    {
        // Events are device-local: polling must run on the owning device.
        DeviceGuard const guard(device_);

        // Per-stream queues are drained independently so one stream's long-running
        // work does not head-of-line block reclamation from other streams.
        for (auto map_it = cuda_events_.begin(); map_it != cuda_events_.end();)
        {
            auto& queue = map_it->second;
            while (!queue.empty())
            {
                cudaEvent_t        event  = queue.front().first;
                cache_block* const block  = queue.front().second;
                cudaError_t const  status = cudaEventQuery(event);
                if (status == cudaSuccess)
                {
                    recycle_event_locked(event);
                    queue.pop_front();
                    block->event_count--;
                    if (block->event_count == 0)
                    {
                        free_block_locked(block);
                    }
                }
                else if (status == cudaErrorNotReady)
                {
                    (void)cudaGetLastError();  // clear the not-ready error state
                    break;
                }
                else
                {
                    throw_on_cuda_error(status, "cudaEventQuery");
                }
            }
            if (queue.empty())
            {
                map_it = cuda_events_.erase(map_it);
            }
            else
            {
                ++map_it;
            }
        }
    }

    void synchronize_and_free_events_locked()
    {
        stats_.num_sync_all_streams++;
        DeviceGuard const guard(device_);
        for (auto map_it = cuda_events_.begin(); map_it != cuda_events_.end();)
        {
            for (auto& entry : map_it->second)
            {
                throw_on_cuda_error(cudaEventSynchronize(entry.first), "cudaEventSynchronize");
                recycle_event_locked(entry.first);
                entry.second->event_count--;
                if (entry.second->event_count == 0)
                {
                    free_block_locked(entry.second);
                }
            }
            map_it = cuda_events_.erase(map_it);
        }
    }

    void release_segment_locked(cache_block* block)
    {
        // Only whole segments (never split) can be returned to the driver.
        DeviceGuard const guard(device_);
        throw_on_cuda_error(cudaFree(block->ptr), "cudaFree");
        stats_.driver_frees++;
        stats_.cache_evictions++;
        stats_.bytes_reserved -= block->size;
        delete block;
    }

    void release_pool_blocks_locked(block_pool& pool)
    {
        auto it = pool.blocks.begin();
        while (it != pool.blocks.end())
        {
            cache_block* block = *it;
            ++it;
            // Free all non-split cached blocks, matching upstream release_blocks:
            // split remainders share a segment with live neighbors and must stay.
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
        synchronize_and_free_events_locked();
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
            // Largest-first among releasable (whole-segment) cached blocks; split
            // remainders belong to a segment with live neighbors and must stay.
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
        DeviceGuard const guard(device_);

        // A segment's base pointer is its first block; collect each segment once
        // (split blocks share their segment with neighbors) and each block once
        // (a block with pending events appears once per queued event). Ordered
        // sets keep teardown deterministic.
        std::set<void*>        segment_ptrs;
        std::set<cache_block*> all_blocks;
        auto                   collect = [&](cache_block* block)
        {
            all_blocks.insert(block);
            cache_block* head = block;
            while (head->prev != nullptr)
            {
                head = head->prev;
            }
            segment_ptrs.insert(head->ptr);
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
        for (auto& entry : cuda_events_)
        {
            for (auto& queued : entry.second)
            {
                cudaEventDestroy(queued.first);
                collect(queued.second);
            }
        }
        cuda_events_.clear();
        for (cudaEvent_t event : event_pool_)
        {
            cudaEventDestroy(event);
        }
        event_pool_.clear();

        for (void* ptr : segment_ptrs)
        {
            cudaFree(ptr);
        }
        for (cache_block* block : all_blocks)
        {
            delete block;
        }
        bytes_cached_      = 0;
        peak_bytes_cached_ = 0;
    }

    int    device_;
    size_t max_cached_bytes_;
    size_t bytes_cached_{0};
    size_t peak_bytes_cached_{0};

    // Recursive, matching upstream: free-memory callbacks run under the lock and
    // may re-enter this allocator to free memory.
    mutable std::recursive_mutex mutex_;
    block_pool                   small_blocks_{true};
    block_pool                   large_blocks_{false};
    // Live allocations by pointer; free blocks live in the pool sets and blocks
    // with outstanding cross-stream events live in the event queues.
    memory_map<void*, cache_block*> allocated_blocks_;
    std::unordered_map<cudaStream_t, std::deque<std::pair<cudaEvent_t, cache_block*>>> cuda_events_;
    std::vector<cudaEvent_t>                                                           event_pool_;
    std::vector<cuda_caching_allocator::free_memory_callback> free_memory_callbacks_;
    std::atomic<int64_t>                                      registration_counter_global_{0};
    unified_cache_stats                                       stats_;
};
#else
struct cuda_caching_allocator::Impl
{
    Impl(int device, size_t max_cached_bytes) : device_(device), max_cached_bytes_(max_cached_bytes)
    {
    }

    void* allocate(size_t, cuda_caching_allocator::stream_type)
    {
        throw std::runtime_error("cuda_caching_allocator requires MEMORY_GPU_BACKEND=cuda or hip");
    }
    void                deallocate(void*, size_t, cuda_caching_allocator::stream_type) {}
    void                record_stream(void*, cuda_caching_allocator::stream_type) {}
    void                add_free_memory_callback(cuda_caching_allocator::free_memory_callback) {}
    void                clear_free_memory_callbacks() {}
    void                empty_cache() {}
    void                set_max_cached_bytes(size_t bytes) { max_cached_bytes_ = bytes; }
    size_t              max_cached_bytes() const { return max_cached_bytes_; }
    unified_cache_stats stats() const { return unified_cache_stats{}; }
    int                 device() const { return device_; }

private:
    int    device_;
    size_t max_cached_bytes_;
};
#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP

cuda_caching_allocator::cuda_caching_allocator(int device, size_t max_cached_bytes)
    : impl_(std::make_unique<Impl>(device, max_cached_bytes))
{
}

cuda_caching_allocator::~cuda_caching_allocator() = default;

cuda_caching_allocator::cuda_caching_allocator(cuda_caching_allocator&&) noexcept = default;

cuda_caching_allocator& cuda_caching_allocator::operator=(cuda_caching_allocator&&) noexcept =
    default;

#if MEMORY_HAS_PROFILER
namespace
{
// HIP shares this translation unit with CUDA (gpu/gpu_runtime.h aliases the
// driver API), so the reported device type is a compile-time choice, same as
// profiler_kineto.cpp's deviceTypeFromActivity -- only one GPU backend is
// ever active in a given build. Values match profiler::device_enum (CPU=0,
// CUDA=1, HIP=2, PrivateUse1=3).
#if MEMORY_HAS_HIP
constexpr int16_t kGpuDeviceType = 2;  // profiler::device_enum::HIP
#elif MEMORY_HAS_CUDA
constexpr int16_t kGpuDeviceType = 1;  // profiler::device_enum::CUDA
#else
// Neither CUDA nor HIP compiled in (e.g. a Metal-backend build, where this
// TU still compiles the throwing stub Impl below -- see
// TestCachingAllocatorStub.cpp). Real GPU allocation on such builds goes
// through metal_caching_allocator instead; this constant only matters if a
// cuda_caching_allocator is constructed directly. Report as a generic custom
// backend rather than falsely claiming CUDA.
constexpr int16_t kGpuDeviceType = 3;  // profiler::device_enum::PrivateUse1
#endif
}  // namespace
#endif

void* cuda_caching_allocator::allocate(size_t size, stream_type stream)
{
    //cppcheck-suppress syntaxError
    if MEMORY_UNLIKELY (size == 0)
    {
        return nullptr;
    }
#if MEMORY_HAS_PROFILER
    // Skip the before/after stats() snapshot entirely (each an O(cached
    // blocks) locked scan) when no session actually wants memory events --
    // report_memory_usage() is a no-op in that case too, but only after
    // paying for the snapshot.
    if (profiler::memory_profiling_active())
    {
        const auto before = impl_->stats();
        void*      ptr     = impl_->allocate(size, stream);
        report_caching_allocator_delta(ptr, before, impl_->stats(), impl_->device(), kGpuDeviceType);
        return ptr;
    }
#endif
    return impl_->allocate(size, stream);
}

void cuda_caching_allocator::deallocate(void* ptr, size_t size, stream_type stream)
{
#if MEMORY_HAS_PROFILER
    if (ptr != nullptr && profiler::memory_profiling_active())
    {
        const auto before = impl_->stats();
        impl_->deallocate(ptr, size, stream);
        report_caching_allocator_delta(ptr, before, impl_->stats(), impl_->device(), kGpuDeviceType);
        return;
    }
#endif
    impl_->deallocate(ptr, size, stream);
}

void cuda_caching_allocator::record_stream(void* ptr, stream_type stream)
{
    impl_->record_stream(ptr, stream);
}

void cuda_caching_allocator::add_free_memory_callback(free_memory_callback callback)
{
    impl_->add_free_memory_callback(std::move(callback));
}

void cuda_caching_allocator::clear_free_memory_callbacks()
{
    impl_->clear_free_memory_callbacks();
}

void cuda_caching_allocator::empty_cache()
{
    impl_->empty_cache();
}

void cuda_caching_allocator::set_max_cached_bytes(size_t bytes)
{
    impl_->set_max_cached_bytes(bytes);
}

size_t cuda_caching_allocator::max_cached_bytes() const
{
    return impl_->max_cached_bytes();
}

unified_cache_stats cuda_caching_allocator::stats() const
{
    return impl_->stats();
}

int cuda_caching_allocator::device() const
{
    return impl_->device();
}

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
cuda_caching_allocator& caching_allocator_for_device(int device_index)
{
    static std::mutex                                                       registry_mutex;
    static std::unordered_map<int, std::unique_ptr<cuda_caching_allocator>> registry;

    std::scoped_lock const lock(registry_mutex);
    auto&                  entry = registry[device_index];
    if (entry == nullptr)
    {
        entry = std::make_unique<cuda_caching_allocator>(device_index);
    }
    return *entry;
}
#endif
}  // namespace gpu
}  // namespace memory
