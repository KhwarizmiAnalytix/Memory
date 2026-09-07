#include "gpu/cuda_caching_allocator.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
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
#include "util/exception.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/device_guard.h"
#include "gpu/gpu_runtime.h"
#endif
#if MEMORY_HAS_CUDA
#include <cuda.h>
#endif

#if MEMORY_HAS_PROFILER
#include "gpu/caching_allocator_profiler_report.h"
#endif

namespace memory
{
namespace gpu
{
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
using caching_config::round_request_size;
using caching_config::segment_size_for;

// Driver-backed segment (cudaMalloc/hipMalloc, or cuMemMap/hipMem* expandable).
struct raw_segment
{
    void*  ptr{nullptr};
    size_t size{0};
    bool   vm{false};
#if MEMORY_HAS_CUDA
    CUmemGenericAllocationHandle cu_handle{};
#endif
#if MEMORY_HAS_HIP && defined(HIP_VERSION) && HIP_VERSION >= 50600000
    hipMemGenericAllocationHandle_t hip_handle{};
#endif
};

#if MEMORY_HAS_CUDA
inline bool try_cu_vm_alloc(int device, size_t size, raw_segment& out)
{
    static std::once_flag init_once;
    std::call_once(init_once, []() { (void)cuInit(0); });

    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device;

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
            CUDA_SUCCESS ||
        granularity == 0)
    {
        return false;
    }
    size_t const padded = ((size + granularity - 1) / granularity) * granularity;
    CUdeviceptr  addr   = 0;
    if (cuMemAddressReserve(&addr, padded, granularity, 0, 0) != CUDA_SUCCESS)
    {
        return false;
    }
    CUmemGenericAllocationHandle handle{};
    if (cuMemCreate(&handle, padded, &prop, 0) != CUDA_SUCCESS)
    {
        (void)cuMemAddressFree(addr, padded);
        return false;
    }
    if (cuMemMap(addr, padded, 0, handle, 0) != CUDA_SUCCESS)
    {
        (void)cuMemRelease(handle);
        (void)cuMemAddressFree(addr, padded);
        return false;
    }
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(addr, padded, &access, 1) != CUDA_SUCCESS)
    {
        (void)cuMemUnmap(addr, padded);
        (void)cuMemRelease(handle);
        (void)cuMemAddressFree(addr, padded);
        return false;
    }
    out.ptr       = reinterpret_cast<void*>(addr);
    out.size      = padded;
    out.vm        = true;
    out.cu_handle = handle;
    return true;
}

inline void cu_vm_free(raw_segment const& seg)
{
    auto const addr = reinterpret_cast<CUdeviceptr>(seg.ptr);
    (void)cuMemUnmap(addr, seg.size);
    (void)cuMemRelease(seg.cu_handle);
    (void)cuMemAddressFree(addr, seg.size);
}
#endif

#if MEMORY_HAS_HIP && defined(HIP_VERSION) && HIP_VERSION >= 50600000
inline bool try_hip_vm_alloc(int device, size_t size, raw_segment& out)
{
    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = device;

    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum) !=
            hipSuccess ||
        granularity == 0)
    {
        return false;
    }
    size_t const padded = ((size + granularity - 1) / granularity) * granularity;
    void*        addr   = nullptr;
    if (hipMemAddressReserve(&addr, padded, granularity, 0, 0) != hipSuccess)
    {
        return false;
    }
    hipMemGenericAllocationHandle_t handle{};
    if (hipMemCreate(&handle, padded, &prop, 0) != hipSuccess)
    {
        (void)hipMemAddressFree(addr, padded);
        return false;
    }
    if (hipMemMap(addr, padded, 0, handle, 0) != hipSuccess)
    {
        (void)hipMemRelease(handle);
        (void)hipMemAddressFree(addr, padded);
        return false;
    }
    hipMemAccessDesc access{};
    access.location = prop.location;
    access.flags    = hipMemAccessFlagsProtReadWrite;
    if (hipMemSetAccess(addr, padded, &access, 1) != hipSuccess)
    {
        (void)hipMemUnmap(addr, padded);
        (void)hipMemRelease(handle);
        (void)hipMemAddressFree(addr, padded);
        return false;
    }
    out.ptr        = addr;
    out.size       = padded;
    out.vm         = true;
    out.hip_handle = handle;
    return true;
}

inline void hip_vm_free(raw_segment const& seg)
{
    (void)hipMemUnmap(seg.ptr, seg.size);
    (void)hipMemRelease(seg.hip_handle);
    (void)hipMemAddressFree(seg.ptr, seg.size);
}
#endif

inline raw_segment malloc_segment(int device, size_t size, cudaError_t* err_out, bool expandable)
{
    device_guard const guard(device);
    raw_segment        out;
    *err_out = cudaSuccess;
    // VM mapping is opt-in: a per-segment cuMemAddressReserve of exactly
    // `size` is not PyTorch expandable_segments (one large VA, incremental
    // physical maps) and was measured 1.25–1.66× slower than cudaMalloc.
#if MEMORY_HAS_CUDA
    if (expandable && try_cu_vm_alloc(device, size, out))
    {
        return out;
    }
#endif
#if MEMORY_HAS_HIP && defined(HIP_VERSION) && HIP_VERSION >= 50600000
    if (expandable && try_hip_vm_alloc(device, size, out))
    {
        return out;
    }
#endif
    void*             ptr = nullptr;
    cudaError_t const err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess)
    {
        (void)cudaGetLastError();
        *err_out = err;
        return {};
    }
    out.ptr  = ptr;
    out.size = size;
    out.vm   = false;
    return out;
}

inline void free_segment(int device, raw_segment const& seg)
{
    if (seg.ptr == nullptr)
    {
        return;
    }
    // A free path has no useful way to report "couldn't switch device" (void
    // return) and is reached from release_all_blocks_noexcept() during process
    // teardown, where the CUDA runtime may already be unloading — so this must
    // not throw the way segment allocation does.
    device_guard const guard(device, std::nothrow);
    if (seg.vm)
    {
#if MEMORY_HAS_CUDA
        cu_vm_free(seg);
#elif MEMORY_HAS_HIP && defined(HIP_VERSION) && HIP_VERSION >= 50600000
        hip_vm_free(seg);
#endif
        return;
    }
    (void)cudaFree(seg.ptr);
}

struct block_pool;

// A cache_block is a subrange of a segment (one driver allocation). Blocks are split on
// reuse and coalesced on free via the intrusive prev/next links; metadata is
// raw-allocated because ownership transfers between the free pools, the active
// map, and merge operations, mirroring the upstream implementation.
struct cache_block
{
    cache_block(void* p, size_t s, cudaStream_t st, block_pool* pl)
        : ptr(p), size(s), stream(st), pool(pl), segment_base(p)
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
    void*                  segment_base{nullptr};
    bool                   vm_backed{false};
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

#if MEMORY_HAS_PROFILER
#if MEMORY_HAS_HIP
constexpr int16_t kGpuDeviceType = 2;  // profiler::device_enum::HIP
#else
constexpr int16_t kGpuDeviceType = 1;  // profiler::device_enum::CUDA
#endif
#endif

}  // namespace

struct cuda_caching_allocator::Impl
{
    Impl(int device, size_t max_cached_bytes) : device_(device), max_cached_bytes_(max_cached_bytes)
    {
        // Validate device
        int device_count = 0;
        throw_on_cuda_error(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        LOGGING_CHECK(  //NOLINT
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
        LOGGING_CHECK(size > 0, "cuda_caching_allocator cannot allocate zero bytes");

        std::unique_lock lock(mutex_);
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
            // Drop the allocator lock across the driver call (PyTorch ~2.7+):
            // cudaMalloc/hipMalloc synchronize the device; holding the mutex
            // would stall every other allocate on this device.
            if (reserved_would_exceed_locked(alloc_size))
            {
                release_cached_blocks_locked();
            }
            if (reserved_would_exceed_locked(alloc_size))
            {
                fail_oom_locked(size, stream);
            }
            block = alloc_segment_unlocked(lock, pool, stream, alloc_size, false);
            if (block == nullptr)
            {
                // OOM chain: flush the entire cache (synchronize pending events and
                // release every releasable cached segment) and retry once before
                // failing, matching the upstream retry behavior.
                release_cached_blocks_locked();
                block = alloc_segment_unlocked(lock, pool, stream, alloc_size, true);
                if (block == nullptr)
                {
                    fail_oom_locked(size, stream);
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
        LOGGING_CHECK(
            it != allocated_blocks_.end(),
            "cuda_caching_allocator does not own the provided pointer");

        cache_block* block = it->second;
        LOGGING_CHECK(block->allocated, "cuda_caching_allocator detected a double free");

        allocated_blocks_.erase(it);
        block->allocated = false;
        stats_.successful_frees++;
        stats_.bytes_allocated -= block->size;
        record_trace_locked(
            gpu_memory_trace_action::free_requested, ptr, block->size, block->stream);
#if MEMORY_HAS_PROFILER
        report_event_locked(ptr, -static_cast<int64_t>(block->size));
#endif

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
        LOGGING_CHECK(
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

    void set_expandable_segments(bool enabled)
    {
        std::scoped_lock const lock(mutex_);
        expandable_segments_ = enabled;
    }

    bool expandable_segments() const
    {
        std::scoped_lock const lock(mutex_);
        return expandable_segments_;
    }

    void set_memory_fraction(double fraction)
    {
        if (fraction <= 0.0 || fraction > 1.0)
        {
            throw std::invalid_argument("set_memory_fraction: fraction must be in (0, 1]");
        }
        size_t const           total = query_device_total_memory();
        std::scoped_lock const lock(mutex_);
        memory_fraction_        = fraction;
        allowed_memory_maximum_ = static_cast<size_t>(fraction * static_cast<double>(total));
    }

    double memory_fraction() const
    {
        std::scoped_lock const lock(mutex_);
        return memory_fraction_;
    }

    void reset_peak_stats()
    {
        std::scoped_lock const lock(mutex_);
        peak_bytes_cached_ = bytes_cached_;
        stats_.peak_bytes_cached.store(bytes_cached_, std::memory_order_relaxed);
        stats_.peak_bytes_allocated.store(
            stats_.bytes_allocated.load(std::memory_order_relaxed), std::memory_order_relaxed);
        stats_.peak_bytes_reserved.store(
            stats_.bytes_reserved.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    size_t device_total_memory() const { return query_device_total_memory(); }

    size_t query_device_total_memory() const
    {
        device_guard const guard(device_);
        size_t             free_b  = 0;
        size_t             total_b = 0;
        throw_on_cuda_error(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        return total_b;
    }

    static void bump_peak_locked(std::atomic<size_t>& peak, size_t value)
    {
        if (value > peak.load(std::memory_order_relaxed))
        {
            peak.store(value, std::memory_order_relaxed);
        }
    }

    bool reserved_would_exceed_locked(size_t alloc_size) const
    {
        return stats_.bytes_reserved.load(std::memory_order_relaxed) + alloc_size >
               allowed_memory_maximum_;
    }

    void record_trace_locked(
        gpu_memory_trace_action action, void* address, size_t size, cudaStream_t stream)
    {
        history_.record(
            action,
            address,
            size,
            stats_.bytes_allocated.load(std::memory_order_relaxed),
            stats_.bytes_reserved.load(std::memory_order_relaxed),
            stream_as_int(stream));
    }

#if MEMORY_HAS_PROFILER
    void report_event_locked(void* ptr, int64_t nbytes)
    {
        report_caching_allocator_event(
            ptr,
            nbytes,
            stats_.bytes_allocated.load(std::memory_order_relaxed),
            stats_.bytes_reserved.load(std::memory_order_relaxed),
            device_,
            kGpuDeviceType);
    }
#endif

    [[noreturn]] void fail_oom_locked(size_t requested, cudaStream_t stream)
    {
        stats_.num_ooms++;
        record_trace_locked(gpu_memory_trace_action::oom, nullptr, requested, stream);
#if MEMORY_HAS_PROFILER
        report_caching_allocator_oom(
            static_cast<int64_t>(requested),
            stats_.bytes_allocated.load(std::memory_order_relaxed),
            stats_.bytes_reserved.load(std::memory_order_relaxed),
            device_,
            kGpuDeviceType);
#endif
        throw std::bad_alloc();
    }

    void record_memory_history(bool enabled, size_t max_entries)
    {
        std::scoped_lock const lock(mutex_);
        history_.set_enabled(enabled, max_entries);
    }

    gpu_memory_snapshot snapshot()
    {
        std::scoped_lock const lock(mutex_);
        record_trace_locked(gpu_memory_trace_action::snapshot, nullptr, 0, nullptr);

        std::map<uintptr_t, gpu_memory_segment_info> segments;
        std::map<void*, cache_block*>                unique;
        auto                                         consider = [&](cache_block* block)
        {
            if (block != nullptr)
            {
                unique[block->ptr] = block;
            }
        };
        for (auto& entry : allocated_blocks_)
        {
            consider(entry.second);
        }
        for (cache_block* block : small_blocks_.blocks)
        {
            consider(block);
        }
        for (cache_block* block : large_blocks_.blocks)
        {
            consider(block);
        }
        for (auto& kv : cuda_events_)
        {
            for (auto& queued : kv.second)
            {
                consider(queued.second);
            }
        }
        for (auto& entry : unique)
        {
            cache_block* block = entry.second;
            void* const  base  = block->segment_base != nullptr ? block->segment_base : block->ptr;
            size_t       seg_size = 0;
            auto         it       = driver_segments_.find(base);
            if (it != driver_segments_.end())
            {
                seg_size = it->second.size;
            }
            const bool active =
                block->allocated || block->event_count > 0 || !block->stream_uses.empty();
            add_snapshot_block(
                segments,
                base,
                seg_size,
                block->pool != nullptr && block->pool->is_small,
                block->vm_backed,
                stream_as_int(block->stream),
                block->ptr,
                block->size,
                block->requested_size,
                block->allocated,
                active);
        }
        return finish_snapshot(std::move(segments), history_.copy());
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

    cache_block* alloc_segment_unlocked(
        std::unique_lock<std::recursive_mutex>& lock,
        block_pool&                             pool,
        cudaStream_t                            stream,
        size_t                                  alloc_size,
        bool                                    is_retry)
    {
        if (is_retry)
        {
            stats_.num_alloc_retries++;
        }
        // Metadata is allocated before the driver call so a throwing new cannot
        // leak a successfully mapped segment.
        auto block = std::make_unique<cache_block>(nullptr, alloc_size, stream, &pool);
        lock.unlock();
        cudaError_t err = cudaSuccess;
        raw_segment raw = malloc_segment(device_, alloc_size, &err, expandable_segments_);
        lock.lock();
        if (raw.ptr == nullptr)
        {
            if (err != cudaSuccess && err != cudaErrorMemoryAllocation)
            {
                throw_on_cuda_error(err, "cudaMalloc");
            }
            return nullptr;
        }
        block->ptr          = raw.ptr;
        block->size         = raw.size;
        block->segment_base = raw.ptr;
        block->vm_backed    = raw.vm;
        block->registration_counter =
            registration_counter_global_.fetch_add(1, std::memory_order_relaxed) + 1;
        driver_segments_.emplace(raw.ptr, raw);
        stats_.driver_allocations++;
        stats_.bytes_reserved += raw.size;
        bump_peak_locked(
            stats_.peak_bytes_reserved, stats_.bytes_reserved.load(std::memory_order_relaxed));
        record_trace_locked(gpu_memory_trace_action::segment_alloc, raw.ptr, raw.size, stream);
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
            block->segment_base         = remaining->segment_base;
            block->vm_backed            = remaining->vm_backed;
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
        bump_peak_locked(
            stats_.peak_bytes_allocated, stats_.bytes_allocated.load(std::memory_order_relaxed));
        record_trace_locked(gpu_memory_trace_action::alloc, block->ptr, block->size, block->stream);
#if MEMORY_HAS_PROFILER
        report_event_locked(block->ptr, static_cast<int64_t>(block->size));
#endif
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
        record_trace_locked(
            gpu_memory_trace_action::free_completed, block->ptr, freed_size, block->stream);
    }

    void erase_from_pool_locked(block_pool& pool, cache_block* block)
    {
        auto const it = pool.blocks.find(block);
        if (it != pool.blocks.end() && *it == block)
        {
            pool.blocks.erase(it);
            return;
        }
        // Comparator lookup can miss if fields were mutated while the block
        // was in the set. Fall back to pointer identity so we never delete a
        // block that remains in the free pool.
        for (auto it2 = pool.blocks.begin(); it2 != pool.blocks.end(); ++it2)
        {
            if (*it2 == block)
            {
                pool.blocks.erase(it2);
                return;
            }
        }
    }

    void try_merge_locked(cache_block* dst, cache_block* src)
    {
        if (src == nullptr || src->allocated || src->event_count > 0 || !src->stream_uses.empty())
        {
            return;
        }
        bool const src_is_prev = dst->prev == src;
        bool const src_is_next = dst->next == src;
        if (!src_is_prev && !src_is_next)
        {
            return;
        }
        auto const* src_bytes = static_cast<char const*>(src->ptr);
        auto const* dst_bytes = static_cast<char const*>(dst->ptr);
        LOGGING_CHECK(
            (src_is_prev && src_bytes + src->size == dst_bytes) ||
                (src_is_next && dst_bytes + dst->size == src_bytes),
            "cuda_caching_allocator: merge of non-adjacent blocks");
        if (src_is_prev)  // [src dst]
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
        erase_from_pool_locked(*dst->pool, src);
        delete src;
    }

    void insert_events_locked(cache_block* block)
    {
        device_guard const     guard(device_);
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
        if (cuda_events_.empty())
        {
            return;
        }
        // Events are device-local: polling must run on the owning device.
        device_guard const guard(device_);

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
        device_guard const guard(device_);
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
        void* const base = block->segment_base != nullptr ? block->segment_base : block->ptr;
        auto        it   = driver_segments_.find(base);
        raw_segment raw;
        if (it != driver_segments_.end())
        {
            raw = it->second;
            driver_segments_.erase(it);
        }
        else
        {
            raw.ptr  = block->ptr;
            raw.size = block->size;
            raw.vm   = block->vm_backed;
        }
        stats_.driver_frees++;
        stats_.cache_evictions++;
        stats_.bytes_reserved -= block->size;
        record_trace_locked(
            gpu_memory_trace_action::segment_free, block->ptr, block->size, block->stream);
        delete block;
        free_segment(device_, raw);
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

    void release_all_blocks_noexcept() noexcept
    {
        device_guard const guard(device_, std::nothrow);

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
            auto it = driver_segments_.find(ptr);
            if (it != driver_segments_.end())
            {
                free_segment(device_, it->second);
                driver_segments_.erase(it);
            }
            else
            {
                cudaFree(ptr);
            }
        }
        driver_segments_.clear();
        for (cache_block* block : all_blocks)
        {
            delete block;
        }
        bytes_cached_      = 0;
        peak_bytes_cached_ = 0;
    }

    int    device_;
    size_t max_cached_bytes_;
    double memory_fraction_{1.0};
    size_t allowed_memory_maximum_{std::numeric_limits<size_t>::max()};
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
    memory_map<void*, raw_segment>                            driver_segments_;
    std::atomic<int64_t>                                      registration_counter_global_{0};
    unified_cache_stats                                       stats_;
    gpu_memory_history                                        history_;
    bool                                                      expandable_segments_{false};
};
#else
struct cuda_caching_allocator::Impl
{
    Impl(int device, size_t max_cached_bytes) : device_(device), max_cached_bytes_(max_cached_bytes)
    {
    }

    void* allocate(size_t, cuda_caching_allocator::stream_type)
    { throw std::runtime_error("cuda_caching_allocator requires MEMORY_GPU_BACKEND=cuda or hip"); }
    void                deallocate(void*, size_t, cuda_caching_allocator::stream_type) {}
    void                record_stream(void*, cuda_caching_allocator::stream_type) {}
    void                add_free_memory_callback(cuda_caching_allocator::free_memory_callback) {}
    void                clear_free_memory_callbacks() {}
    void                empty_cache() {}
    void                set_max_cached_bytes(size_t bytes) { max_cached_bytes_ = bytes; }
    size_t              max_cached_bytes() const { return max_cached_bytes_; }
    void                set_expandable_segments(bool enabled) { expandable_segments_ = enabled; }
    bool                expandable_segments() const { return expandable_segments_; }
    void                set_memory_fraction(double fraction) { memory_fraction_ = fraction; }
    double              memory_fraction() const { return memory_fraction_; }
    void                reset_peak_stats() {}
    size_t              device_total_memory() const { return 0; }
    unified_cache_stats stats() const { return unified_cache_stats{}; }
    void                record_memory_history(bool, size_t) {}
    gpu_memory_snapshot snapshot() { return gpu_memory_snapshot{}; }
    int                 device() const { return device_; }

private:
    int    device_;
    size_t max_cached_bytes_;
    bool   expandable_segments_{false};
    double memory_fraction_{1.0};
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

void* cuda_caching_allocator::allocate(size_t size, stream_type stream)
{
    //cppcheck-suppress syntaxError
    if MEMORY_UNLIKELY (size == 0)
    {
        return nullptr;
    }
    return impl_->allocate(size, stream);
}

void cuda_caching_allocator::deallocate(void* ptr, size_t size, stream_type stream)
{ impl_->deallocate(ptr, size, stream); }

void cuda_caching_allocator::record_stream(void* ptr, stream_type stream)
{ impl_->record_stream(ptr, stream); }

void cuda_caching_allocator::add_free_memory_callback(free_memory_callback callback)
{ impl_->add_free_memory_callback(std::move(callback)); }

void cuda_caching_allocator::clear_free_memory_callbacks()
{ impl_->clear_free_memory_callbacks(); }

void cuda_caching_allocator::empty_cache()
{ impl_->empty_cache(); }

void cuda_caching_allocator::set_max_cached_bytes(size_t bytes)
{ impl_->set_max_cached_bytes(bytes); }

size_t cuda_caching_allocator::max_cached_bytes() const
{ return impl_->max_cached_bytes(); }

void cuda_caching_allocator::set_expandable_segments(bool enabled)
{ impl_->set_expandable_segments(enabled); }

bool cuda_caching_allocator::expandable_segments() const
{ return impl_->expandable_segments(); }

void cuda_caching_allocator::set_memory_fraction(double fraction)
{ impl_->set_memory_fraction(fraction); }

double cuda_caching_allocator::memory_fraction() const
{ return impl_->memory_fraction(); }

void cuda_caching_allocator::reset_peak_stats()
{ impl_->reset_peak_stats(); }

size_t cuda_caching_allocator::device_total_memory() const
{ return impl_->device_total_memory(); }

unified_cache_stats cuda_caching_allocator::stats() const
{ return impl_->stats(); }

void cuda_caching_allocator::record_memory_history(bool enabled, size_t max_entries)
{ impl_->record_memory_history(enabled, max_entries); }

gpu_memory_snapshot cuda_caching_allocator::snapshot()
{ return impl_->snapshot(); }

int cuda_caching_allocator::device() const
{ return impl_->device(); }

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
