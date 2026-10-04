#include "gpu/cuda_caching_allocator.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "include/util/exception.h"

#include "common/cleanup_diagnostic.h"
#include "common/memory_containers.h"
#include "common/memory_macros.h"
#include "common/storage_identity.h"
#include "gpu/caching_allocator_config.h"

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
// per-device registry backing allocator<T>'s CUDA/HIP path). The #else stub below exists purely so
// this translation unit still compiles in Metal builds; constructing the allocator
// there throws at runtime.
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
namespace
{

using caching_config::add_saturating;
using caching_config::kMinBlockSize;
using caching_config::kSmallSize;
using caching_config::round_request_size;
using caching_config::round_up_saturating;
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
    size_t const padded = round_up_saturating(size, granularity);
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
    out.ptr       = reinterpret_cast<void*>(addr);  // NOLINT(performance-no-int-to-ptr): CUDA VM API
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
    size_t const padded = round_up_saturating(size, granularity);
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

// Node recycler for the cache's node-based containers (plan 3.8, §6.1): a node
// freed by erase() goes on a freelist instead of back to the heap, so the warm
// allocate/free path (erase from the free pool + insert on free, insert/erase in
// the live map) stops calling operator new once the cache has warmed up. Single
// node requests of the first-seen size are recycled; anything else (bucket
// arrays, other node types) falls through to operator new. Not thread-safe: the
// owner's mutex serializes every container operation.
class node_pool
{
public:
    node_pool() = default;
    node_pool(node_pool const&)            = delete;
    node_pool& operator=(node_pool const&) = delete;

    ~node_pool()
    {
        while (head_ != nullptr)
        {
            entry* const next = head_->next;
            ::operator delete(static_cast<void*>(head_));
            head_ = next;
        }
    }

    void* allocate(size_t bytes)
    {
        if (node_size_ == 0 && bytes >= sizeof(entry))
        {
            node_size_ = bytes;
        }
        if (bytes == node_size_ && head_ != nullptr)
        {
            entry* const e = head_;
            head_          = e->next;
            return e;
        }
        return ::operator new(bytes);
    }

    void deallocate(void* p, size_t bytes) noexcept
    {
        if (bytes == node_size_)
        {
            head_ = new (p) entry{head_};
            return;
        }
        ::operator delete(p);
    }

private:
    struct entry
    {
        entry* next;
    };
    entry* head_{nullptr};
    size_t node_size_{0};
};

template <typename T>
struct pool_allocator
{
    using value_type = T;
    template <typename U>
    struct rebind
    {
        using other = pool_allocator<U>;
    };

    explicit pool_allocator(node_pool* pool) noexcept : pool_(pool) {}
    template <typename U>
    pool_allocator(pool_allocator<U> const& other) noexcept : pool_(other.pool_)  // NOLINT
    {
    }

    T* allocate(size_t n)
    {
        if (n == 1)
        {
            return static_cast<T*>(pool_->allocate(sizeof(T)));
        }
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, size_t n) noexcept
    {
        if (n == 1)
        {
            pool_->deallocate(p, sizeof(T));
            return;
        }
        ::operator delete(static_cast<void*>(p));
    }

    template <typename U>
    bool operator==(pool_allocator<U> const& o) const noexcept
    {
        return pool_ == o.pool_;
    }
    template <typename U>
    bool operator!=(pool_allocator<U> const& o) const noexcept
    {
        return pool_ != o.pool_;
    }

    node_pool* pool_;
};


// Inline set for recording cross-stream uses (plan §6.1, P3.2, H2).
// Holds up to kInline streams without heap allocation; a heap-allocated
// overflow std::set handles the rare case of > kInline distinct streams.
struct inline_stream_set
{
    static constexpr int kInline = 4;

    bool empty() const noexcept { return size_ == 0; }

    void insert(cudaStream_t s)
    {
        for (int i = 0; i < size_; ++i)
        {
            if (slots_[i] == s)
                return;
        }
        if (size_ < kInline)
        {
            slots_[size_++] = s;
        }
        else
        {
            if (!overflow_)
                overflow_ = std::make_unique<std::set<cudaStream_t>>();
            overflow_->insert(s);
        }
    }

    size_t size() const noexcept
    {
        return static_cast<size_t>(size_) + (overflow_ ? overflow_->size() : 0);
    }

    // Visits every recorded stream once, without copying the set.
    template <typename F>
    void for_each(F&& f) const
    {
        for (int i = 0; i < size_; ++i)
            f(slots_[i]);
        if (overflow_)
        {
            for (cudaStream_t s : *overflow_)
                f(s);
        }
    }

    void clear() noexcept
    {
        size_ = 0;
        overflow_.reset();
    }

private:
    int          size_{0};
    cudaStream_t slots_[kInline]{};
    std::unique_ptr<std::set<cudaStream_t>> overflow_;
};

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

    void*              ptr;
    size_t             size;
    size_t             requested_size{0};
    cudaStream_t       stream;
    block_pool*        pool;
    bool               allocated{false};
    cache_block*       prev{nullptr};
    cache_block*       next{nullptr};
    int                event_count{0};
    inline_stream_set  stream_uses;
    void*              segment_base{nullptr};
    // Identity of the allocation that last occupied this block (task 7.2).
    // Assigned when the block is handed out, kept through free, deferred
    // completion and quarantine, and replaced (never reused) by the next
    // allocation. A free tail created by a split carries 0.
    uint64_t           alloc_id{0};
    bool                   vm_backed{false};
    // Set when cudaEventRecord fails for one of this block's recorded
    // cross-stream uses partway through insert_events_locked(): the streams
    // after the failure never got an event, so their pending work cannot be
    // proven complete. A quarantined block is permanently withheld from the
    // free pools even once its (partial) event_count reaches zero, rather
    // than being reused while a use we couldn't track might still be live.
    bool quarantined{false};
    // Segment creation order; equal-size free blocks recycle FIFO (upstream
    // registration_counter). Search keys keep the -1 default so lower_bound
    // finds the oldest matching block.
    int64_t registration_counter{-1};
};

// Lightweight key for heterogeneous lookup in block_pool::blocks.  Avoids
// constructing a full cache_block just to call lower_bound.
// registration_counter=-1 finds the oldest (FIFO)
// block of a given (stream, size) pair because all real blocks are assigned
// counter values >= 1 by alloc_segment_unlocked.
struct block_search_key
{
    cudaStream_t stream{nullptr};
    size_t       size{0};
    int64_t      registration_counter{-1};
    void*        ptr{nullptr};
};

struct cache_block_comparator
{
    // Transparent comparator: std::set::lower_bound accepts block_search_key
    // directly without a cache_block wrapper.
    using is_transparent = void;

    bool operator()(const cache_block* a, const cache_block* b) const
    {
        return less(a->stream, a->size, a->registration_counter, a->ptr,
                    b->stream, b->size, b->registration_counter, b->ptr);
    }
    bool operator()(const block_search_key& a, const cache_block* b) const
    {
        return less(a.stream, a.size, a.registration_counter, a.ptr,
                    b->stream, b->size, b->registration_counter, b->ptr);
    }
    bool operator()(const cache_block* a, const block_search_key& b) const
    {
        return less(a->stream, a->size, a->registration_counter, a->ptr,
                    b.stream, b.size, b.registration_counter, b.ptr);
    }

private:
    static bool less(cudaStream_t sa, size_t za, int64_t ra, void* pa,
                     cudaStream_t sb, size_t zb, int64_t rb, void* pb)
    {
        auto sv = [](cudaStream_t s) { return reinterpret_cast<uintptr_t>(s); };
        auto pv = [](void* p) { return reinterpret_cast<uintptr_t>(p); };
        if (sa != sb) return sv(sa) < sv(sb);
        if (za != zb) return za < zb;
        if (ra != rb) return ra < rb;
        return pv(pa) < pv(pb);
    }
};

// Free blocks of one size class, ordered by (stream, size, ptr): blocks are only
// ever reused on the stream they were allocated on.
struct block_pool
{
    block_pool(bool is_small_pool, node_pool& nodes)
        : blocks(cache_block_comparator{}, pool_allocator<cache_block*>(&nodes)),
          is_small(is_small_pool)
    {
    }

    std::set<cache_block*, cache_block_comparator, pool_allocator<cache_block*>> blocks;
    const bool                                                                   is_small;
};

// Block freelist for P3.2 (plan §6.1, H3): recycles cache_block metadata
// allocations from splits/merges so the warm alloc/free path avoids the
// general-purpose allocator.
//
// release(b): calls b->~cache_block() then reuses the raw storage as a
//             free_entry link; acquire(...): pops that storage and
//             placement-news a fresh cache_block there.
struct block_freelist
{
    struct free_entry { free_entry* next; };
    static_assert(sizeof(cache_block) >= sizeof(free_entry),
                  "cache_block too small for block_freelist chain");
    static_assert(alignof(cache_block) >= alignof(free_entry),
                  "cache_block alignment insufficient for block_freelist");

    cache_block* acquire(void* ptr, size_t sz, cudaStream_t stream, block_pool* pool)
    {
        if (head_)
        {
            free_entry* e = head_;
            head_         = e->next;
            return new (e) cache_block(ptr, sz, stream, pool);
        }
        return new cache_block(ptr, sz, stream, pool);
    }

    void release(cache_block* b)
    {
        b->~cache_block();
        head_ = new (b) free_entry{head_};
    }

    ~block_freelist()
    {
        while (head_)
        {
            free_entry* nxt = head_->next;
            ::operator delete(static_cast<void*>(head_));
            head_ = nxt;
        }
    }

    free_entry* head_{nullptr};
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
        LOGGING_CHECK(  // NOLINT
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
        LOGGING_CHECK_DEBUG(size > 0, "cuda_caching_allocator cannot allocate zero bytes");

        deferred_failure_flush const flush{*this};  // runs after `lock` is released
        std::unique_lock             lock(mutex_);
        process_events_locked();

        size_t const rounded    = round_request_size(size);
        block_pool&  pool       = rounded <= kSmallSize ? small_blocks_ : large_blocks_;
        size_t const alloc_size = segment_size_for(rounded);

        cache_block* block = get_free_block_locked(pool, stream, rounded);
        if (block == nullptr && !pending_events_.empty())
        {
            // Pressure: the bounded poll above may have left completed events
            // unreaped; reap them all before spending a driver call.
            process_events_locked(kPollUnbounded);
            block = get_free_block_locked(pool, stream, rounded);
        }
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
                // alloc_segment_unlocked's own driver call drops the lock, so a
                // concurrent set_memory_fraction() can shrink headroom between
                // this function's earlier checks and this retry. Recheck before
                // spending a second driver call so a stale "fits" decision
                // cannot commit an over-budget segment.
                if (reserved_would_exceed_locked(alloc_size))
                {
                    fail_oom_locked(size, stream);
                }
                block = alloc_segment_unlocked(lock, pool, stream, alloc_size, true);
                if (block == nullptr)
                {
                    fail_oom_locked(size, stream);
                }
            }
        }

        void* ptr = nullptr;
        try
        {
            ptr = alloc_found_block_locked(block, rounded, size);
        }
        catch (...)
        {
            // Nothing was committed: return the block (a cached hit, or the
            // fresh segment) to its pool so neither budget nor segment leaks.
            restore_unallocated_block_noexcept(block);
            throw;
        }
        // process_events_locked above may have grown the cache past the cap.
        // The allocation is already committed: a trim failure must not turn it
        // into an error the caller would answer by losing the pointer.
        trim_cache_best_effort_locked();
        return ptr;
    }

    void deallocate(void* ptr, size_t /*size*/, cudaStream_t stream)
    {
        if (ptr == nullptr)
        {
            return;
        }

        deferred_failure_flush const flush{*this};  // runs after the lock is released
        std::scoped_lock const       lock(mutex_);
        process_events_locked();

        auto it = allocated_blocks_.find(ptr);
        LOGGING_CHECK(
            it != allocated_blocks_.end(),
            "cuda_caching_allocator does not own the provided pointer");

        cache_block* block = it->second;
        LOGGING_CHECK(block->allocated, "cuda_caching_allocator detected a double free");

        // The stream hint maps to recordStream semantics: freeing after use on a
        // stream other than the allocation stream counts as a cross-stream use.
        // nullptr is CUDA/HIP's own spelling of the default stream, a real
        // stream identity distinct from any non-default allocation stream, so
        // it must not be treated as "no hint" here — a block allocated on a
        // non-default stream and freed after use on the default stream is a
        // genuine cross-stream use that needs event-deferred reclamation.
        // Recording it can allocate, so it comes before any state change: a
        // throw leaves the block live and the free retryable (task 1.7).
        if (stream != block->stream)
        {
            block->stream_uses.insert(stream);
        }

        allocated_blocks_.erase(it);
        block->allocated = false;
        stats_.successful_frees++;
        stats_.bytes_allocated -= block->size;
        record_trace_locked(
            gpu_memory_trace_action::free_requested,
            ptr,
            block->size,
            block->stream,
            block->requested_size,
            block->alloc_id);
#if MEMORY_HAS_PROFILER
        report_event_locked(ptr, -static_cast<int64_t>(block->size));
#endif

        if (!block->stream_uses.empty())
        {
            insert_events_locked(block);
        }
        else
        {
            free_block_locked(block);
        }

        // The block is already free: a trim failure must not surface as a failed
        // free the caller might retry (a double free).
        trim_cache_best_effort_locked();
    }

    void deallocate_with_stream_lookup(void* ptr, size_t /*nbytes*/) noexcept
    {
        if (ptr == nullptr)
        {
            return;
        }

        // Failures are reported through cleanup_diagnostic only after mutex_ is
        // released: the diagnostic may run a user handler (plan §5.2, task 1.3).
        bool                         failed = false;
        deferred_failure_flush const flush{*this};  // declared before the lock: runs after it
        try
        {
            std::scoped_lock const lock(mutex_);
            process_events_locked();

            auto it = allocated_blocks_.find(ptr);
            // Not owned by this allocator, or a double free: a deleter path
            // cannot throw, so it counts instead.
            if (it == allocated_blocks_.end() || !it->second->allocated)
            {
                failed = true;
            }
            else
            {
                cache_block* block = it->second;

                // Look up the allocation stream from the block and deallocate.
                // Since we're deallocating on the same stream it was allocated on,
                // there are no new cross-stream uses to record beyond what was
                // already recorded via record_stream().
                allocated_blocks_.erase(it);
                block->allocated = false;
                stats_.successful_frees++;
                stats_.bytes_allocated -= block->size;
                record_trace_locked(
                    gpu_memory_trace_action::free_requested,
                    ptr,
                    block->size,
                    block->stream,
                    block->requested_size,
                    block->alloc_id);
#if MEMORY_HAS_PROFILER
                report_event_locked(ptr, -static_cast<int64_t>(block->size));
#endif

                // No additional stream hints provided by this deleter path, so just
                // check if prior record_stream() calls created any cross-stream uses.
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
        }
        catch (...)
        {
            failed = true;
        }
        if (failed)
        {
            cleanup_diagnostic::record_failure(cleanup_source::gpu_cache);
        }
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
        // stream == nullptr denotes the default stream (a valid, distinct
        // stream identity), not "no stream to record" — only a null ptr
        // makes this a no-op. See the deallocate() comment on the same
        // default-stream/no-hint distinction.
        if (ptr == nullptr)
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

    bool owns_live_allocation(void const* ptr) const
    {
        std::scoped_lock const lock(mutex_);
        auto const             it = allocated_blocks_.find(const_cast<void*>(ptr));
        return it != allocated_blocks_.end() && it->second->allocated;
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
        if (std::isnan(fraction) || fraction <= 0.0 || fraction > 1.0)
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
        peak_bytes_cached_ = bytes_cached_.load(std::memory_order_relaxed);
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

    // Includes pending_reserved_bytes_: budget reserved by other threads that
    // have already passed this check and dropped the lock to call cudaMalloc,
    // but have not yet updated stats_.bytes_reserved with the driver result.
    // Without it, N threads can each observe headroom for `alloc_size` and all
    // proceed to the driver, jointly reserving up to N * alloc_size over the
    // cap before any of them account for the others.
    bool reserved_would_exceed_locked(size_t alloc_size) const
    {
        size_t const reserved = stats_.bytes_reserved.load(std::memory_order_relaxed);
        size_t const total =
            add_saturating(add_saturating(reserved, pending_reserved_bytes_), alloc_size);
        return total > allowed_memory_maximum_;
    }

    // Telemetry runs after the operation's state is committed (or while the
    // original error is already propagating), so it must never fail the
    // operation: a failed trace record is counted, not thrown (task 1.7).
    void record_trace_locked(
        gpu_memory_trace_action action,
        void*                   address,
        size_t                  size,
        cudaStream_t            stream,
        size_t                  requested_size = 0,
        uint64_t                alloc_id       = 0) noexcept
    {
        try
        {
            history_.record(
                action,
                address,
                size,
                requested_size,
                stats_.bytes_allocated.load(std::memory_order_relaxed),
                stats_.bytes_reserved.load(std::memory_order_relaxed),
                stream_as_int(stream),
                alloc_id);
        }
        catch (...)
        {
            note_failure_locked();
        }
    }

#if MEMORY_HAS_PROFILER
    void report_event_locked(void* ptr, int64_t nbytes) noexcept
    {
        try
        {
            report_caching_allocator_event(
                ptr,
                nbytes,
                stats_.bytes_allocated.load(std::memory_order_relaxed),
                stats_.bytes_reserved.load(std::memory_order_relaxed),
                device_,
                kGpuDeviceType);
        }
        catch (...)
        {
            note_failure_locked();
        }
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
        for (pending_event const& queued : pending_events_)
        {
            consider(queued.block);
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

    // O(1) lock-free reads for the four basic stats (plan §6.1, P3.5).
    // Single relaxed atomic load — no mutex.  For a consistent full snapshot
    // (including bytes_cached and cache_blocks), use stats().
    size_t bytes_allocated_now()      const noexcept { return stats_.bytes_allocated.load(std::memory_order_relaxed); }
    size_t peak_bytes_allocated_now() const noexcept { return stats_.peak_bytes_allocated.load(std::memory_order_relaxed); }
    size_t bytes_reserved_now()       const noexcept { return stats_.bytes_reserved.load(std::memory_order_relaxed); }
    size_t peak_bytes_reserved_now()  const noexcept { return stats_.peak_bytes_reserved.load(std::memory_order_relaxed); }
    size_t bytes_cached_now()         const noexcept { return bytes_cached_.load(std::memory_order_relaxed); }
    size_t peak_bytes_cached_now()    const noexcept { return peak_bytes_cached_.load(std::memory_order_relaxed); }

    int device() const { return device_; }

private:
    // Single writer (mutex_ held): a relaxed load/store pair, not an atomic RMW.
    // Publishes the new value (and its peak) for lock-free readers.
    void add_cached_locked(std::ptrdiff_t delta) noexcept
    {
        size_t const now = bytes_cached_.load(std::memory_order_relaxed) + static_cast<size_t>(delta);
        bytes_cached_.store(now, std::memory_order_relaxed);
        if (now > peak_bytes_cached_.load(std::memory_order_relaxed))
        {
            peak_bytes_cached_.store(now, std::memory_order_relaxed);
        }
    }

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
        // Use a lightweight search key so no cache_block construction
        // is needed here.
        block_search_key const key{stream, size};
        auto                   it = pool.blocks.lower_bound(key);
        // Free pools are stream-scoped: a block belonging to another stream is
        // never reused (upstream get_free_block rule).
        if (it == pool.blocks.end() || (*it)->stream != stream)
        {
            return nullptr;
        }
        cache_block* block = *it;
        pool.blocks.erase(it);
        add_cached_locked(-static_cast<std::ptrdiff_t>(block->size));
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

        cudaError_t err = cudaSuccess;
        raw_segment raw;
        {
            // Reserve this request's budget, and snapshot expandable_segments_,
            // under the lock before dropping it: reserved_would_exceed_locked()
            // on a concurrent thread must see this allocation as already spoken
            // for, and expandable_segments_ must not be read concurrently with
            // set_expandable_segments()'s write to it.
            pending_reserved_bytes_ += alloc_size;
            bool const expandable = expandable_segments_;
            lock.unlock();

            // Rolls back this request's pending-budget reservation on every
            // exit from this scope -- success, a driver-reported failure, or
            // an exception thrown by malloc_segment itself (its device_guard's
            // cudaGetDevice/cudaSetDevice can throw). Re-acquires `lock` first
            // if it isn't already held: malloc_segment can throw before
            // control returns here to relock manually, which used to skip the
            // decrement entirely and leak reserved headroom permanently.
            struct pending_reservation_guard
            {
                Impl&                                   self;
                std::unique_lock<std::recursive_mutex>& lock;
                size_t                                   amount;

                pending_reservation_guard(
                    Impl& s,
                    std::unique_lock<std::recursive_mutex>& l,
                    size_t a) noexcept
                    : self(s), lock(l), amount(a)
                {
                }

                ~pending_reservation_guard() noexcept
                {
                    if (!lock.owns_lock())
                    {
                        try
                        {
                            lock.lock();
                        }
                        catch (...)  // NOLINT(bugprone-empty-catch)
                        {
                            // Suppress lock failure during cleanup; the only path
                            // here is exception unwinding, and we cannot propagate.
                        }
                    }
                    self.pending_reserved_bytes_ -= amount;
                }
            } const pending_guard{*this, lock, alloc_size};

            raw = malloc_segment(device_, alloc_size, &err, expandable);
            // pending_guard's destructor fires at the end of this scope (or
            // during unwind if malloc_segment threw), reacquiring `lock` and
            // decrementing exactly once either way.
        }
        // `lock` is guaranteed held from here on: either malloc_segment
        // returned normally and the guard reacquired it above, or an
        // exception already propagated past this point during unwind.
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
        // The driver segment exists from here. Registering it can allocate; if
        // that throws the segment must be returned to the driver, not leaked
        // (the budget reservation is rolled back by pending_guard above, and
        // nothing was added to stats_ yet).
        try
        {
            driver_segments_.emplace(raw.ptr, raw);
        }
        catch (...)
        {
            free_segment(device_, raw);
            throw;
        }
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

    // Transactional (task 1.7): every fallible step (metadata, map insert, pool
    // insert) happens before the block list or counters change, and each later
    // fallible step undoes the earlier mutations on failure. On a throw nothing
    // is committed and the caller still owns @p block (see
    // restore_unallocated_block_noexcept).
    void* alloc_found_block_locked(cache_block* block, size_t rounded, size_t orig_size)
    {
        cache_block* const remaining = block;
        cache_block*       head      = block;  // the block handed to the caller
        bool const         split     = should_split(block, rounded);
        if (split)
        {
            // `head` takes the first `rounded` bytes; `remaining` keeps the tail.
            head = block_freelist_.acquire(
                remaining->ptr, rounded, remaining->stream, remaining->pool);
        }

        // A live pointer must map to exactly one block: a pre-existing entry is
        // a stale map entry and is rejected rather than silently kept.
        try
        {
            if (!allocated_blocks_.emplace(head->ptr, head).second)
            {
                throw std::logic_error(
                    "cuda_caching_allocator: stale allocated-block entry for a reused address");
            }
        }
        catch (...)
        {
            if (split)
            {
                block_freelist_.release(head);
            }
            throw;
        }

        if (split)
        {
            void* const    old_ptr  = remaining->ptr;
            uint64_t const old_id   = remaining->alloc_id;
            size_t const   old_req  = remaining->requested_size;
            cache_block* old_prev = remaining->prev;
            head->registration_counter = remaining->registration_counter;
            head->segment_base         = remaining->segment_base;
            head->vm_backed            = remaining->vm_backed;
            head->prev                 = old_prev;
            if (old_prev != nullptr)
            {
                old_prev->next = head;
            }
            head->next      = remaining;
            remaining->prev = head;
            remaining->ptr  = static_cast<char*>(remaining->ptr) + rounded;
            remaining->size -= rounded;
            remaining->alloc_id       = 0;
            remaining->requested_size = 0;
            try
            {
                remaining->pool->blocks.insert(remaining);
            }
            catch (...)
            {
                remaining->size += rounded;
                remaining->alloc_id       = old_id;
                remaining->requested_size = old_req;
                remaining->ptr  = old_ptr;
                remaining->prev = old_prev;
                if (old_prev != nullptr)
                {
                    old_prev->next = remaining;
                }
                allocated_blocks_.erase(head->ptr);
                block_freelist_.release(head);
                throw;
            }
            add_cached_locked(static_cast<std::ptrdiff_t>(remaining->size));
        }

        // Committed: nothing below can fail.
        head->allocated      = true;
        head->requested_size = orig_size;
        head->alloc_id       = next_allocation_id().value;
        stats_.successful_allocations++;
        stats_.bytes_allocated += head->size;
        bump_peak_locked(
            stats_.peak_bytes_allocated, stats_.bytes_allocated.load(std::memory_order_relaxed));
        record_trace_locked(
            gpu_memory_trace_action::alloc,
            head->ptr,
            head->size,
            head->stream,
            orig_size,
            head->alloc_id);
#if MEMORY_HAS_PROFILER
        report_event_locked(head->ptr, static_cast<int64_t>(head->size));
#endif
        return head->ptr;
    }

    // Put a block that was taken for an allocation that then failed back into
    // its free pool. If even that cannot allocate (a double fault) the segment
    // stays registered in driver_segments_ and is freed at teardown; only the
    // block metadata is lost, and the failure is counted.
    void restore_unallocated_block_noexcept(cache_block* block) noexcept
    {
        try
        {
            block->pool->blocks.insert(block);
            add_cached_locked(static_cast<std::ptrdiff_t>(block->size));
            
        }
        catch (...)
        {
            note_failure_locked();
        }
    }

    void trim_cache_best_effort_locked() noexcept
    {
        try
        {
            trim_cache_locked();
        }
        catch (...)
        {
            note_failure_locked();
        }
    }

    // Failures noticed while mutex_ is held are only counted here and reported to
    // cleanup_diagnostic by deferred_failure_flush once the lock is released.
    void note_failure_locked() noexcept { deferred_failures_.fetch_add(1, std::memory_order_relaxed); }

    struct deferred_failure_flush
    {
        Impl& self;
        ~deferred_failure_flush()
        {
            size_t n = self.deferred_failures_.exchange(0, std::memory_order_relaxed);
            while (n-- > 0)
            {
                cleanup_diagnostic::record_failure(cleanup_source::gpu_cache);
            }
        }
    };

    // Quarantined metadata is tracked for teardown; failing to track it loses only
    // the metadata (the memory stays withheld), and is counted.
    void track_quarantined_metadata_locked(cache_block* block) noexcept
    {
        try
        {
            quarantine_metadata_.push_back(block);
        }
        catch (...)
        {
            note_failure_locked();
        }
    }

    void free_block_locked(cache_block* block)
    {
        size_t const freed_size = block->size;
        // Capture the address actually being freed before merging can move
        // it: a merge with a preceding free block reassigns block->ptr to
        // that neighbor's (earlier) base address (see try_merge_locked's
        // src_is_prev branch), so recording block->ptr *after* merging would
        // pair this free_completed trace entry with the wrong address and
        // break alloc/free event pairing for anything replaying the trace.
        void* const    freed_ptr       = block->ptr;
        size_t const   freed_requested = block->requested_size;
        uint64_t const freed_id        = block->alloc_id;
        try_merge_locked(block, block->prev);
        try_merge_locked(block, block->next);

        // Merging only relabels sizes already counted in the pool; the net new
        // cached bytes are the freed block's own (pre-merge) size.
        try
        {
            block->pool->blocks.insert(block);
        }
        catch (...)
        {
            // Out of memory for the pool node: the (possibly merged) block cannot
            // be cached. The block lists are consistent, so withhold it as a
            // quarantined block and count the failure; the free itself stands.
            block->quarantined = true;
            track_quarantined_metadata_locked(block);
            note_failure_locked();
            return;
        }
        add_cached_locked(static_cast<std::ptrdiff_t>(freed_size));
        
        record_trace_locked(
            gpu_memory_trace_action::free_completed,
            freed_ptr,
            freed_size,
            block->stream,
            freed_requested,
            freed_id);
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
        if (src == nullptr || src->allocated || src->quarantined || src->event_count > 0 ||
            !src->stream_uses.empty())
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
        LOGGING_CHECK_DEBUG(
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
        // Null the list pointers before freeing: any stale dereference of src
        // (e.g. through a mis-linked neighbor) sees null rather than dangling data.
        src->prev = nullptr;
        src->next = nullptr;
        block_freelist_.release(src);
    }

    void insert_events_locked(cache_block* block)
    {
        // Tracks an event acquired from the pool but not yet confirmed queued
        // into pending_events_ (i.e. its cudaEventRecord has not yet succeeded).
        // A failure between acquiring it and queuing it must recycle it here,
        // or the underlying driver event object leaks: it would be neither in
        // event_pool_ (available for reuse) nor pending_events_ (destroyed at
        // allocator teardown via the pool).
        cudaEvent_t pending_event = nullptr;
        // Boundary/interop path: a CUDA error here must not orphan the block
        // between the pools and the event queues. Queue room is reserved first
        // so nothing after a successful cudaEventRecord can throw.
        try
        {
            pending_events_.reserve(pending_events_.size() + block->stream_uses.size());
            // The device is activated only to create an event (plan 3.3): a
            // pooled event is recorded without touching the current device,
            // because an event stays bound to the device it was created on.
            std::optional<device_guard> guard;
            block->stream_uses.for_each(
                [&](cudaStream_t stream)
                {
                    pending_event = acquire_event_locked(guard);
                    throw_on_cuda_error(cudaEventRecord(pending_event, stream), "cudaEventRecord");
                    pending_events_.push_back({pending_event, block, stream});
                    block->event_count++;
                    pending_event = nullptr;  // ownership transferred to pending_events_
                });
            block->stream_uses.clear();
        }
        catch (...)
        {
            if (pending_event != nullptr)
            {
                // A failed cudaEventRecord does not invalidate the event
                // object itself (only this record attempt); recycle it for
                // reuse rather than leaking the driver resource.
                recycle_event_locked(pending_event);
            }
            block->stream_uses.clear();
            // This path is only entered for a block that has recorded stream
            // uses, so their completion is unproven unless every event was
            // queued (and then nothing could have thrown). Reusing the block
            // would race a new allocation into memory still in use. Quarantine:
            // already-queued events still recycle normally, but
            // block->quarantined prevents free_block_locked from handing this
            // memory out again (and try_merge_locked from absorbing it).
            block->quarantined = true;
            if (block->event_count == 0)
            {
                // No event was queued: the block will never appear in
                // pending_events_ and cannot be found by release_all_blocks_noexcept
                // via any pool or map. Track it here so teardown can free the
                // cache_block metadata (the GPU memory stays withheld).
                track_quarantined_metadata_locked(block);
            }
            throw;
        }
    }

    cudaEvent_t acquire_event_locked(std::optional<device_guard>& guard)
    {
        if (!event_pool_.empty())
        {
            cudaEvent_t event = event_pool_.back();
            event_pool_.pop_back();
            return event;
        }
        if (!guard)
        {
            guard.emplace(device_);
        }
        cudaEvent_t event = nullptr;
        throw_on_cuda_error(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags");
        return event;
    }

    // Called from catch blocks and cleanup paths, so it must not throw: if the
    // pool cannot grow, the (idle) event is destroyed instead of leaked.
    void recycle_event_locked(cudaEvent_t event) noexcept
    {
        try
        {
            event_pool_.push_back(event);
        }
        catch (...)
        {
            (void)cudaEventDestroy(event);
        }
    }

    // Polls pending cross-stream events. Idle (nothing pending) costs one
    // branch and no driver call. Events complete in order per stream, so after a
    // stream's first not-ready event its later events are skipped without a
    // query. `max_queries` bounds the driver queries of one call (plan 3.3);
    // pressure paths pass kPollUnbounded to force progress. cudaEventQuery does
    // not need the owning device to be current (an event stays bound to the
    // device that created it), so no device guard is taken here.
    static constexpr size_t kPollBudget    = 16;
    static constexpr size_t kPollUnbounded = std::numeric_limits<size_t>::max();

    void process_events_locked(size_t max_queries = kPollBudget)
    {
        if (pending_events_.empty())
        {
            return;
        }
        constexpr size_t kMaxBlocked = 8;
        cudaStream_t     blocked[kMaxBlocked];
        size_t           blocked_count = 0;
        size_t           queries       = 0;
        size_t const     count         = pending_events_.size();
        size_t           write         = 0;
        size_t           read          = 0;

        // Moves the unexamined tail down so the vector stays consistent when the
        // scan stops early (budget spent, or a driver error).
        auto const keep_tail = [&]()
        {
            for (; read < count; ++read, ++write)
            {
                pending_events_[write] = pending_events_[read];
            }
            pending_events_.resize(write);
        };

        for (; read < count; ++read)
        {
            pending_event const entry = pending_events_[read];
            bool                skip  = false;
            for (size_t i = 0; i < blocked_count; ++i)
            {
                skip = skip || blocked[i] == entry.stream;
            }
            if (!skip && queries >= max_queries)
            {
                break;
            }
            if (!skip)
            {
                ++queries;
                cudaError_t const status = cudaEventQuery(entry.event);
                if (status == cudaSuccess)
                {
                    recycle_event_locked(entry.event);
                    complete_event_locked(entry.block);
                    continue;
                }
                if (status != cudaErrorNotReady)
                {
                    keep_tail();
                    throw_on_cuda_error(status, "cudaEventQuery");
                }
                (void)cudaGetLastError();  // clear the not-ready error state
                if (blocked_count < kMaxBlocked)
                {
                    blocked[blocked_count++] = entry.stream;
                }
            }
            pending_events_[write++] = entry;
        }
        keep_tail();
    }

    // One queued event of `block` has completed. A quarantined block (see
    // insert_events_locked) is never returned to a free pool: its untracked
    // streams' uses were never proven complete, so it stays withheld rather
    // than becoming reusable on the strength of a partial count. Its metadata
    // is tracked for teardown (the GPU memory remains withheld).
    void complete_event_locked(cache_block* block)
    {
        block->event_count--;
        if (block->event_count != 0)
        {
            return;
        }
        if (block->quarantined)
        {
            track_quarantined_metadata_locked(block);
        }
        else
        {
            free_block_locked(block);
        }
    }

    void synchronize_and_free_events_locked()
    {
        stats_.num_sync_all_streams++;
        device_guard const guard(device_);
        size_t             done = 0;
        try
        {
            for (; done < pending_events_.size(); ++done)
            {
                pending_event const entry = pending_events_[done];
                // On failure the entry stays queued and untouched (no event is
                // both pooled and queued).
                throw_on_cuda_error(cudaEventSynchronize(entry.event), "cudaEventSynchronize");
                recycle_event_locked(entry.event);
                complete_event_locked(entry.block);
            }
        }
        catch (...)
        {
            pending_events_.erase(
                pending_events_.begin(),
                pending_events_.begin() + static_cast<std::ptrdiff_t>(done));
            throw;
        }
        pending_events_.clear();
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
                add_cached_locked(-static_cast<std::ptrdiff_t>(block->size));
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
            add_cached_locked(-static_cast<std::ptrdiff_t>(victim->size));
            victim->pool->blocks.erase(victim);
            release_segment_locked(victim);
        }
    }

    void release_all_blocks_noexcept() noexcept  // NOLINT(bugprone-exception-escape)
    {
        device_guard const guard(device_, std::nothrow);

        // A segment's base pointer is its first block; collect each segment once
        // (split blocks share their segment with neighbors) and each block once
        // (a block with pending events appears once per queued event). Ordered
        // sets keep teardown deterministic. Set construction/insertion can throw
        // on allocation failure, so we wrap in try-catch for the noexcept contract.
        std::set<void*>        segment_ptrs;
        std::set<cache_block*> all_blocks;

        try
        {
            auto collect = [&](cache_block* block)
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
            for (pending_event const& queued : pending_events_)
            {
                cudaEventDestroy(queued.event);
                collect(queued.block);
            }
            for (cache_block* block : quarantine_metadata_)
            {
                collect(block);
            }
            quarantine_metadata_.clear();
        }
        catch (...)  // NOLINT(bugprone-empty-catch)
        {
            // Allocation failure during set ops; continue anyway to clean up.
        }

        pending_events_.clear();
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
    // Written under mutex_, read lock-free by bytes_cached_now() (plan 3.5).
    std::atomic<size_t> bytes_cached_{0};
    std::atomic<size_t> peak_bytes_cached_{0};
    // Budget reserved for in-flight alloc_segment_unlocked() driver calls, not
    // yet reflected in stats_.bytes_reserved. See reserved_would_exceed_locked().
    size_t pending_reserved_bytes_{0};
    // Failures counted while mutex_ is held; flushed to cleanup_diagnostic after
    // the lock is released (see deferred_failure_flush).
    std::atomic<size_t> deferred_failures_{0};

    // Recursive, matching upstream: free-memory callbacks run under the lock and
    // may re-enter this allocator to free memory.
    mutable std::recursive_mutex mutex_;
    // Declared before the containers that allocate from them (destroyed after).
    node_pool                    free_pool_nodes_;
    node_pool                    live_map_nodes_;
    block_pool                   small_blocks_{true, free_pool_nodes_};
    block_pool                   large_blocks_{false, free_pool_nodes_};
    // Live allocations by pointer; free blocks live in the pool sets and blocks
    // with outstanding cross-stream events live in the event queues.
    std::unordered_map<
        void*,
        cache_block*,
        std::hash<void*>,
        std::equal_to<void*>,
        pool_allocator<std::pair<void* const, cache_block*>>>
        allocated_blocks_{
            0,
            std::hash<void*>{},
            std::equal_to<void*>{},
            pool_allocator<std::pair<void* const, cache_block*>>(&live_map_nodes_)};
    // Outstanding cross-stream events in submission order (per-stream order is
    // completion order); capacity is retained so steady state does not allocate.
    struct pending_event
    {
        cudaEvent_t  event;
        cache_block* block;
        cudaStream_t stream;
    };
    std::vector<pending_event> pending_events_;
    std::vector<cudaEvent_t>   event_pool_;
    // Quarantined blocks whose last event has already completed (or whose event
    // recording failed entirely): the GPU memory is permanently withheld, but
    // the cache_block metadata must be freed at teardown to satisfy LSan.
    std::vector<cache_block*> quarantine_metadata_;
    std::vector<cuda_caching_allocator::free_memory_callback> free_memory_callbacks_;
    memory_map<void*, raw_segment>                            driver_segments_;
    std::atomic<int64_t>                                      registration_counter_global_{0};
    unified_cache_stats                                       stats_;
    gpu_memory_history                                        history_;
    bool                                                      expandable_segments_{false};
    block_freelist                                            block_freelist_;
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
    bool                owns_live_allocation(void const*) const { return false; }
    void                add_free_memory_callback(const cuda_caching_allocator::free_memory_callback&) {}
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
    size_t              bytes_allocated_now()      const noexcept { return 0; }
    size_t              peak_bytes_allocated_now() const noexcept { return 0; }
    size_t              bytes_reserved_now()       const noexcept { return 0; }
    size_t              peak_bytes_reserved_now()  const noexcept { return 0; }
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

namespace
{
// The public API is opaque (void*); the GPU Impl works in the vendor stream type.
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
using impl_stream_t = cudaStream_t;
#else
using impl_stream_t = cuda_caching_allocator::stream_type;
#endif
impl_stream_t native_stream(cuda_caching_allocator::stream_type stream) noexcept
{
    return static_cast<impl_stream_t>(stream);
}
}  // namespace

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
    // cppcheck-suppress syntaxError
    if MEMORY_UNLIKELY (size == 0)
    {
        return nullptr;
    }
    return impl_->allocate(size, native_stream(stream));
}

void cuda_caching_allocator::deallocate(void* ptr, size_t size, stream_type stream)
{
    impl_->deallocate(ptr, size, native_stream(stream));
}

void cuda_caching_allocator::deallocate_with_stream_lookup(void* ptr, size_t nbytes) noexcept
{
    impl_->deallocate_with_stream_lookup(ptr, nbytes);
}

bool cuda_caching_allocator::owns_live_allocation(void const* ptr) const
{
    return impl_->owns_live_allocation(ptr);
}

void cuda_caching_allocator::record_stream(void* ptr, stream_type stream)
{
    impl_->record_stream(ptr, native_stream(stream));
}

void cuda_caching_allocator::add_free_memory_callback(const free_memory_callback& callback)
{
    impl_->add_free_memory_callback(callback);
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

void cuda_caching_allocator::set_expandable_segments(bool enabled)
{
    impl_->set_expandable_segments(enabled);
}

bool cuda_caching_allocator::expandable_segments() const
{
    return impl_->expandable_segments();
}

void cuda_caching_allocator::set_memory_fraction(double fraction)
{
    impl_->set_memory_fraction(fraction);
}

double cuda_caching_allocator::memory_fraction() const
{
    return impl_->memory_fraction();
}

void cuda_caching_allocator::reset_peak_stats()
{
    impl_->reset_peak_stats();
}

size_t cuda_caching_allocator::device_total_memory() const
{
    return impl_->device_total_memory();
}

unified_cache_stats cuda_caching_allocator::stats() const
{
    return impl_->stats();
}

size_t cuda_caching_allocator::bytes_allocated_now()      const noexcept { return impl_->bytes_allocated_now(); }
size_t cuda_caching_allocator::peak_bytes_allocated_now() const noexcept { return impl_->peak_bytes_allocated_now(); }
size_t cuda_caching_allocator::bytes_reserved_now()       const noexcept { return impl_->bytes_reserved_now(); }
size_t cuda_caching_allocator::peak_bytes_reserved_now()  const noexcept { return impl_->peak_bytes_reserved_now(); }
size_t cuda_caching_allocator::bytes_cached_now()         const noexcept { return impl_->bytes_cached_now(); }
size_t cuda_caching_allocator::peak_bytes_cached_now()    const noexcept { return impl_->peak_bytes_cached_now(); }

void cuda_caching_allocator::record_memory_history(bool enabled, size_t max_entries)
{
    impl_->record_memory_history(enabled, max_entries);
}

gpu_memory_snapshot cuda_caching_allocator::snapshot()
{
    return impl_->snapshot();
}

int cuda_caching_allocator::device() const
{
    return impl_->device();
}

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
namespace
{
// Process-lifetime per-device registry (plan §6.3, P3.1).
// Each slot is initialized at most once; the resulting allocator pointer is
// stored with release semantics so any subsequent acquire load sees the fully
// constructed object without holding a mutex.  The allocators themselves are
// intentional leaks — destroying them at process exit would race with
// static-storage destructors that may still be using them.
std::once_flag                          s_device_once[kMaxDevices];
std::atomic<cuda_caching_allocator*>    s_device_cache[kMaxDevices]{};
}  // namespace

cuda_caching_allocator& caching_allocator_for_device(int device_index)
{
    if (device_index < 0 || device_index >= kMaxDevices)
    {
        throw std::out_of_range(
            "caching_allocator_for_device: device_index " +
            std::to_string(device_index) + " out of range [0, " +
            std::to_string(kMaxDevices) + ")");
    }
    // Fast path: no lock on the warm path — every call after the first for
    // this device returns here (0 registry locks, plan §6.1).
    cuda_caching_allocator* p = s_device_cache[device_index].load(std::memory_order_acquire);
    if (p != nullptr)
    {
        return *p;
    }
    // Slow path (first call for this device): initialize exactly once.
    std::call_once(s_device_once[device_index], [device_index]() {
        // new is intentional: see comment above.
        auto* a = new cuda_caching_allocator(device_index);
        s_device_cache[device_index].store(a, std::memory_order_release);
    });
    return *s_device_cache[device_index].load(std::memory_order_acquire);
}

void shutdown()
{
    // Release cached (unreferenced) segments back to the driver in
    // device-index order so multi-device teardown is deterministic (§6.3).
    for (int i = 0; i < kMaxDevices; ++i)
    {
        cuda_caching_allocator* p = s_device_cache[i].load(std::memory_order_acquire);
        if (p != nullptr)
        {
            p->empty_cache();
        }
    }
}
#endif
}  // namespace gpu
}  // namespace memory
