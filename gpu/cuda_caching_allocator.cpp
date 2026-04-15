#include "gpu/cuda_caching_allocator.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/memory_macros.h"
#include "util/exception.h"

#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "common/memory_containers.h"

namespace memory
{
namespace gpu
{
namespace
{

#if MEMORY_HAS_CUDA
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
        std::string message = std::string(what) + ": CUDA not available";
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
        int current = 0;
        throw_on_cuda_error(cudaGetDevice(&current), "cudaGetDevice");
        prev_ = current;
        if (current != device)
        {
            throw_on_cuda_error(cudaSetDevice(device), "cudaSetDevice");
            changed_ = true;
        }
    }

    DeviceGuard(const DeviceGuard&)            = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    ~DeviceGuard()
    {
        if (changed_)
        {
            cudaSetDevice(prev_);
        }
    }

private:
    int  prev_{0};
    bool changed_{false};
};

}  // namespace

struct cuda_caching_allocator::Impl
{
    struct Block
    {
        void*  ptr  = nullptr;
        size_t size = 0;
#if MEMORY_HAS_CUDA
        cudaStream_t last_stream = nullptr;
        cudaEvent_t  event       = nullptr;
#else
        void* last_stream = nullptr;
        void* event       = nullptr;
#endif
        bool in_use           = false;
        bool event_pending    = false;
        bool in_free_list     = false;
        bool in_deferred_list = false;
    };

    Impl(int device, size_t max_cached_bytes) : device_(device), max_cached_bytes_(max_cached_bytes)
    {
        // Log info (simplified for build compatibility)

        // Validate device
#if MEMORY_HAS_CUDA
        int device_count = 0;
        throw_on_cuda_error(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        MEMORY_CHECK(  //NOLINT
            device >= 0 && device < device_count,
            "Invalid CUDA device index: {} (available: 0-{})",
            device,
            device_count - 1);
#endif
    }

    ~Impl()
    {
        std::scoped_lock const lock(mutex_);
        release_all_blocks_noexcept();
    }

    void* allocate(size_t size, cuda_caching_allocator::stream_type stream)
    {
        MEMORY_CHECK(size > 0, "cuda_caching_allocator cannot allocate zero bytes");

        // Debug log (simplified for build compatibility)

        std::scoped_lock const lock(mutex_);
        reclaim_deferred_blocks_locked();

        Block* block = find_suitable_block_locked(size);

        if (block == nullptr)
        {
            block = create_block_locked(size);
            stats_.cache_misses++;
        }
        else
        {
            cached_bytes_ -= block->size;
            stats_.bytes_cached = cached_bytes_;
            stats_.cache_hits++;
        }

        block->in_use           = true;
        block->last_stream      = stream;
        block->event_pending    = false;
        block->in_free_list     = false;
        block->in_deferred_list = false;

        bytes_in_use_ += block->size;

        // Update allocation statistics
        stats_.successful_allocations++;
        stats_.bytes_allocated += block->size;

        // Debug log (simplified for build compatibility)

        return block->ptr;
    }

    void deallocate(void* ptr, size_t /*size*/, cuda_caching_allocator::stream_type stream)
    {
        if (ptr == nullptr)
        {
            return;
        }

        // Debug log (simplified for build compatibility)

        std::scoped_lock const lock(mutex_);
        auto                   it = blocks_.find(ptr);

        MEMORY_CHECK(
            it != blocks_.end(), "cuda_caching_allocator does not own the provided pointer");

        Block* block = it->second.get();

        MEMORY_CHECK(block->in_use, "cuda_caching_allocator detected a double free");

        block->in_use = false;
        bytes_in_use_ -= block->size;

        // Update deallocation statistics
        stats_.successful_frees++;

        if (!should_cache(block->size))
        {
            // Debug log (simplified for build compatibility)
            release_block_locked(it);
            return;
        }

        cached_bytes_ += block->size;
        stats_.bytes_cached = cached_bytes_;

        auto* effective_stream = stream != nullptr ? stream : block->last_stream;

        if (effective_stream == nullptr || effective_stream == block->last_stream)
        {
            block->last_stream = effective_stream;
            insert_ready_block_locked(block);
        }
        else
        {
            record_event_locked(block, effective_stream);
        }

        trim_cache_locked();

        // Debug log (simplified for build compatibility)
    }

    void empty_cache()
    {
        std::scoped_lock const lock(mutex_);
        DeviceGuard const      guard(device_);
        reclaim_deferred_blocks_locked(true);

        while (!free_blocks_.empty())
        {
            auto   it    = free_blocks_.begin();
            Block* block = it->second;
            destroy_event(block);
            throw_on_cuda_error(cudaFree(block->ptr), "cudaFree");
            stats_.driver_frees++;
            blocks_.erase(block->ptr);
            free_blocks_.erase(it);
        }

        for (Block* block : deferred_blocks_)
        {
            destroy_event(block);
            throw_on_cuda_error(cudaFree(block->ptr), "cudaFree");
            stats_.driver_frees++;
            blocks_.erase(block->ptr);
        }
        deferred_blocks_.clear();

        cached_bytes_       = 0;
        stats_.bytes_cached = 0;
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
        // Create a copy of the atomic stats structure
        unified_cache_stats const stats_copy(stats_);
        return stats_copy;
    }

    int device() const { return device_; }

private:
    using BlockMap = memory_map<void*, std::unique_ptr<Block>>;
    using FreeList = std::multimap<size_t, Block*>;

    bool should_cache(size_t size) const
    {
        return max_cached_bytes_ == std::numeric_limits<size_t>::max() || size <= max_cached_bytes_;
    }

    Block* find_suitable_block_locked(size_t size)
    {
        auto it = free_blocks_.lower_bound(size);
        if (it == free_blocks_.end())
        {
            return nullptr;
        }
        Block* block = it->second;
        free_blocks_.erase(it);
        block->in_free_list = false;
        return block;
    }

    Block* create_block_locked(size_t size)
    {
        DeviceGuard const guard(device_);
        void*             ptr    = nullptr;
        cudaError_t const result = cudaMalloc(&ptr, size);
        if (result != cudaSuccess)
        {
            throw std::bad_alloc();
        }

        auto block         = std::make_unique<Block>();
        block->ptr         = ptr;
        block->size        = size;
        block->last_stream = nullptr;
        block->in_use      = false;

        Block* raw = block.get();  //NOLINT
        blocks_.emplace(ptr, std::move(block));

        stats_.driver_allocations++;
        return raw;
    }

    void insert_ready_block_locked(Block* block)
    {
        if (block->in_free_list)
        {
            return;
        }
        free_blocks_.emplace(block->size, block);
        block->in_free_list = true;
    }

    void record_event_locked(Block* block, cudaStream_t stream)
    {
        DeviceGuard const guard(device_);
        if (block->event == nullptr)
        {
            throw_on_cuda_error(
                cudaEventCreateWithFlags(&block->event, cudaEventDisableTiming),
                "cudaEventCreateWithFlags");
        }
        throw_on_cuda_error(cudaEventRecord(block->event, stream), "cudaEventRecord");
        block->event_pending = true;
        block->last_stream   = stream;
        if (!block->in_deferred_list)
        {
            deferred_blocks_.push_back(block);
            block->in_deferred_list = true;
        }
    }

    void reclaim_deferred_blocks_locked(bool force = false)
    {
        if (deferred_blocks_.empty())
        {
            return;
        }

        DeviceGuard const guard(device_);
        size_t            index = 0;
        while (index < deferred_blocks_.size())
        {
            Block* block = deferred_blocks_[index];
            bool   ready = false;

            if (!block->event_pending)
            {
                ready = true;
            }
            else if (force)
            {
                throw_on_cuda_error(cudaEventSynchronize(block->event), "cudaEventSynchronize");
                ready = true;
            }
            else
            {
                cudaError_t const status = cudaEventQuery(block->event);
                if (status == cudaSuccess)
                {
                    ready = true;
                }
                else if (status == cudaErrorNotReady)
                {
                    ++index;
                    continue;
                }
                else
                {
                    throw_on_cuda_error(status, "cudaEventQuery");
                }
            }

            if (ready)
            {
                block->event_pending    = false;
                block->in_deferred_list = false;
                deferred_blocks_[index] = deferred_blocks_.back();
                deferred_blocks_.pop_back();
                insert_ready_block_locked(block);
            }
        }
    }

    void trim_cache_locked()
    {
        if (max_cached_bytes_ == std::numeric_limits<size_t>::max())
        {
            return;
        }

        DeviceGuard const guard(device_);
        while (cached_bytes_ > max_cached_bytes_ && !free_blocks_.empty())
        {
            auto   it    = std::prev(free_blocks_.end());
            Block* block = it->second;
            cached_bytes_ -= block->size;
            stats_.bytes_cached = cached_bytes_;
            destroy_event(block);
            throw_on_cuda_error(cudaFree(block->ptr), "cudaFree");
            stats_.driver_frees++;
            blocks_.erase(block->ptr);
            free_blocks_.erase(it);
        }
    }

    void release_block_locked(BlockMap::iterator it)
    {
        Block* block = it->second.get();

        DeviceGuard const guard(device_);
        destroy_event(block);
        throw_on_cuda_error(cudaFree(block->ptr), "cudaFree");
        stats_.driver_frees++;

        blocks_.erase(it);
    }

    static void destroy_event(Block* block)
    {
        if (block->event != nullptr)
        {
            cudaEventDestroy(block->event);
            block->event = nullptr;
        }
        block->event_pending    = false;
        block->in_deferred_list = false;
    }

    void release_all_blocks_noexcept()
    {
        DeviceGuard const guard(device_);
        for (auto& entry : blocks_)
        {
            Block* block = entry.second.get();
            if (block->event != nullptr)
            {
                cudaEventDestroy(block->event);
                block->event = nullptr;
            }
            if (block->ptr != nullptr)
            {
                cudaFree(block->ptr);
            }
        }
        blocks_.clear();
        free_blocks_.clear();
        deferred_blocks_.clear();
        cached_bytes_ = 0;
        bytes_in_use_ = 0;
    }

    int    device_;
    size_t max_cached_bytes_;
    size_t cached_bytes_{0};
    size_t bytes_in_use_{0};

    mutable std::mutex  mutex_;
    BlockMap            blocks_;
    FreeList            free_blocks_;
    std::vector<Block*> deferred_blocks_;
    unified_cache_stats stats_;
};

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
{
    impl_->deallocate(ptr, size, stream);
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
}  // namespace gpu
}  // namespace memory
