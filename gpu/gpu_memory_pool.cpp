#include "gpu/gpu_memory_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>

#include "common/memory_macros.h"
#include "logger.h"
#include "util/exception.h"
#include "util/flat_hash.h"

#if MEMORY_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace memory
{
namespace gpu
{

namespace
{

/**
 * @brief Custom hash function for void pointers
 *
 * Provides a simple hash implementation for void* that doesn't rely on
 * std::__hash_memory which may not be available in all libc++ versions.
 */
struct void_ptr_hash
{
    std::size_t operator()(void* ptr) const noexcept
    {
        return static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(ptr));
    }
};

/**
 * @brief Internal implementation of GPU memory pool
 *
 * This class provides the concrete implementation of the GPU memory pool
 * interface, managing memory blocks across different size classes and
 * device types with thread-safe operations.
 */
class gpu_memory_pool_impl : public gpu_memory_pool
{
private:
    /** @brief Pool configuration */
    gpu_memory_pool_config config_;

    /** @brief Mutex for thread-safe operations */
    mutable std::mutex mutex_;

    /** @brief Map of size class to cached blocks */
    logging::logging_map<size_t, std::vector<gpu_memory_block>> cached_blocks_;

    /** @brief Map of active allocations for tracking (using custom hash for void*) */
    logging::logging_map<void*, gpu_memory_block, void_ptr_hash> active_allocations_;

    /** @brief Current total allocated bytes */
    std::atomic<size_t> allocated_bytes_{0};

    /** @brief Peak allocated bytes */
    std::atomic<size_t> peak_allocated_bytes_{0};

    /** @brief Total number of allocations performed */
    std::atomic<size_t> total_allocations_{0};

    /** @brief Total number of deallocations performed */
    std::atomic<size_t> total_deallocations_{0};

    /** @brief Total number of cache hits */
    std::atomic<size_t> cache_hits_{0};

    /** @brief Total number of cache misses */
    std::atomic<size_t> cache_misses_{0};

    /** @brief Total bytes allocated (cumulative) */
    std::atomic<size_t> total_bytes_allocated_{0};

    /** @brief Total bytes deallocated (cumulative) */
    std::atomic<size_t> total_bytes_deallocated_{0};

    /**
     * @brief Calculate the appropriate size class for a given size
     * @param size Requested size in bytes
     * @return Size class that can accommodate the requested size
     */
    size_t calculate_size_class(size_t size) const
    {
        if (size <= config_.min_block_size)
        {
            return config_.min_block_size;
        }

        if (size >= config_.max_block_size)
        {
            return config_.max_block_size;
        }

        // Calculate the size class using geometric progression
        double const log_ratio = std::log(static_cast<double>(size) / config_.min_block_size) /
                                 std::log(config_.block_growth_factor);
        auto const class_index = static_cast<size_t>(std::ceil(log_ratio));

        return static_cast<size_t>(
            config_.min_block_size * std::pow(config_.block_growth_factor, class_index));
    }

    /**
     * @brief Allocate memory directly from the GPU
     * @param size Size to allocate
     * @param device_type Device type
     * @param device_index Device index
     * @return Allocated memory block
     */
    gpu_memory_block allocate_direct(size_t size, device_enum device_type, int device_index) const
    {
        void*               ptr = nullptr;
        device_option const device(device_type, device_index);

        switch (device_type)
        {
#if MEMORY_HAS_CUDA
        case device_enum::CUDA:
        {
            // Set the device context
            cudaSetDevice(device_index);

            // Allocate aligned memory if requested
            if (config_.enable_alignment)
            {
                size_t const aligned_size =
                    ((size + config_.alignment_boundary - 1) / config_.alignment_boundary) *
                    config_.alignment_boundary;
                cudaError_t const result = cudaMalloc(&ptr, aligned_size);
                if (result != cudaSuccess)
                {
                    MEMORY_THROW(
                        "CUDA memory allocation failed: {}",
                        std::string(cudaGetErrorString(result)));
                }
                size = aligned_size;
            }
            else
            {
                cudaError_t const result = cudaMalloc(&ptr, size);
                if (result != cudaSuccess)
                {
                    MEMORY_THROW(
                        "CUDA memory allocation failed: {}",
                        std::string(cudaGetErrorString(result)));
                }
            }
            break;
        }
#endif
        default:
            MEMORY_THROW("Unsupported device type for GPU memory allocation");
        }

        if (config_.debug_mode)
        {
            LOGGING_LOG_INFO("Direct GPU allocation: {} bytes on device {}", size, device_index);
        }

        return {ptr, size, device};
    }

    /**
     * @brief Deallocate memory directly to the GPU
     * @param block Memory block to deallocate
     */
    void deallocate_direct(const gpu_memory_block& block) const
    {
        if (block.ptr == nullptr)
        {
            return;
        }

        switch (block.device.type())
        {
#if MEMORY_HAS_CUDA
        case device_enum::CUDA:
        {
            cudaSetDevice(block.device.index());
            cudaError_t const result = cudaFree(block.ptr);
            if (result != cudaSuccess && config_.debug_mode)
            {
                LOGGING_LOG_WARNING(
                    "CUDA memory deallocation warning: {}",
                    std::string(cudaGetErrorString(result)));
            }
            break;
        }
#endif
        default:
            if (config_.debug_mode)
            {
                LOGGING_LOG_WARNING("Unknown device type in deallocation");
            }
        }

        if (config_.debug_mode)
        {
            LOGGING_LOG_INFO(
                "Direct GPU deallocation: {} bytes from device {}",
                block.size,
                block.device.index());
        }
    }

public:
    explicit gpu_memory_pool_impl(const gpu_memory_pool_config& config) : config_(config)
    {
        if (config_.min_block_size == 0 || config_.max_block_size == 0)
        {
            MEMORY_THROW("Invalid block size configuration");
        }

        if (config_.min_block_size > config_.max_block_size)
        {
            MEMORY_THROW("Minimum block size cannot be larger than maximum block size");
        }

        if (config_.block_growth_factor <= 1.0)
        {
            MEMORY_THROW("Block growth factor must be greater than 1.0");
        }

        if (config_.debug_mode)
        {
            LOGGING_LOG_INFO(
                "GPU memory pool initialized with min_block={}, max_block={}, growth_factor={}",
                config_.min_block_size,
                config_.max_block_size,
                config_.block_growth_factor);
        }
    }

    ~gpu_memory_pool_impl() override
    {
        gpu_memory_pool_impl::clear_cache();

        if (config_.debug_mode && !active_allocations_.empty())
        {
            LOGGING_LOG_WARNING(
                "GPU memory pool destroyed with {} active allocations (potential memory leak)",
                active_allocations_.size());
        }
    }

    gpu_memory_block allocate(size_t size, device_enum device_type, int device_index) override
    {
        if (size == 0)
        {
            MEMORY_THROW("Cannot allocate zero bytes");
        }

        size_t const size_class = calculate_size_class(size);
        total_allocations_.fetch_add(1);

        std::scoped_lock const lock(mutex_);

        // Try to find a cached block of the appropriate size class
        auto cache_it = cached_blocks_.find(size_class);
        if (cache_it != cached_blocks_.end() && !cache_it->second.empty())
        {
            // Find a block for the correct device
            auto& blocks = cache_it->second;
            for (auto it = blocks.begin(); it != blocks.end(); ++it)
            {
                if (it->device.type() == device_type && it->device.index() == device_index)
                {
                    gpu_memory_block block = std::move(*it);
                    blocks.erase(it);

                    block.in_use.store(true);
                    block.reuse_count.fetch_add(1);

                    // Create a copy for return before moving to active_allocations_
                    gpu_memory_block result_block(block.ptr, block.size, block.device);
                    result_block.in_use.store(true);
                    result_block.reuse_count.store(block.reuse_count.load());

                    active_allocations_.emplace(block.ptr, std::move(block));

                    cache_hits_.fetch_add(1);
                    total_bytes_allocated_.fetch_add(result_block.size);

                    if (config_.debug_mode)
                    {
                        LOGGING_LOG_INFO("Cache hit: reusing block of size {}", result_block.size);
                    }

                    return result_block;
                }
            }
        }

        // No suitable cached block found, allocate new one
        cache_misses_.fetch_add(1);
        gpu_memory_block block = allocate_direct(size_class, device_type, device_index);
        block.in_use.store(true);

        // Create a copy for return before moving to active_allocations_
        gpu_memory_block result_block(block.ptr, block.size, block.device);
        result_block.in_use.store(true);
        result_block.reuse_count.store(block.reuse_count.load());

        active_allocations_.emplace(block.ptr, std::move(block));

        // Update statistics
        size_t const current_allocated =
            allocated_bytes_.fetch_add(result_block.size) + result_block.size;
        total_bytes_allocated_.fetch_add(result_block.size);
        size_t current_peak = peak_allocated_bytes_.load();
        while (current_allocated > current_peak &&
               !peak_allocated_bytes_.compare_exchange_weak(current_peak, current_allocated))
        {
            // Retry if another thread updated peak_allocated_bytes_
        }

        return result_block;
    }

    void deallocate(const gpu_memory_block& block) override
    {
        if (block.ptr == nullptr)
        {
            MEMORY_THROW("Cannot deallocate null pointer");
        }

        std::scoped_lock const lock(mutex_);

        auto it = active_allocations_.find(block.ptr);
        if (it == active_allocations_.end())
        {
            MEMORY_THROW("Attempting to deallocate unknown memory block");
        }

        gpu_memory_block cached_block = std::move(it->second);
        active_allocations_.erase(it);
        allocated_bytes_.fetch_sub(cached_block.size);
        total_deallocations_.fetch_add(1);
        total_bytes_deallocated_.fetch_add(cached_block.size);

        // Check if we should cache this block
        auto& cache_vector = cached_blocks_[cached_block.size];
        if (cache_vector.size() < config_.max_cached_blocks)
        {
            cached_block.in_use.store(false);
            const auto cached_size = cached_block.size;
            cache_vector.emplace_back(std::move(cached_block));

            if (config_.debug_mode)
            {
                LOGGING_LOG_INFO("Cached block of size {}", cached_size);
            }
        }
        else
        {
            // Cache is full, deallocate directly
            deallocate_direct(cached_block);
        }
    }

    size_t get_allocated_bytes() const override { return allocated_bytes_.load(); }

    size_t get_peak_allocated_bytes() const override { return peak_allocated_bytes_.load(); }

    size_t get_active_allocations() const override
    {
        std::scoped_lock const lock(mutex_);
        return active_allocations_.size();
    }

    void clear_cache() override
    {
        std::scoped_lock const lock(mutex_);

        for (auto& [size_class, blocks] : cached_blocks_)
        {
            for (const auto& block : blocks)
            {
                deallocate_direct(block);
            }
        }

        cached_blocks_.clear();

        if (config_.debug_mode)
        {
            LOGGING_LOG_INFO("GPU memory pool cache cleared");
        }
    }

    std::string get_memory_report() const override
    {
        std::scoped_lock const lock(mutex_);

        std::ostringstream oss;
        oss << "GPU Memory Pool Report:\n";
        oss << "  Current allocated: " << (allocated_bytes_.load() / 1024.0 / 1024.0) << " MB\n";
        oss << "  Peak allocated: " << (peak_allocated_bytes_.load() / 1024.0 / 1024.0) << " MB\n";
        oss << "  Active allocations: " << active_allocations_.size() << "\n";
        oss << "  Total allocations: " << total_allocations_.load() << "\n";
        oss << "  Cache hits: " << cache_hits_.load() << "\n";
        oss << "  Cache hit rate: " << std::fixed << std::setprecision(2)
            << (100.0 * cache_hits_.load() / std::max(total_allocations_.load(), size_t(1)))
            << "%\n";

        size_t total_cached = 0;
        for (const auto& [size_class, blocks] : cached_blocks_)
        {
            total_cached += blocks.size() * size_class;
        }
        oss << "  Cached memory: " << (total_cached / 1024.0 / 1024.0) << " MB\n";
        oss << "  Cache entries: " << cached_blocks_.size() << " size classes\n";

        return oss.str();
    }

    gpu_memory_pool_statistics get_statistics() const override
    {
        std::scoped_lock const lock(mutex_);

        gpu_memory_pool_statistics stats;

        // Basic allocation/deallocation counts
        stats.total_allocations   = total_allocations_.load();
        stats.total_deallocations = total_deallocations_.load();

        // Cache performance metrics
        stats.cache_hits   = cache_hits_.load();
        stats.cache_misses = cache_misses_.load();

        // Calculate cache hit rate
        size_t const total_requests = stats.cache_hits + stats.cache_misses;
        if (total_requests > 0)
        {
            stats.cache_hit_rate = static_cast<double>(stats.cache_hits) / total_requests;
        }
        else
        {
            stats.cache_hit_rate = 0.0;
        }

        // Memory usage statistics
        stats.total_bytes_allocated   = total_bytes_allocated_.load();
        stats.total_bytes_deallocated = total_bytes_deallocated_.load();
        stats.current_bytes_in_use    = allocated_bytes_.load();
        stats.peak_bytes_in_use       = peak_allocated_bytes_.load();
        stats.active_allocations      = active_allocations_.size();

        // Calculate cached memory
        size_t cached_memory = 0;
        for (const auto& [size_class, blocks] : cached_blocks_)
        {
            cached_memory += blocks.size() * size_class;
        }
        stats.cached_memory = cached_memory;

        return stats;
    }
};

}  // anonymous namespace

std::unique_ptr<gpu_memory_pool> gpu_memory_pool::create(const gpu_memory_pool_config& config)
{
    return std::make_unique<gpu_memory_pool_impl>(config);
}

}  // namespace gpu
}  // namespace memory
