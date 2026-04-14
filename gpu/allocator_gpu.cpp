/*
 * Quarisma: High-Performance Computational Library
 * Copyright 2025 Quarisma Contributors
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "gpu/allocator_gpu.h"

#include <algorithm>
#include <memory>
#include <sstream>

#include "logger.h"
#include "helper/memory_allocator.h"
#if MEMORY_HAS_NATIVE_PROFILER && 0
#include "native/tracing/traceme.h"
#endif
#include "util/exception.h"

#if MEMORY_HAS_CUDA
#include <cuda.h>  // For CUDA Driver API
#include <cuda_runtime.h>
#endif

#if MEMORY_HAS_HIP
#include <hip/hip_runtime.h>
#endif

namespace memory
{
namespace gpu
{

//==============================================================================
// basic_gpu_allocator Implementation
//==============================================================================

basic_gpu_allocator::basic_gpu_allocator(
    int                         device_id,
    const std::vector<Visitor>& alloc_visitors,
    const std::vector<Visitor>& free_visitors,
    MEMORY_UNUSED int         numa_node)
    : sub_allocator(alloc_visitors, free_visitors), device_id_(device_id)
{
    // Verify device is accessible by attempting to set it
    if (!memory::gpu::memory_allocator::set_device(device_id))
    {
        MEMORY_THROW(
            "Failed to set GPU device {}: device may not exist or be accessible", device_id);
    }

    LOGGING_LOG_INFO("Created GPU sub-allocator for device {}", device_id);
}

basic_gpu_allocator::~basic_gpu_allocator()
{
    std::scoped_lock const lock(stats_mutex_);
    if (total_allocated_.load() > 0)
    {
        LOGGING_LOG_WARNING(
            "GPU sub-allocator destroyed with {} bytes still allocated", total_allocated_.load());
    }
}

void* basic_gpu_allocator::Alloc(
    MEMORY_UNUSED size_t alignment, size_t num_bytes, size_t* bytes_received)
{
#if MEMORY_HAS_NATIVE_PROFILER && 0
    memory::traceme const traceme("basic_gpu_allocator::Alloc");
#endif

    void* ptr       = nullptr;
    *bytes_received = num_bytes;
    if (num_bytes > 0)
    {
        ptr = memory::gpu::memory_allocator::allocate(num_bytes, device_id_);

        if (ptr != nullptr)
        {
            // Update statistics
            size_t const new_total    = total_allocated_.fetch_add(num_bytes) + num_bytes;
            size_t       current_peak = peak_allocated_.load();
            while (new_total > current_peak &&
                   !peak_allocated_.compare_exchange_weak(current_peak, new_total))
            {
                // Retry if another thread updated peak_allocated_
            }

            LOGGING_LOG_INFO_DEBUG(
                "GPU allocated {} bytes at {} on device {}", num_bytes, ptr, device_id_);

            // Call allocation visitors for monitoring
            VisitAlloc(ptr, device_id_, num_bytes);
        }
        else
        {
            *bytes_received = 0;
        }
    }
    return ptr;
}

void basic_gpu_allocator::Free(void* ptr, size_t num_bytes)
{
#if MEMORY_HAS_NATIVE_PROFILER && 0
    memory::traceme const traceme("basic_gpu_allocator::Free");
#endif

    if (num_bytes > 0)
    {
        VisitFree(ptr, device_id_, num_bytes);
        memory::gpu::memory_allocator::free(ptr, num_bytes, device_id_);

        // Update statistics
        total_allocated_.fetch_sub(num_bytes);

        LOGGING_LOG_INFO_DEBUG(
            "GPU freed {} bytes at {} on device {}", num_bytes, ptr, device_id_);
    }
}

//==============================================================================
// allocator_gpu Implementation
//==============================================================================

allocator_gpu::allocator_gpu(int device_id, const Options& options, std::string name)
    : device_id_(device_id), options_(options), name_(std::move(name))
{
#if MEMORY_HAS_NATIVE_PROFILER && 0
    memory::traceme const traceme("allocator_gpu::allocator_gpu");
#endif

    // Verify device is accessible by attempting to set it
    if (!memory::gpu::memory_allocator::set_device(device_id))
    {
        MEMORY_THROW(
            "Failed to set GPU device {}: device may not exist or be accessible", device_id);
    }

    // Get the allocation method name for logging
    const char* method_name;
    auto        strategy = memory::gpu::memory_allocator::get_allocation_strategy();
    switch (strategy)
    {
    case memory::gpu::memory_allocator::allocation_strategy::SYNC:
        method_name = "SYNC";
        break;
    case memory::gpu::memory_allocator::allocation_strategy::ASYNC:
        method_name = "ASYNC";
        break;
    case memory::gpu::memory_allocator::allocation_strategy::POOL_ASYNC:
        method_name = "POOL_ASYNC";
        break;
    default:
        method_name = "UNKNOWN";
        break;
    }

    LOGGING_LOG_INFO(
        "Created GPU allocator '{}' with {} method on device {}", name_, method_name, device_id);
}

allocator_gpu::~allocator_gpu()
{
#if MEMORY_HAS_NATIVE_PROFILER && 0
    memory::traceme const traceme("allocator_gpu::~allocator_gpu");
#endif

    // Check for memory leaks
    size_t total_allocated = total_allocated_.load();
    if (total_allocated > 0)
    {
        LOGGING_LOG_WARNING(
            "GPU allocator '{}' destroyed with {} bytes still allocated", name_, total_allocated);
    }

    LOGGING_LOG_INFO("Destroyed GPU allocator '{}'", name_);
}

void* allocator_gpu::allocate_gpu_memory(size_t num_bytes, void* stream) const
{
    // Use the new gpu::memory_allocator with stream and memory pool support
    void* gpu_stream = (stream != nullptr) ? stream : options_.gpu_stream;
    return memory::gpu::memory_allocator::allocate(
        num_bytes, device_id_, gpu_stream, options_.memory_pool);
}

void allocator_gpu::deallocate_gpu_memory(void* ptr, size_t num_bytes, void* stream) const
{
    if (ptr == nullptr)
    {
        return;
    }

    // Use the new gpu::memory_allocator with stream support
    void* gpu_stream = (stream != nullptr) ? stream : options_.gpu_stream;
    memory::gpu::memory_allocator::free(ptr, num_bytes, device_id_, gpu_stream);
}

bool allocator_gpu::set_device_context() const
{
    return memory::gpu::memory_allocator::set_device(device_id_);
}

void* allocator_gpu::allocate_raw(size_t alignment, size_t num_bytes)
{
    return allocate_raw(alignment, num_bytes, allocation_attributes{});
}

void* allocator_gpu::allocate_raw(
    MEMORY_UNUSED size_t                       alignment,
    size_t                                       num_bytes,
    MEMORY_UNUSED const allocation_attributes& allocation_attr)
{
#if MEMORY_HAS_NATIVE_PROFILER && 0
    memory::traceme const traceme("allocator_gpu::allocate_raw");
#endif

    if (num_bytes == 0)
    {
        return nullptr;
    }

    if (!set_device_context())
    {
        return nullptr;
    }

    std::scoped_lock const lock(device_mutex_);

    // Allocate memory using direct GPU API
    void* ptr = allocate_gpu_memory(num_bytes, nullptr);  //NOLINT

    if (ptr != nullptr && options_.enable_statistics)
    {
        // Update statistics
        total_allocated_.fetch_add(num_bytes);
        allocation_count_.fetch_add(1);

        // Update peak allocation
        size_t const new_total    = total_allocated_.load();
        size_t       current_peak = peak_allocated_.load();
        while (new_total > current_peak &&
               !peak_allocated_.compare_exchange_weak(current_peak, new_total))
        {
            // Retry if another thread updated peak_allocated_
        }

        LOGGING_LOG_INFO_DEBUG(
            "GPU allocated {} bytes at {} on device {}", num_bytes, ptr, device_id_);
    }

    return ptr;
}

void allocator_gpu::deallocate_raw(void* ptr)
{
#if MEMORY_HAS_NATIVE_PROFILER && 0
    memory::traceme const traceme("allocator_gpu::deallocate_raw");
#endif

    if (ptr == nullptr)
    {
        return;
    }

    if (!set_device_context())
    {
        return;
    }

    std::scoped_lock const lock(device_mutex_);

    // Note: We don't have the original size for direct GPU API calls
    // This is a limitation of the direct approach vs backend allocators
    // For statistics, we'll need to track allocations separately if needed
    deallocate_gpu_memory(ptr, 0, nullptr);

    if (options_.enable_statistics)
    {
        deallocation_count_.fetch_add(1);
        LOGGING_LOG_INFO_DEBUG("GPU freed memory at {} on device {}", ptr, device_id_);
    }
}

bool allocator_gpu::tracks_allocation_sizes() const noexcept
{
    // Direct GPU allocation doesn't track individual allocation sizes
    // This would require additional bookkeeping which we've eliminated for performance
    return false;
}

size_t allocator_gpu::RequestedSize(MEMORY_UNUSED const void* ptr) const noexcept
{
    // Direct GPU allocation doesn't track requested sizes

    return 0;
}

size_t allocator_gpu::AllocatedSize(MEMORY_UNUSED const void* ptr) const noexcept
{
    return 0;
}

int64_t allocator_gpu::AllocationId(MEMORY_UNUSED const void* ptr) const
{
    return 0;
}

std::optional<allocator_stats> allocator_gpu::GetStats() const
{
    if (!options_.enable_statistics)
    {
        return std::nullopt;
    }

    allocator_stats stats;
    stats.bytes_in_use       = total_allocated_.load();
    stats.peak_bytes_in_use  = peak_allocated_.load();
    stats.largest_alloc_size = 0;  // Not tracked in direct mode
    stats.num_allocs         = allocation_count_.load();
    stats.num_deallocs       = deallocation_count_.load();

    return stats;
}

bool allocator_gpu::ClearStats()
{
    if (!options_.enable_statistics)
    {
        return false;
    }

    total_allocated_.store(0);
    peak_allocated_.store(0);
    allocation_count_.store(0);
    deallocation_count_.store(0);

    return true;
}

//==============================================================================
// Factory Functions
//==============================================================================

std::unique_ptr<allocator_gpu> create_gpu_allocator(int device_id, const std::string& name)
{
    allocator_gpu::Options options;
    options.enable_statistics = true;

    return std::make_unique<allocator_gpu>(device_id, options, name);
}

std::unique_ptr<allocator_gpu> create_gpu_allocator(
    int device_id, const allocator_gpu::Options& options, const std::string& name)
{
    return std::make_unique<allocator_gpu>(device_id, options, name);
}

}  // namespace gpu
}  // namespace memory
