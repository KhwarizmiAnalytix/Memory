/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "common/storage_handle.h"

#include <cstddef>
#include <stdexcept>

#include "common/cleanup_diagnostic.h"
#include "common/device.h"
#include "common/execution_context.h"
#include "helper/memory_allocator.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
#include "gpu/caching_allocator.h"
#endif

namespace memory
{

namespace
{
// CPU free callback wired into every storage_handle created by allocate_bytes
// for device_enum::CPU.  ctx is unused (always nullptr for CPU allocations).
void cpu_free_fn(void* /*ctx*/, void* ptr, std::size_t nbytes) noexcept
{
    memory::cpu::memory_allocator::free(ptr, nbytes);
}

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
// GPU free callback wired into every GPU storage_handle created by allocate_bytes.
// ctx is the cache pointer (obtained at allocate time); this deleter can be called
// by the handle's destructor without requiring the owner to know the stream.
// The cache looks up the allocation stream from the cache_block and deallocates
// using that stream (plan §2.10, R1).
void gpu_free_fn(void* cache_ctx, void* ptr, std::size_t nbytes) noexcept
{
    static_cast<gpu::caching_allocator*>(cache_ctx)->deallocate_with_stream_lookup(ptr, nbytes);
}
#endif

// Guard: is `t` the GPU backend that was compiled in?
constexpr bool is_active_gpu(device_enum t) noexcept
{
#if MEMORY_HAS_CUDA
    return t == device_enum::CUDA;
#elif MEMORY_HAS_HIP
    return t == device_enum::HIP;
#elif MEMORY_HAS_METAL
    return t == device_enum::METAL;
#else
    (void)t;
    return false;
#endif
}
}  // namespace

storage_handle allocate_bytes(std::size_t nbytes, std::size_t alignment,
                               execution_context ctx)
{
    if (nbytes == 0)
    {
        // Zero-size: no allocation, but still assign a unique identity so callers
        // that track allocation_id (e.g. data_ptr, TestPhase3Identity) get a valid ID.
        device const dev{ctx.device_type, static_cast<std::int16_t>(ctx.device_index)};
        return storage_handle(nullptr, 0, nullptr, nullptr, dev, next_allocation_id());
    }

    device const dev{ctx.device_type, static_cast<std::int16_t>(ctx.device_index)};

    if (ctx.device_type == device_enum::CPU)
    {
        void* ptr = cpu::memory_allocator::allocate(nbytes, alignment);
        if (!ptr)
        {
            throw std::bad_alloc{};
        }
        return storage_handle(ptr, nbytes, &cpu_free_fn, nullptr, dev,
                              next_allocation_id());
    }

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
    if (is_active_gpu(ctx.device_type))
    {
        // Obtain the per-device cache ONCE; store its address in ctx_ so the
        // free path can reach it directly without a registry lookup
        // (plan §4.3, P2.3 gate: 0 lookups on free).
        gpu::caching_allocator& cache =
            gpu::caching_allocator_for_device(ctx.device_index);
        void* ptr = cache.allocate(nbytes, ctx.stream);
        if (!ptr)
        {
            throw std::bad_alloc{};
        }
        // GPU free: deleter_ = gpu_free_fn (looks up the allocation stream
        // from the cache_block and deallocates). The cache pointer is stored
        // in ctx_ so the deleter can reach the cache without a registry lookup
        // (plan §2.10, R1: GPU handle frees itself).
        return storage_handle(ptr, nbytes, &gpu_free_fn,
                              static_cast<void*>(&cache),
                              dev, next_allocation_id());
    }
#endif

    throw std::invalid_argument("allocate_bytes: unsupported device type");
}

void free_gpu_with_stream(void* cache_ctx, void* ptr, std::size_t nbytes,
                           stream_handle_t stream) noexcept
{
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
    try
    {
        static_cast<gpu::caching_allocator*>(cache_ctx)->deallocate(ptr, nbytes, stream);
    }
    catch (...)
    {
        cleanup_diagnostic::record_failure();
    }
#else
    (void)cache_ctx;
    (void)ptr;
    (void)nbytes;
    (void)stream;
#endif
}

}  // namespace memory
