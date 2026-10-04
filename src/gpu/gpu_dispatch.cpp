/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

// Backend-neutral GPU entry points (see gpu/gpu_dispatch.h). Compiled for
// CUDA, HIP and Metal; empty when no GPU backend is active.

#include "gpu/gpu_dispatch.h"

#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL

#include "gpu/caching_allocator.h"

namespace memory::gpu
{

namespace
{
// CUDA/HIP caches take the vendor's stream pointer type; Metal takes void*.
caching_allocator::stream_type backend_stream(stream_handle_t stream) noexcept
{
    return static_cast<caching_allocator::stream_type>(stream);
}
}  // namespace

void* allocate_device_bytes(std::size_t nbytes, int device_index, stream_handle_t stream)
{
    return caching_allocator_for_device(device_index).allocate(nbytes, backend_stream(stream));
}

void free_device_bytes(void* ptr, int device_index, stream_handle_t stream)
{
    caching_allocator_for_device(device_index).deallocate(ptr, 0, backend_stream(stream));
}

void record_stream_use(void* ptr, int device_index, stream_handle_t stream)
{
    caching_allocator_for_device(device_index).record_stream(ptr, backend_stream(stream));
}

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
bool owns_live_allocation(void const* ptr, int device_index)
{
    return caching_allocator_for_device(device_index).owns_live_allocation(ptr);
}
#endif

void empty_cache(int device_index)
{
    caching_allocator_for_device(device_index).empty_cache();
}

// O(1) lock-free basic stats (plan 6.1, P3.5): direct atomic reads, no mutex.
std::size_t memory_allocated(int device_index)
{
    return caching_allocator_for_device(device_index).bytes_allocated_now();
}

std::size_t max_memory_allocated(int device_index)
{
    return caching_allocator_for_device(device_index).peak_bytes_allocated_now();
}

std::size_t memory_reserved(int device_index)
{
    return caching_allocator_for_device(device_index).bytes_reserved_now();
}

std::size_t max_memory_reserved(int device_index)
{
    return caching_allocator_for_device(device_index).peak_bytes_reserved_now();
}

void reset_peak_memory_stats(int device_index)
{
    caching_allocator_for_device(device_index).reset_peak_stats();
}

void set_memory_fraction(double fraction, int device_index)
{
    caching_allocator_for_device(device_index).set_memory_fraction(fraction);
}

}  // namespace memory::gpu

#endif
