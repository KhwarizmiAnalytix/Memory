/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

// Backend-neutral GPU entry points for the storage and transfer layers (plan
// 4.1, task 2.6). Declarations only: this header includes no vendor runtime
// header, so allocator<T>, data_ptr<T> and friends can call the per-device
// caching allocator without the consumer's include path carrying CUDA/HIP.
//
// Defined in src/gpu/gpu_dispatch.cpp for whichever backend is compiled in
// (CUDA, HIP or Metal). Callers guard use with MEMORY_HAS_CUDA/HIP/METAL.

#include <cstddef>

#include "common/execution_context.h"
#include "common/memory_export.h"

namespace memory::gpu
{

/// Allocate @p nbytes from the per-device cache on @p stream. Throws what the
/// cache throws (bad_alloc on exhaustion). Never returns null.
MEMORY_API void* allocate_device_bytes(
    std::size_t nbytes, int device_index, stream_handle_t stream);

/// Return a block to the per-device cache. @p stream is the caller's view of the
/// allocation stream; the cache validates ownership and throws on a foreign pointer.
MEMORY_API void free_device_bytes(void* ptr, int device_index, stream_handle_t stream);

/// Record a cross-stream use of a live allocation (torch recordStream).
MEMORY_API void record_stream_use(void* ptr, int device_index, stream_handle_t stream);

// torch.cuda.memory analogues over the per-device cache.
MEMORY_API void        empty_cache(int device_index = 0);
MEMORY_API std::size_t memory_allocated(int device_index = 0);
MEMORY_API std::size_t max_memory_allocated(int device_index = 0);
MEMORY_API std::size_t memory_reserved(int device_index = 0);
MEMORY_API std::size_t max_memory_reserved(int device_index = 0);
MEMORY_API void        reset_peak_memory_stats(int device_index = 0);
MEMORY_API void        set_memory_fraction(double fraction, int device_index = 0);

}  // namespace memory::gpu
