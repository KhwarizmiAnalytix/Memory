/*
 * Quarisma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of Quarisma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>

#include "common/memory_export.h"
#include "common/memory_macros.h"
#include "profiler/unified_memory_stats.h"

namespace memory
{
namespace gpu
{
/**
 * @brief Metal caching allocator with PyTorch CUDACachingAllocator semantics
 *
 * Ports the CUDA caching allocator's segment/block model onto shared-storage
 * MTLBuffers (Apple Silicon unified memory):
 * - Requests rounded to 512-byte multiples; small (<= 1 MiB) requests are
 *   packed into 2 MiB segments, 1-10 MiB into 20 MiB segments, larger
 *   requests rounded up to 2 MiB multiples — one newBufferWithLength per
 *   segment, many blocks per segment
 * - Oversized cached blocks are split on reuse; freed blocks coalesce with
 *   free neighbors
 * - Free pools use a single default "stream" (void*); Vectorization Metal
 *   dispatch is synchronous today, so record_stream is a documented no-op
 * - On driver allocation failure the cache is flushed and the allocation
 *   retried once before throwing std::bad_alloc
 * - Optional max_cached_bytes cap with largest-first trimming of whole
 *   segments (default: unlimited)
 *
 * Host pointers returned by allocate() are buffer.contents + block offset and
 * are directly host-dereferenceable (MTLResourceStorageModeShared). Mid-segment
 * pointers resolve via resolve_live_allocation() for kernel binding.
 *
 * @note Compiled only when MEMORY_HAS_METAL=1
 */
class MEMORY_VISIBILITY metal_caching_allocator
{
public:
    using stream_type = void*;

    /**
     * @brief Construct a Metal caching allocator
     * @param device Device index (only 0 is supported — system default device)
     * @param max_cached_bytes Maximum bytes to cache (default: unlimited)
     * @throws std::runtime_error if device is invalid or Metal is unavailable
     */
    MEMORY_API explicit metal_caching_allocator(
        int device = 0, size_t max_cached_bytes = std::numeric_limits<size_t>::max());

    MEMORY_API ~metal_caching_allocator();

    /**
     * @brief Allocate GPU memory with caching
     *
     * Returned pointers are at least 256-byte aligned (Metal buffer bases plus
     * 512-byte-rounded block offsets).
     *
     * @param size Number of bytes to allocate
     * @param stream Unused for v1 (kept for API parity with CUDA); always nullptr
     * @return Host-visible pointer into a shared MTLBuffer
     * @throws std::bad_alloc if allocation fails
     */
    MEMORY_API void* allocate(size_t size, stream_type stream = nullptr);

    /**
     * @brief Deallocate GPU memory (may cache for reuse)
     * @param ptr Pointer previously returned by allocate()
     * @param size Unused; kept for interface compatibility
     * @param stream Unused for v1
     * @throws std::invalid_argument if ptr is not owned by this allocator
     * @throws std::logic_error if double free detected
     */
    MEMORY_API void deallocate(void* ptr, size_t size, stream_type stream = nullptr);

    /**
     * @brief Record a cross-stream use (no-op for Metal v1 — sync dispatch only)
     */
    MEMORY_API void record_stream(void* ptr, stream_type stream);

    MEMORY_API void empty_cache();

    MEMORY_API void set_max_cached_bytes(size_t bytes);

    MEMORY_API size_t max_cached_bytes() const;

    using free_memory_callback = std::function<bool()>;

    MEMORY_API void add_free_memory_callback(free_memory_callback callback);

    MEMORY_API void clear_free_memory_callbacks();

    MEMORY_API unified_cache_stats stats() const;

    MEMORY_API int device() const;

    /**
     * @brief Resolve a live allocation to its owning MTLBuffer and byte offset
     *
     * Used by memory::metal::mtl_buffer_handle / mtl_buffer_offset so Vectorization
     * can bind mid-segment pointers with setBuffer:offset:.
     *
     * @param ptr Live host pointer from allocate()
     * @param handle_out Receives __bridge void* of id<MTLBuffer> (not retained)
     * @param offset_out Receives byte offset within the MTLBuffer
     * @return true if ptr is a live allocation of this allocator
     */
    MEMORY_API bool resolve_live_allocation(void* ptr, void** handle_out, size_t* offset_out) const;

    metal_caching_allocator(const metal_caching_allocator&)            = delete;
    metal_caching_allocator& operator=(const metal_caching_allocator&) = delete;
    MEMORY_API               metal_caching_allocator(metal_caching_allocator&&) noexcept;
    MEMORY_API metal_caching_allocator& operator=(metal_caching_allocator&&) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Process-wide caching allocator for a Metal device index
 *
 * Lazily creates one metal_caching_allocator per device (only index 0 is
 * valid today — MTLCreateSystemDefaultDevice). Lives in the Memory library
 * so all linkers share one registry.
 */
MEMORY_API metal_caching_allocator& metal_caching_allocator_for_device(int device_index);

}  // namespace gpu
}  // namespace memory
