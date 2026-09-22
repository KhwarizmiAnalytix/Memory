/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#pragma once

#include <cstddef>
#include <memory>

#include "common/memory_export.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/gpu_runtime.h"
#endif

namespace memory::cpu
{
struct MEMORY_VISIBILITY pinned_memory_stats
{
    std::size_t bytes_requested{0};  // Live user-requested bytes.
    std::size_t bytes_allocated{0};  // Live rounded block capacity.
    std::size_t bytes_cached{0};     // Immediately reusable capacity.
    std::size_t bytes_pending{0};    // Awaiting streams, including quarantined blocks.
    std::size_t bytes_reserved{0};   // Driver-requested bytes, including alignment padding.
    std::size_t peak_bytes_reserved{0};
    std::size_t cache_hits{0};
    std::size_t cache_misses{0};
    std::size_t driver_allocations{0};
    std::size_t driver_frees{0};
    std::size_t num_ooms{0};
    std::size_t num_errors{0};
};

/** Page-locked host buffers for CUDA/HIP transfers, with a bounded reuse cache.
 *
 * Each instance records streams on one device. Host allocations are portable
 * across runtime contexts, but streams passed here must belong to device().
 * Register every asynchronous use before deallocation; registered streams must
 * remain valid until deallocation. Null denotes the real default stream.
 * Recording protects lifetime, not producer/consumer ordering or CPU access.
 * Allocator instances must outlive their allocations. Destruction waits for
 * recorded work. Metal/CPU-only builds explicitly reject nonzero allocations.
 */
class MEMORY_VISIBILITY pinned_memory_allocator
{
public:
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
    using stream_type = cudaStream_t;
#else
    using stream_type = void*;
#endif
    static constexpr std::size_t alignment                = 64;
    static constexpr std::size_t default_max_cached_bytes = 64 * 1024 * 1024;

    MEMORY_API explicit pinned_memory_allocator(
        int device = 0, std::size_t max_cached_bytes = default_max_cached_bytes);
    MEMORY_API ~pinned_memory_allocator();
    pinned_memory_allocator(const pinned_memory_allocator&)            = delete;
    pinned_memory_allocator& operator=(const pinned_memory_allocator&) = delete;

    static constexpr bool supported() noexcept
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        return true;
#else
        return false;
#endif
    }

    /// Uninitialized, 64-byte aligned storage. Zero returns nullptr; OOM throws.
    MEMORY_API void* allocate(std::size_t bytes);
    /// Exact allocation base only. False for foreign/double frees or runtime
    /// errors. On runtime failure the allocation is consumed and quarantined,
    /// never reused.
    MEMORY_API bool deallocate(void* ptr) noexcept;
    /// Accepts a live base or interior pointer. May precede or follow submission.
    MEMORY_API void record_stream(const void* ptr, stream_type stream);

    /// Bounds-check the pinned endpoint and register the stream before
    /// enqueueing. The caller owns the device endpoint's lifetime and execution
    /// dependencies. Wait for stream completion before reading/writing the host
    /// buffer again.
    MEMORY_API void copy_to_device_async(
        void* destination, const void* source, std::size_t bytes, stream_type stream = nullptr);
    MEMORY_API void copy_from_device_async(
        void* destination, const void* source, std::size_t bytes, stream_type stream = nullptr);

    /// Poll completed transfers and trim reusable cache without waiting for work.
    MEMORY_API void poll();
    /// Wait for released buffers' recorded work, then release reusable blocks.
    /// Live allocations are unaffected. Unsafe blocks stay quarantined on errors.
    MEMORY_API void empty_cache();
    /// Limits reusable bytes only; live and pending transfers may exceed the cap.
    MEMORY_API void set_max_cached_bytes(std::size_t bytes);
    MEMORY_API std::size_t         max_cached_bytes() const;
    MEMORY_API pinned_memory_stats stats() const;
    MEMORY_API int                 device() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// One shared pinned pool per device. All callers must use the matching pool to
/// free.
MEMORY_API pinned_memory_allocator& pinned_allocator_for_device(int device = 0);
}  // namespace memory::cpu
