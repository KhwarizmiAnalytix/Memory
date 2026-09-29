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

#include <cstddef>      // for size_t, ptrdiff_t
#include <cstdint>      // for uintptr_t
#include <cstring>      // for memcpy
#include <exception>    // for bad_alloc
#include <limits>       // for numeric_limits
#include <stdexcept>    // for invalid_argument, overflow_error
#include <string>       // for runtime_error messages
#include <type_traits>  // for is_same_v

#include "common/copy_token.h"         // for copy_token
#include "common/device.h"            // for device_enum
#include "common/execution_context.h" // for execution_context
#include "common/memory_macros.h"     // MEMORY_ALIGNMENT, MEMORY_DELETE_CLASS, MEMORY_FORCE_INLINE
#include "helper/memory_allocator.h"  // for cpu::memory_allocator

// GPU caching allocator (CUDA, HIP, or Metal — compile-time exclusive).
// Unified registry: gpu::caching_allocator_for_device(i).
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL

#include "gpu/caching_allocator.h"

#endif

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
#include "gpu/caching_allocator_config.h"  // for caching_config::kMinBlockSize
#endif
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/device_guard.h"
#include "gpu/gpu_runtime.h"
#endif

namespace memory
{
constexpr bool is_gpu_device(device_enum device_type)
{
    return device_type == device_enum::CUDA || device_type == device_enum::HIP ||
           device_type == device_enum::METAL;
}

/** True when @p device_type is served by the compiled GPU caching allocator. */
constexpr bool is_active_gpu_device(device_enum device_type)
{
#if MEMORY_HAS_CUDA
    return device_type == device_enum::CUDA;
#elif MEMORY_HAS_HIP
    return device_type == device_enum::HIP;
#elif MEMORY_HAS_METAL
    return device_type == device_enum::METAL;
#else
    (void)device_type;
    return false;
#endif
}

constexpr bool has_gpu_support()
{
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
    return true;
#else
    return false;
#endif
}

/**
 * @brief `count * elem_size` with overflow checked, throwing std::overflow_error
 *        instead of silently wrapping.
 *
 * `allocate()`/`copy()` previously computed `n * scalar_size` unchecked, so an
 * oversized `n` could wrap to a small byte count that then succeeds and
 * underallocates. This guards every element-count-to-byte-count conversion on
 * the allocation and copy paths.
 */
MEMORY_FORCE_INLINE constexpr std::size_t checked_byte_count(
    std::size_t count, std::size_t elem_size)
{
    if (elem_size != 0 && count > std::numeric_limits<std::size_t>::max() / elem_size)
    {
        throw std::overflow_error(
            "memory::allocator: element count * element size overflows size_t");
    }
    return count * elem_size;
}

constexpr std::size_t optimal_alignment(device_enum device_type)
{
    switch (device_type)
    {
    case device_enum::CPU:
        return MEMORY_ALIGNMENT;
    case device_enum::CUDA:
    case device_enum::HIP:
    case device_enum::METAL:
        return 256;
    default:
        return 32;
    }
}

/**
 * @brief Unified memory allocator supporting both CPU and GPU memory management
 *
 * Allocation strategy per device:
 * - CPU: direct calls into cpu::memory_allocator (mimalloc / TBB / platform).
 * - CUDA/HIP: gpu::caching_allocator_for_device (PyTorch-style segment cache;
 *   HIP uses the same Impl via gpu/gpu_runtime.h).
 * - Metal: same registry name → metal_caching_allocator (Shared MTLBuffers).
 *
 * @tparam T The type of elements to allocate
 * @tparam alignment Memory alignment requirement in bytes
 */
template <class T, std::size_t alignment = MEMORY_ALIGNMENT>
struct allocator
{
    MEMORY_DELETE_CLASS(allocator)

public:
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
    using stream_t = cudaStream_t;
#else
    using stream_t = void*;
#endif

    static constexpr size_type scalar_size    = sizeof(value_type);
    static constexpr size_type alignment_size = alignment / scalar_size;
    static constexpr size_type alignment_mask = alignment_size - 1;

    /**
     * @brief Allocate memory from execution context (preferred API)
     */
    MEMORY_FORCE_INLINE static pointer allocate(size_type n, execution_context ctx)
    {
        return allocate(n, ctx.device_type, ctx.device_index, ctx.stream);
    }

    /**
     * @brief Allocate memory on the specified device (backward compatible)
     */
    MEMORY_FORCE_INLINE static pointer allocate(
        size_type   n,
        device_enum type         = device_enum::CPU,
        int         device_index = 0,
        stream_t    stream       = nullptr)
    {
        if (n == 0)
        {
            return nullptr;
        }

        pointer         ptr    = nullptr;
        size_type const nbytes = checked_byte_count(n, scalar_size);

        if (type == device_enum::CPU)
        {
            (void)stream;
            ptr = static_cast<pointer>(memory::cpu::memory_allocator::allocate(nbytes, alignment));
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        else if (is_active_gpu_device(type))
        {
#if MEMORY_HAS_METAL
            if constexpr (std::is_same_v<T, double>)
            {
                throw std::invalid_argument(
                    "Metal backend does not support double precision (no fp64 on Apple "
                    "GPU hardware); use device_enum::CPU for double tensors.");
            }
#endif
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
            // The GPU segment cache rounds every block to a kMinBlockSize
            // multiple and does not thread a caller alignment through block
            // splitting/reuse; it cannot guarantee more than that. Reject
            // requests it cannot satisfy rather than silently under-aligning.
            if constexpr (alignment > gpu::caching_config::kMinBlockSize)
            {
                throw std::invalid_argument(
                    "allocator<T, alignment>: GPU caching allocator guarantees at most "
                    "kMinBlockSize-byte alignment; requested alignment exceeds it");
            }
#endif
            ptr = static_cast<pointer>(
                gpu::caching_allocator_for_device(device_index).allocate(nbytes, stream));
        }
#endif
        else
        {
            // With no GPU backend compiled in, is_active_gpu_device() is constexpr-false, so
            // this is the only reachable branch for a non-CPU device type. The #if above (not
            // just guarding this branch's body) keeps the two throwing branches from becoming
            // identical clones when GPU is off, which is otherwise a bugprone-branch-clone hit.
            throw std::invalid_argument("Unsupported device type for allocation");
        }

        if (ptr == nullptr)
        {
            throw std::bad_alloc();
        }
        return ptr;
    }

    /**
     * @brief Free memory allocated on the specified device
     */
    MEMORY_FORCE_INLINE static void free(
        pointer&    ptr,
        device_enum type         = device_enum::CPU,
        int         device_index = 0,
        size_type   count        = 0,
        stream_t    stream       = nullptr)
    {
        (void)count;
        if (ptr == nullptr)
        {
            return;
        }

        if (type == device_enum::CPU)
        {
            (void)stream;
            memory::cpu::memory_allocator::free(ptr);
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        else if (is_active_gpu_device(type))
        {
            gpu::caching_allocator_for_device(device_index).deallocate(ptr, 0, stream);
        }
#endif
        else
        {
            throw std::invalid_argument("Unsupported device type for deallocation");
        }

        ptr = nullptr;
    }

    /**
     * @brief Release unused cached GPU segments (torch.cuda.empty_cache)
     */
    MEMORY_FORCE_INLINE static void empty_cache(int device_index = 0)
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        gpu::empty_cache(device_index);
#else
        (void)device_index;
#endif
    }

    MEMORY_FORCE_INLINE static size_type memory_allocated(int device_index = 0)
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        return gpu::memory_allocated(device_index);
#else
        (void)device_index;
        return 0;
#endif
    }

    MEMORY_FORCE_INLINE static size_type max_memory_allocated(int device_index = 0)
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        return gpu::max_memory_allocated(device_index);
#else
        (void)device_index;
        return 0;
#endif
    }

    MEMORY_FORCE_INLINE static size_type memory_reserved(int device_index = 0)
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        return gpu::memory_reserved(device_index);
#else
        (void)device_index;
        return 0;
#endif
    }

    MEMORY_FORCE_INLINE static size_type max_memory_reserved(int device_index = 0)
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        return gpu::max_memory_reserved(device_index);
#else
        (void)device_index;
        return 0;
#endif
    }

    MEMORY_FORCE_INLINE static void reset_peak_memory_stats(int device_index = 0)
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        gpu::reset_peak_memory_stats(device_index);
#else
        (void)device_index;
#endif
    }

    MEMORY_FORCE_INLINE static void set_memory_fraction(double fraction, int device_index = 0)
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        gpu::set_memory_fraction(fraction, device_index);
#else
        (void)fraction;
        (void)device_index;
#endif
    }

    /**
     * @brief Record that @p ptr is used on @p stream (CUDA/HIP caching allocator).
     *
     * No-op for CPU and Metal. Matches PyTorch recordStream: the block is not
     * reused until @p stream completes.
     *
     * @p stream == nullptr is CUDA/HIP's own spelling for the default stream,
     * not "no stream" — it must still be forwarded. A block allocated on a
     * non-default stream and then used on the default stream is a real
     * cross-stream use; short-circuiting here on a null stream used to drop
     * that use silently. `cuda_caching_allocator::record_stream` already
     * compares against the block's own allocation stream and no-ops when they
     * match, so forwarding unconditionally is safe for the same-stream case.
     */
    MEMORY_FORCE_INLINE static void record_stream(
        pointer     ptr,  // cppcheck-suppress constParameterPointer
        device_enum type,
        int         device_index = 0,
        stream_t    stream       = nullptr)  // cppcheck-suppress constParameterPointer
    {
        if (ptr == nullptr)
        {
            return;
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
        if (is_active_gpu_device(type))
        {
            gpu::caching_allocator_for_device(device_index).record_stream(ptr, stream);
        }
#else
        (void)type;
        (void)device_index;
        (void)stream;
#endif
    }

    /**
     * @brief Copy memory between device spaces
     */
    MEMORY_FORCE_INLINE static void copy(
        const_pointer from,
        size_type     n,
        pointer       to,
        device_enum   from_type  = device_enum::CPU,
        device_enum   to_type    = device_enum::CPU,
        int           from_index = 0,
        int           to_index   = 0,
        stream_t      stream     = nullptr)
    {
        if (from == nullptr || to == nullptr || n == 0)
        {
            return;
        }

        const auto nbytes = checked_byte_count(n, scalar_size);

        if (from_type == device_enum::CPU && to_type == device_enum::CPU)
        {
            std::memcpy(to, from, nbytes);
            return;
        }

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (from_type == device_enum::CUDA || to_type == device_enum::CUDA ||
            from_type == device_enum::HIP || to_type == device_enum::HIP)
        {
            // Register stream uses on GPU endpoints BEFORE enqueuing the copy.
            // This closes the window between submission and registration: a
            // deallocation arriving between cudaMemcpyAsync and record_stream
            // could reclaim a block that the in-flight copy still references.
            // null stream is the legacy default CUDA stream — a valid stream
            // identity, not "no stream" (see cuda_caching_allocator::record_stream).
            // copy_sync passes stream=nullptr and is caller-responsible for blocking;
            // copy_async always passes a non-null stream.
            if (from_type == device_enum::CUDA || from_type == device_enum::HIP)
            {
                record_stream(const_cast<pointer>(from), from_type, from_index, stream);
            }
            if (to_type == device_enum::CUDA || to_type == device_enum::HIP)
            {
                record_stream(to, to_type, to_index, stream);
            }

            cudaError_t result = cudaSuccess;
            if ((from_type == device_enum::CUDA || from_type == device_enum::HIP) &&
                (to_type == device_enum::CUDA || to_type == device_enum::HIP) &&
                from_index != to_index)
            {
                result = cudaMemcpyPeerAsync(
                    to, to_index, from, from_index, nbytes,
                    stream != nullptr ? static_cast<cudaStream_t>(stream)
                                      : static_cast<cudaStream_t>(nullptr));
            }
            else
            {
                cudaMemcpyKind copy_kind;
                if (from_type == device_enum::CPU &&
                    (to_type == device_enum::CUDA || to_type == device_enum::HIP))
                {
                    copy_kind = cudaMemcpyHostToDevice;
                }
                else if (
                    (from_type == device_enum::CUDA || from_type == device_enum::HIP) &&
                    to_type == device_enum::CPU)
                {
                    copy_kind = cudaMemcpyDeviceToHost;
                }
                else if (
                    (from_type == device_enum::CUDA || from_type == device_enum::HIP) &&
                    (to_type == device_enum::CUDA || to_type == device_enum::HIP))
                {
                    copy_kind = cudaMemcpyDeviceToDevice;
                }
                else
                {
                    throw std::invalid_argument(
                        "Unsupported GPU device combination for memory copy");
                }

                int const gpu_index = (to_type == device_enum::CUDA || to_type == device_enum::HIP)
                                          ? to_index
                                          : from_index;
                gpu::device_guard const guard(gpu_index);
                // Always use the async form so enqueuing is non-blocking.
                // stream == nullptr means the legacy default CUDA stream (0).
                result = cudaMemcpyAsync(
                    to, from, nbytes, copy_kind,
                    stream != nullptr ? static_cast<cudaStream_t>(stream)
                                      : static_cast<cudaStream_t>(nullptr));
            }
            if (result != cudaSuccess)
            {
                throw std::runtime_error(
                    "GPU memory copy failed: " + std::string(cudaGetErrorString(result)));
            }
            return;
        }
#elif MEMORY_HAS_METAL
        (void)from_index;
        (void)to_index;
        (void)stream;
        // Shared-storage MTLBuffers are host-addressable — all METAL sides are memcpy.
        if (from_type == device_enum::METAL || to_type == device_enum::METAL)
        {
            std::memcpy(to, from, nbytes);
            return;
        }
#endif

        throw std::invalid_argument("Unsupported device combination for memory copy");
    }

    // --- Explicit sync / async copy helpers (Order 3) ---

    // copy_sync: blocking cross-device copy. Never returns until the transfer
    // is complete.  Equivalent to copy(..., stream=nullptr).
    MEMORY_FORCE_INLINE static void copy_sync(
        const_pointer from,
        size_type     n,
        pointer       to,
        device_enum   from_type  = device_enum::CPU,
        device_enum   to_type    = device_enum::CPU,
        int           from_index = 0,
        int           to_index   = 0)
    {
        copy(from, n, to, from_type, to_type, from_index, to_index, nullptr);
    }

    // copy_async: enqueue a non-blocking copy on @p stream and return a
    // copy_token.  Both GPU endpoints have record_stream called BEFORE the
    // copy is submitted, so the caching allocator defers their reuse until
    // the stream catches up regardless of whether the caller holds the token.
    // record_stream requires the pointer to be a live allocation from this
    // caching allocator; interior or foreign GPU pointers are caller-managed
    // (record_stream will CHECK-fail if the base is not found in the cache).
    // For pageable CPU endpoints the caller must wait() the token before
    // accessing the host buffer again.
    MEMORY_FORCE_INLINE static copy_token copy_async(
        const_pointer from,
        size_type     n,
        pointer       to,
        stream_t      stream,
        device_enum   from_type  = device_enum::CPU,
        device_enum   to_type    = device_enum::CPU,
        int           from_index = 0,
        int           to_index   = 0)
    {
        copy(from, n, to, from_type, to_type, from_index, to_index, stream);
        // Build a context that identifies which device/stream to wait on.
        // Use is_gpu_device (enum-based) rather than is_active_gpu_device
        // (compile-time backend check) so the selection is based on whether the
        // endpoint IS a GPU device, not on which backend happens to be compiled in.
        device_enum gpu_dev = (is_gpu_device(to_type) ? to_type : from_type);
        int         gpu_idx = (is_gpu_device(to_type) ? to_index : from_index);
        execution_context ctx;
        ctx.device_type  = gpu_dev;
        ctx.device_index = gpu_idx;
        ctx.stream       = stream;
        return copy_token(ctx);
    }

    MEMORY_FORCE_INLINE static size_type first_aligned(const_pointer array, size_type size)
    {
        if constexpr ((alignment % scalar_size) != 0)
        {
            return size;
        }

        if ((reinterpret_cast<std::uintptr_t>(array) & (scalar_size - 1)) != 0U)
        {
            return size;
        }

        size_type const first =
            (alignment_size -
             ((reinterpret_cast<std::uintptr_t>(array) / scalar_size) & alignment_mask)) &
            alignment_mask;
        return (first < size) ? first : size;
    }

    MEMORY_FORCE_INLINE static size_type last_aligned(
        size_type aligned_start, size_type size, size_type simd_stride)
    {
        return aligned_start + (((size - aligned_start) / simd_stride) * simd_stride);
    }
};

}  // namespace memory
