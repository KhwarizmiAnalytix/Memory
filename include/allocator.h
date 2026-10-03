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
#include "common/retained_ptr.h"      // for retained_ptr
#include "common/retained_operation_service.h"
#include "helper/memory_allocator.h"  // for cpu::memory_allocator

#include "common/transfer.h"  // byte copy router, token driver operations

// GPU caching allocator (CUDA, HIP, or Metal — compile-time exclusive), reached
// through backend-neutral entry points so no vendor runtime header is included.
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
#include "gpu/caching_allocator_config.h"  // for caching_config::kMinBlockSize
#include "gpu/gpu_dispatch.h"
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

    using stream_t = stream_handle_t;

    static constexpr size_type scalar_size    = sizeof(value_type);
    static constexpr size_type alignment_bytes = alignment;
    static constexpr size_type alignment_size  = alignment / scalar_size;
    static constexpr size_type alignment_mask  = alignment_size - 1;

    /**
     * @brief Allocate memory from execution context (preferred API)
     */
    MEMORY_FORCE_INLINE static pointer allocate(size_type n, execution_context ctx)
    {
        return allocate(n, ctx.device_type(), ctx.device_index(), ctx.stream);
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
            ptr = static_cast<pointer>(gpu::allocate_device_bytes(nbytes, device_index, stream));
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
            gpu::free_device_bytes(ptr, device_index, stream);
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
            gpu::record_stream_use(ptr, device_index, stream);
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
        copy_impl<true>(from, n, to, from_type, to_type, from_index, to_index, stream);
    }

    // Endpoint kinds a copy can touch in this build: host memory and the one
    // compiled GPU backend (CUDA and HIP are interchangeable spellings).
    static constexpr bool is_copy_endpoint(device_enum type)
    {
        if (type == device_enum::CPU)
        {
            return true;
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        return type == device_enum::CUDA || type == device_enum::HIP;
#elif MEMORY_HAS_METAL
        return type == device_enum::METAL;
#else
        return false;
#endif
    }

    // Pre-submission validation (plan §5.1/§5.2, task 1.5). Throws before any
    // resource is acquired or any work is submitted: a zero count is a no-op
    // (the caller returns early), positive counts reject null endpoints, byte
    // overflow and unsupported backend combinations.
    MEMORY_FORCE_INLINE static void validate_copy(
        const_pointer from, size_type n, const_pointer to, device_enum from_type, device_enum to_type)
    {
        if (n == 0)
        {
            return;
        }
        if (from == nullptr || to == nullptr)
        {
            throw std::invalid_argument("memory copy: null endpoint with a non-zero element count");
        }
        (void)checked_byte_count(n, scalar_size);
        if (!is_copy_endpoint(from_type) || !is_copy_endpoint(to_type))
        {
            throw std::invalid_argument("Unsupported device combination for memory copy");
        }
    }

    // @p submitted (optional) is set to true immediately before the driver is
    // asked to move data. A throw with *submitted == false means nothing was
    // started (setup failure); with true, work may be in flight and the caller
    // must prove completion before treating the endpoints as safe.
    template <bool track_gpu_streams>
    MEMORY_FORCE_INLINE static void copy_impl(
        const_pointer from,
        size_type     n,
        pointer       to,
        device_enum   from_type,
        device_enum   to_type,
        int           from_index,
        int           to_index,
        stream_t      stream,
        bool*         submitted = nullptr)
    {
        if (n == 0)
        {
            return;
        }
        validate_copy(from, n, to, from_type, to_type);

        const auto nbytes = checked_byte_count(n, scalar_size);

        if (from_type == device_enum::CPU && to_type == device_enum::CPU)
        {
            if (submitted != nullptr)
            {
                *submitted = true;
            }
            std::memcpy(to, from, nbytes);
            return;
        }

        // Everything that touches a GPU endpoint goes through the byte router
        // (src/transfer.cpp): stream-use registration, peer vs host/device copy
        // kinds and the driver calls live there, not in this header.
        detail::route_copy_bytes(
            from,
            to,
            nbytes,
            device{from_type, static_cast<std::int16_t>(from_index)},
            device{to_type, static_cast<std::int16_t>(to_index)},
            stream,
            track_gpu_streams,
            submitted);
    }

    // --- Explicit sync / async copy helpers (Order 3) ---

    // copy_sync: blocking cross-device copy. Never returns until the transfer
    // is complete.  Equivalent to copy(..., stream=nullptr).
    // Synchronous copy: blocks until transfer is complete and visible.
    // For CPU↔CPU: uses memcpy (synchronous by nature)
    // For GPU transfers: submits async work, waits for completion before returning
    // Guarantees: After return, destination is stable and visible to consumers
    MEMORY_FORCE_INLINE static void copy_sync(
        const_pointer from,
        size_type     n,
        pointer       to,
        device_enum   from_type  = device_enum::CPU,
        device_enum   to_type    = device_enum::CPU,
        int           from_index = 0,
        int           to_index   = 0)
    {
        // CPU↔CPU copy is inherently synchronous
        if (from_type == device_enum::CPU && to_type == device_enum::CPU)
        {
            copy(from, n, to, from_type, to_type, from_index, to_index, nullptr);
            return;
        }

        // GPU transfers: use async infrastructure with explicit wait
        // Use nullptr stream (legacy default CUDA stream) for sync behavior
        copy_token token = copy_async(from, n, to, nullptr, from_type, to_type,
                                      from_index, to_index);
        token.wait();  // Block until transfer completes
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
        return copy_async_impl<true>(
            from, n, to, stream, from_type, to_type, from_index, to_index, {}, false);
    }

    // Three phases (plan §5.3):
    //   1. Validate: throws before anything is acquired (nothing to roll back).
    //   2. Reserve: token state, event and (for retained copies) service
    //      admission. A failure here leaves nothing submitted; what was reserved
    //      is released by RAII or cancel(), exactly once.
    //   3. Submit: if the driver was never asked to move data the reservation is
    //      cancelled without waiting on any stream. Once it was, completion is
    //      proven on the submitting stream or the operation is quarantined;
    //      safety is never inferred from the error code.
    template <bool track_gpu_streams>
    MEMORY_FORCE_INLINE static copy_token copy_async_impl(
        const_pointer from,
        size_type     n,
        pointer       to,
        stream_t      stream,
        device_enum   from_type,
        device_enum   to_type,
        int           from_index,
        int           to_index,
        std::shared_ptr<void> retained,
        bool          register_with_service)
    {
        // Phase 1.
        if (n == 0)
        {
            return copy_token{};  // zero-count copy is a no-op; payload released on return
        }
        validate_copy(from, n, to, from_type, to_type);

        // Phase 2. Build a context that identifies which device/stream to wait on.
        // Use is_gpu_device (enum-based) rather than is_active_gpu_device
        // (compile-time backend check) so the selection is based on whether the
        // endpoint IS a GPU device, not on which backend happens to be compiled in.
        device_enum gpu_dev = (is_gpu_device(to_type) ? to_type : from_type);
        int         gpu_idx = (is_gpu_device(to_type) ? to_index : from_index);
        execution_context ctx;
        ctx.dev.type  = gpu_dev;
        ctx.dev.index = gpu_idx;
        ctx.stream       = stream;
        copy_token token(ctx);
        if (retained)
        {
            detail::copy_token_access::set_retained(token, std::move(retained));
        }
        token.prepare_event();  // throws: nothing admitted or submitted
        bool admitted = false;
        if (register_with_service)
        {
            admitted = retained_operation_service::instance().enqueue(token);  // throws: not admitted
        }

        // Phase 3.
        bool submitted = false;
        try
        {
            copy_impl<track_gpu_streams>(
                from, n, to, from_type, to_type, from_index, to_index, stream, &submitted);
            token.record_event();
        }
        catch (...)
        {
            bool const proven_idle = !submitted || !ctx.is_gpu() || detail::stream_proven_idle(ctx);
            if (proven_idle)
            {
                // Nothing is in flight: restore admission once and release the payload.
                if (admitted)
                {
                    (void)retained_operation_service::instance().cancel(token);
                }
                detail::copy_token_access::complete(token);
            }
            else
            {
                // Possibly in flight and unproven: keep owners, quarantine.
                detail::copy_token_access::fail(token);
                if (admitted)
                {
                    (void)retained_operation_service::instance().quarantine(token);
                }
            }
            throw;
        }
        return token;
    }

    // --- Phase 3: Retained storage and adoption (§3.3, 3.4) ---

    // allocate_adopted: Take ownership of foreign memory and wrap in retained_ptr.
    // Assigns unique allocation_id; calls deleter on destruction.
    // Supported deleters: nullptr (no-op), lambda, std::function.
    // Throws std::invalid_argument if ptr is null or count is zero.
    MEMORY_FORCE_INLINE static retained_ptr<T> allocate_adopted(
        pointer                                            ptr,
        size_type                                          count,
        execution_context                                  ctx,
        std::function<void(T*, size_t, execution_context const&)> deleter)
    {
        if (ptr == nullptr || count == 0)
        {
            throw std::invalid_argument(
                "allocate_adopted: ptr and count must be non-null and non-zero");
        }

        if (!deleter)
        {
            throw std::invalid_argument(
                "allocate_adopted: an explicit deleter is required; "
                "pass a no-op lambda to adopt without taking ownership");
        }

        return retained_ptr<T>::adopt(ptr, count, ctx, std::move(deleter));
    }

    // copy_async_retained: Enqueue a non-blocking copy using retained storage.
    // Both endpoints are kept alive through operation completion via reference counting.
    // Caller may drop retained_ptr instances; async operation holds references.
    // Returns copy_token for completion monitoring.
    MEMORY_FORCE_INLINE static copy_token copy_async_retained(
        retained_ptr<T> const& from,
        retained_ptr<T> const& to,
        stream_t               stream = nullptr)
    {
        if (from.empty() || to.empty())
        {
            throw std::invalid_argument("copy_async_retained: source and destination must be non-empty");
        }

        if (from.size() != to.size())
        {
            throw std::invalid_argument(
                "copy_async_retained: source and destination extents must match");
        }

        struct retained_holder
        {
            retained_ptr<T> from;
            retained_ptr<T> to;
        };
        auto holder = std::make_shared<retained_holder>(retained_holder{from, to});
        return copy_async_impl<false>(
            from.data(), from.size(), to.data(), stream,
            from.ctx().device_type(), to.ctx().device_type(),
            from.ctx().device_index(), to.ctx().device_index(),
            std::static_pointer_cast<void>(holder), true);
    }

    // SIMD loop peeling is not a memory concern; it moves to Vectorization (plan 4.6, task 2.7).
    [[deprecated("use the Vectorization library; this forwarder is removed next release")]]
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

    [[deprecated("use the Vectorization library; this forwarder is removed next release")]]
    MEMORY_FORCE_INLINE static size_type last_aligned(
        size_type aligned_start, size_type size, size_type simd_stride)
    {
        return aligned_start + (((size - aligned_start) / simd_stride) * simd_stride);
    }
};

}  // namespace memory
