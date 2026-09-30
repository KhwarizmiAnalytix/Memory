/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/execution_context.h"
#include "common/memory_export.h"
#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/gpu_runtime.h"
#endif

namespace memory
{

// Completion state for copy operations
enum class completion_state : std::uint8_t {
    pending = 0,   // Operation enqueued, awaiting completion
    complete = 1,  // Operation finished successfully
    failed = 2     // Operation or completion tracking failed
};

// Completion token returned by allocator<T>::copy_async.
//
// The token records the execution context (backend, device, stream) the copy
// was enqueued on and (for GPU ops) an operation-specific completion marker.
// Both GPU endpoints have record_stream called BEFORE the copy is submitted,
// so the caching allocator defers their reuse until the stream catches up
// regardless of whether the token is kept.  This guarantee holds only for
// pointers that are live allocations in the caching allocator; interior or
// foreign GPU pointers remain caller-managed (see copy_async).
//
// For pageable (non-pinned) CPU endpoints the copy_async caller is responsible
// for keeping the host buffer alive and unmodified until the copy completes.
// The token does NOT hold a reference to the host buffer — call wait() (or
// synchronize the stream manually) before touching the host memory again.
//
// A default-constructed copy_token is immediately complete (CPU no-op).
class MEMORY_VISIBILITY copy_token
{
public:
    copy_token() noexcept = default;

    explicit copy_token(execution_context ctx) noexcept : ctx_(ctx) {}

    copy_token(copy_token&&) noexcept            = default;
    copy_token& operator=(copy_token&&) noexcept = default;
    copy_token(copy_token const&) noexcept       = default;
    copy_token& operator=(copy_token const&) noexcept = default;

    // Query completion state without blocking.
    // For CPU operations, always returns 'complete'.
    // For GPU operations, queries the operation's event or stream.
    completion_state state() const noexcept
    {
        if (!ctx_.is_gpu())
        {
            return completion_state::complete;
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        // Phase 2 interim: check stream (Phase 2+ will use operation-specific events)
        cudaError_t const r = (ctx_.stream != nullptr)
                                  ? cudaStreamQuery(static_cast<cudaStream_t>(ctx_.stream))
                                  : cudaStreamQuery(nullptr);
        if (r == cudaSuccess)
        {
            return completion_state::complete;
        }
        // cudasErrorNotReady means stream is still working (not an error)
        return completion_state::pending;
#else
        return completion_state::complete;
#endif
    }

    // Returns true if the copy has already completed (or was a CPU copy).
    // Queries the operation's state without blocking.
    bool ready() const noexcept
    {
        return state() == completion_state::complete;
    }

    // Blocks until the copy completes. No-op for CPU copies.
    // Throws std::runtime_error if the operation failed.
    void wait() const
    {
        if (!ctx_.is_gpu())
        {
            return;
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (ctx_.stream != nullptr)
        {
            cudaError_t result = cudaStreamSynchronize(static_cast<cudaStream_t>(ctx_.stream));
            if (result != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("copy_token::wait() failed: ") +
                    cudaGetErrorString(result));
            }
        }
        else
        {
            cudaError_t result = cudaDeviceSynchronize();
            if (result != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("copy_token::wait() failed: ") +
                    cudaGetErrorString(result));
            }
        }
#endif
    }

    execution_context const& ctx() const noexcept { return ctx_; }

private:
    execution_context ctx_{};
};

}  // namespace memory
