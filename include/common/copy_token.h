/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstddef>

#include "common/execution_context.h"
#include "common/memory_export.h"
#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/gpu_runtime.h"
#endif

namespace memory
{

// Completion token returned by allocator<T>::copy_async.
//
// The token records the execution context (backend, device, stream) the copy
// was enqueued on.  Both GPU endpoints have record_stream called BEFORE the
// copy is submitted, so the caching allocator defers their reuse until the
// stream catches up regardless of whether the token is kept.  This guarantee
// holds only for pointers that are live allocations in the caching allocator;
// interior or foreign GPU pointers remain caller-managed (see copy_async).
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
    copy_token(copy_token const&)                = default;
    copy_token& operator=(copy_token const&)     = default;

    // Blocks until the copy's stream has completed all work enqueued before
    // and including the copy_async call.  No-op for CPU copies.
    void wait() const noexcept
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (!ctx_.is_gpu())
        {
            return;
        }
        if (ctx_.stream != nullptr)
        {
            cudaStreamSynchronize(static_cast<cudaStream_t>(ctx_.stream));
        }
        else
        {
            cudaDeviceSynchronize();
        }
#endif
    }

    // Returns true if the copy has already completed (or was a CPU copy).
    bool ready() const noexcept
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (!ctx_.is_gpu())
        {
            return true;
        }
        cudaError_t const r = (ctx_.stream != nullptr)
                                  ? cudaStreamQuery(static_cast<cudaStream_t>(ctx_.stream))
                                  : cudaStreamQuery(nullptr);
        return r == cudaSuccess;
#else
        return true;
#endif
    }

    execution_context const& ctx() const noexcept { return ctx_; }

private:
    execution_context ctx_{};
};

}  // namespace memory
