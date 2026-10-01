/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "common/execution_context.h"
#include "common/memory_export.h"
#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/gpu_runtime.h"
#include "gpu/device_guard.h"
#endif

namespace memory
{

// Completion state for copy operations
enum class completion_state : std::uint8_t
{
    pending  = 0,  // Operation enqueued, awaiting completion
    complete = 1,  // Operation finished successfully
    failed   = 2   // Operation or completion tracking failed
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

    explicit copy_token(execution_context ctx) : state_(std::make_shared<shared_state>(ctx)) {}

    copy_token(copy_token&&) noexcept                 = default;
    copy_token& operator=(copy_token&&) noexcept      = default;
    copy_token(copy_token const&) noexcept            = default;
    copy_token& operator=(copy_token const&) noexcept = default;

    // Query completion state without blocking.
    // For CPU operations, always returns 'complete'.
    // For allocator-submitted GPU operations, queries the operation's event. Tokens
    // constructed directly without an event retain stream-query compatibility.
    completion_state state() const noexcept
    {
        if (state_ && state_->canceled.load(std::memory_order_acquire))
        {
            return completion_state::complete;
        }
        if (state_ && state_->forced_failed.load(std::memory_order_acquire))
        {
            return completion_state::failed;
        }
        if (!state_ || !state_->ctx.is_gpu())
        {
            return completion_state::complete;
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        gpu::device_guard guard(state_->ctx.device_index, std::nothrow);
        if (state_->event_created)
        {
            if (!state_->event_recorded.load(std::memory_order_acquire))
            {
                return completion_state::pending;
            }
            cudaError_t const r = cudaEventQuery(state_->event);
            if (r == cudaSuccess)
            {
                return completion_state::complete;
            }
            return r == cudaErrorNotReady ? completion_state::pending : completion_state::failed;
        }

        // Compatibility for externally constructed tokens without an event.
        cudaError_t const r = cudaStreamQuery(static_cast<cudaStream_t>(state_->ctx.stream));
        if (r == cudaSuccess)
        {
            return completion_state::complete;
        }
        if (r == cudaErrorNotReady)
        {
            // Clear sticky error state so repeated queries work
            (void)cudaGetLastError();
            return completion_state::pending;
        }
        // Any other error is a real failure
        return completion_state::failed;
#else
        return completion_state::complete;
#endif
    }

    // Returns true if the copy has already completed (or was a CPU copy).
    // Queries the operation's state without blocking.
    bool ready() const noexcept { return state() == completion_state::complete; }

    // Blocks until the copy completes. No-op for CPU copies.
    // Throws std::runtime_error if the operation failed.
    void wait() const
    {
        if (!state_ || !state_->ctx.is_gpu())
        {
            return;
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        gpu::device_guard guard(state_->ctx.device_index, std::nothrow);
        if (state_->event_created)
        {
            if (!state_->event_recorded.load(std::memory_order_acquire))
            {
                throw std::runtime_error("copy_token::wait() called before operation submission");
            }
            cudaError_t result = cudaEventSynchronize(state_->event);
            if (result != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("copy_token::wait() failed: ") + cudaGetErrorString(result));
            }
        }
        else
        {
            cudaError_t result = (state_->ctx.stream != nullptr)
                ? cudaStreamSynchronize(static_cast<cudaStream_t>(state_->ctx.stream))
                : cudaDeviceSynchronize();
            if (result != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("copy_token::wait() failed: ") + cudaGetErrorString(result));
            }
        }
#endif
    }

    execution_context const& ctx() const noexcept
    {
        static constexpr execution_context cpu_context{};
        return state_ ? state_->ctx : cpu_context;
    }

    // Store retained pointers so they stay alive until token completion.
    // The payload is opaque (a holder for retained_ptr copies).
    void set_retained(std::shared_ptr<void> retained)
    {
        ensure_state();
        state_->retained = std::move(retained);
    }

    void prepare_event()
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (!state_ || !state_->ctx.is_gpu())
        {
            return;
        }
        gpu::device_guard guard(state_->ctx.device_index);
        cudaError_t const result = cudaEventCreateWithFlags(&state_->event, cudaEventDisableTiming);
        if (result != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("copy_token: event creation failed: ") + cudaGetErrorString(result));
        }
        state_->event_created = true;
#endif
    }

    void record_event()
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (state_ && state_->event_created)
        {
            gpu::device_guard guard(state_->ctx.device_index);
            cudaError_t const result = cudaEventRecord(
                state_->event, static_cast<cudaStream_t>(state_->ctx.stream));
            if (result != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("copy_token: event recording failed: ") + cudaGetErrorString(result));
            }
            state_->event_recorded.store(true, std::memory_order_release);
        }
#endif
    }

    void mark_complete() noexcept
    {
        if (state_)
        {
            state_->canceled.store(true, std::memory_order_release);
            state_->retained.reset();
        }
    }

    void mark_failed() noexcept
    {
        if (state_)
        {
            state_->forced_failed.store(true, std::memory_order_release);
        }
    }

private:
    struct shared_state
    {
        explicit shared_state(execution_context value) noexcept : ctx(value) {}
        ~shared_state()
        {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
            if (event_created && event != nullptr)
            {
                gpu::device_guard guard(ctx.device_index, std::nothrow);
                (void)cudaEventDestroy(event);
            }
#endif
        }

        execution_context             ctx{};
        std::shared_ptr<void>         retained;
    #if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        cudaEvent_t                   event{};
    #endif
        bool                          event_created{false};
        std::atomic<bool>             event_recorded{false};
        std::atomic<bool>             canceled{false};
        std::atomic<bool>             forced_failed{false};
    };

    void ensure_state()
    {
        if (!state_)
        {
            state_ = std::make_shared<shared_state>(execution_context::cpu());
        }
    }

    std::shared_ptr<shared_state> state_;
};

}  // namespace memory
