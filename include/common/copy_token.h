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
#include "common/transfer.h"

namespace memory
{

// Completion state for copy operations
enum class completion_state : std::uint8_t
{
    pending  = 0,  // Operation enqueued, awaiting completion
    complete = 1,  // Operation finished successfully
    failed   = 2   // Operation or completion tracking failed
};

namespace detail
{
struct copy_token_access;
}

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
//
// Terminal state (plan §5.2, task 1.4): copies of a token share one operation
// state. The first terminal result (complete or failed) is published once, with
// its diagnostic context, and is stable: every copy, every thread and both
// state() and wait() observe the same result afterwards, whatever work is later
// queued on the stream. Pending is re-queried. Only the submitting code in this
// library can force a terminal state (detail::copy_token_access).
class MEMORY_VISIBILITY copy_token
{
public:
    copy_token() noexcept = default;

    explicit copy_token(execution_context ctx) : state_(std::make_shared<shared_state>(ctx)) {}

    copy_token(copy_token&&) noexcept                 = default;
    copy_token& operator=(copy_token&&) noexcept      = default;
    copy_token(copy_token const&) noexcept            = default;
    copy_token& operator=(copy_token const&) noexcept = default;

    // Query completion state without blocking or throwing.
    // For CPU operations, always returns 'complete'.
    // For allocator-submitted GPU operations, queries the operation's event.
    // Terminal states (complete, failed) are cached on the shared state: once
    // observed they are returned without a driver call on every copy.
    completion_state state() const noexcept
    {
        if (!state_)
        {
            return completion_state::complete;
        }

        // Fast path: a published terminal result.
        auto const cached = decode_state(state_->terminal.load(std::memory_order_acquire));
        if (cached != completion_state::pending)
        {
            return cached;
        }
        if (!state_->ctx.is_gpu())
        {
            return completion_state::complete;
        }

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        int const device = state_->ctx.device_index();
        if (state_->event_created)
        {
            if (!state_->event_recorded.load(std::memory_order_acquire))
            {
                return completion_state::pending;
            }
            detail::driver_result const r = detail::token_event_query(device, state_->event);
            if (r.status == detail::driver_status::ok)
            {
                settle(completion_state::complete, failure_kind::none, 0);
            }
            else if (r.status == detail::driver_status::error)
            {
                settle(completion_state::failed, failure_kind::driver_query, r.code);
            }
        }
        else
        {
            // Compatibility for externally constructed tokens without an event.
            detail::driver_result const r =
                detail::token_stream_query(device, state_->ctx.stream);
            if (r.status == detail::driver_status::ok)
            {
                settle(completion_state::complete, failure_kind::none, 0);
            }
            else if (r.status == detail::driver_status::error)
            {
                settle(completion_state::failed, failure_kind::driver_query, r.code);
            }
        }
        // Another thread may have settled first; report the published result.
        return decode_state(state_->terminal.load(std::memory_order_acquire));
#else
        return completion_state::complete;
#endif
    }

    // Returns true only for a complete operation (never for pending or failed).
    // Queries the operation's state without blocking.
    bool ready() const noexcept { return state() == completion_state::complete; }

    // Returns on complete, blocks while pending, throws std::runtime_error on
    // failed. Agrees with state(): a failed token throws on every call, and a
    // complete token never throws. No-op for CPU copies.
    void wait() const
    {
        if (!state_)
        {
            return;
        }
        auto const published = state_->terminal.load(std::memory_order_acquire);
        if (decode_state(published) == completion_state::complete)
        {
            return;
        }
        if (decode_state(published) == completion_state::failed)
        {
            throw std::runtime_error(describe_failure(published));
        }
        if (!state_->ctx.is_gpu())
        {
            return;
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        int const                  device = state_->ctx.device_index();
        detail::driver_result      result;
        if (state_->event_created)
        {
            if (!state_->event_recorded.load(std::memory_order_acquire))
            {
                throw std::runtime_error("copy_token::wait() called before operation submission");
            }
            result = detail::token_event_synchronize(device, state_->event);
        }
        else
        {
            result = detail::token_stream_synchronize(device, state_->ctx.stream);
        }
        if (result.status == detail::driver_status::ok)
        {
            settle(completion_state::complete, failure_kind::none, 0);
        }
        else
        {
            settle(completion_state::failed, failure_kind::driver_wait, result.code);
        }
        auto const final_state = state_->terminal.load(std::memory_order_acquire);
        if (decode_state(final_state) == completion_state::failed)
        {
            throw std::runtime_error(describe_failure(final_state));
        }
#endif
    }

    // True when both tokens are copies of one operation (share its state).
    bool same_operation(copy_token const& other) const noexcept
    {
        return state_ != nullptr && state_ == other.state_;
    }

    execution_context const& ctx() const noexcept
    {
        static constexpr execution_context cpu_context{};
        return state_ ? state_->ctx : cpu_context;
    }

    // Submission-side API: call before the token is shared with other threads.
    void prepare_event()
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (!state_ || !state_->ctx.is_gpu())
        {
            return;
        }
        void*                       created = nullptr;
        detail::driver_result const result  = detail::token_event_create(
            state_->ctx.device_index(), &created);
        if (result.status != detail::driver_status::ok)
        {
            throw std::runtime_error(
                std::string("copy_token: event creation failed: ") +
                detail::driver_error_string(result.code));
        }
        state_->event         = created;
        state_->event_created = true;
#endif
    }

    void record_event()
    {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        if (state_ && state_->event_created)
        {
            detail::driver_result const result = detail::token_event_record(
                state_->ctx.device_index(), state_->event, state_->ctx.stream);
            if (result.status != detail::driver_status::ok)
            {
                throw std::runtime_error(
                    std::string("copy_token: event recording failed: ") +
                    detail::driver_error_string(result.code));
            }
            state_->event_recorded.store(true, std::memory_order_release);
        }
#endif
    }

private:
    friend struct detail::copy_token_access;

    // Why a token failed; stored with the driver code as diagnostic context.
    enum class failure_kind : std::uint8_t
    {
        none = 0,
        driver_query,      // event/stream query reported an error
        driver_wait,       // event/stream synchronize reported an error
        submission_unsafe  // submission failed and completion could not be proven
    };

    // terminal word: 0 = pending; otherwise state | kind << 8 | code << 32.
    static constexpr std::uint64_t encode(completion_state s, failure_kind k, int code) noexcept
    {
        return static_cast<std::uint64_t>(s) | (static_cast<std::uint64_t>(k) << 8) |
               (static_cast<std::uint64_t>(static_cast<std::uint32_t>(code)) << 32);
    }
    static constexpr completion_state decode_state(std::uint64_t word) noexcept
    {
        return word == 0 ? completion_state::pending : static_cast<completion_state>(word & 0xFFU);
    }
    static constexpr failure_kind decode_kind(std::uint64_t word) noexcept
    {
        return static_cast<failure_kind>((word >> 8) & 0xFFU);
    }
    static constexpr int decode_code(std::uint64_t word) noexcept
    {
        return static_cast<int>(static_cast<std::uint32_t>(word >> 32));
    }

    // Publish the first terminal result; later results are ignored. Returns
    // true when this call published. Complete encodes as 1, so a terminal word
    // is never confused with pending (0).
    bool settle(completion_state s, failure_kind k, int code) const noexcept
    {
        std::uint64_t expected = 0;
        return state_->terminal.compare_exchange_strong(
            expected, encode(s, k, code), std::memory_order_acq_rel, std::memory_order_acquire);
    }

    static std::string describe_failure(std::uint64_t word)
    {
        int const   code = decode_code(word);
        std::string text;
        switch (decode_kind(word))
        {
        case failure_kind::driver_query:
            text = "copy_token: operation query failed";
            break;
        case failure_kind::driver_wait:
            text = "copy_token::wait() failed";
            break;
        case failure_kind::submission_unsafe:
            return "copy_token: submission failed and completion could not be proven; "
                   "endpoints are quarantined";
        case failure_kind::none:
        default:
            return "copy_token: operation failed";
        }
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        text += std::string(": ") + detail::driver_error_string(code);
#else
        text += ": error " + std::to_string(code);
#endif
        return text;
    }

    // Store retained pointers so they stay alive until token completion.
    // The payload is opaque (a holder for retained_ptr copies). Submission side
    // only: call before the token is shared.
    void set_retained(std::shared_ptr<void> retained)
    {
        ensure_state();
        state_->retained = std::move(retained);
    }

    // Setup failed or the stream was proven idle after a failed submission:
    // nothing is in flight, so the operation is complete and the payload can go.
    void mark_complete() noexcept
    {
        if (state_ && settle(completion_state::complete, failure_kind::none, 0))
        {
            state_->retained.reset();
        }
    }

    // Completion could not be proven: the operation is failed and its payload
    // stays retained.
    void mark_failed() noexcept
    {
        if (state_)
        {
            settle(completion_state::failed, failure_kind::submission_unsafe, 0);
        }
    }

    struct shared_state
    {
        explicit shared_state(execution_context value) noexcept : ctx(value) {}
        ~shared_state()
        {
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
            if (event_created && event != nullptr)
            {
                detail::token_event_destroy(ctx.device_index(), event);
            }
#endif
        }

        execution_context     ctx{};
        std::shared_ptr<void> retained;
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
        void* event{nullptr};  // cudaEvent_t / hipEvent_t, opaque in this header
#endif
        bool              event_created{false};
        std::atomic<bool> event_recorded{false};
        // Published terminal result (see encode()); 0 while pending. Set once.
        std::atomic<std::uint64_t> terminal{0};
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

namespace detail
{
// Library-internal access to the token's forced terminal transitions. Not part
// of the public API: user code must not decide that an operation completed.
struct copy_token_access
{
    static void set_retained(copy_token& t, std::shared_ptr<void> retained)
    {
        t.set_retained(std::move(retained));
    }
    static void complete(copy_token& t) noexcept { t.mark_complete(); }
    static void fail(copy_token& t) noexcept { t.mark_failed(); }
};
}  // namespace detail

}  // namespace memory
