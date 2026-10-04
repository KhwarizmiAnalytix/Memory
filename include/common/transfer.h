/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

// Byte-level copy router and completion-token driver operations (plan 4.4,
// task 2.6). Declarations only: no vendor runtime header is included, so the
// typed owners and copy_token compile in a consumer TU without CUDA/HIP on the
// include path. Defined in src/transfer.cpp.

#include <cstddef>
#include <cstdint>
#include <new>

#include "common/device.h"
#include "common/execution_context.h"
#include "common/memory_export.h"

namespace memory::detail
{

// --- Context validation (plan 4.3, 4.5) -------------------------------------

/// Which stream a null `stream_handle_t` names. CUDA/HIP translation units compiled
/// with CUDA_API_PER_THREAD_DEFAULT_STREAM (or the HIP equivalent) treat the null
/// stream as the per-thread default stream; otherwise it is the legacy default
/// stream that implicitly synchronizes with other blocking streams. The two are
/// different physical streams with different ordering, and a per-thread default
/// stream is a different stream on every thread, so a null handle cannot be a
/// cache identity in that mode.
enum class default_stream_mode : std::uint8_t
{
    legacy     = 0,
    per_thread = 1
};

/// Mode the calling translation unit was compiled in (what its null means).
constexpr default_stream_mode caller_default_stream_mode() noexcept
{
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM) || defined(HIP_API_PER_THREAD_DEFAULT_STREAM)
    return default_stream_mode::per_thread;
#else
    return default_stream_mode::legacy;
#endif
}

/// Mode the Memory library itself was compiled in.
MEMORY_API default_stream_mode library_default_stream_mode() noexcept;

/// Test hook: pretend the library was compiled in @p mode. Not for production use.
MEMORY_API void set_library_default_stream_mode_for_testing(default_stream_mode mode) noexcept;

/// Rejects a stream handle that cannot name an unambiguous stream, before any
/// resource is acquired: a null handle when the caller's mode differs from the
/// library's, a null handle in per-thread mode (each thread would share one cache
/// identity), and the explicit per-thread sentinel (`cudaStreamPerThread`).
/// Throws std::invalid_argument. Legacy mode with a null or explicit stream is
/// accepted; the legacy sentinel and null are distinct cache identities.
MEMORY_API void validate_stream_handle(stream_handle_t stream, default_stream_mode caller);

/// Validates the stream of a GPU context before an allocation or copy acquires
/// anything. CPU contexts have no stream and are accepted untouched.
inline void validate_context_stream(execution_context const& ctx)
{
    if (ctx.is_gpu())
    {
        validate_stream_handle(ctx.stream, caller_default_stream_mode());
    }
}

/// Rejects an endpoint combination this build cannot perform, before submission:
/// a GPU-to-GPU copy across devices that cannot access each other (peer access is
/// a permission the copy never grants; it only checks it), an out-of-range device
/// index, or a non-host non-GPU endpoint. Throws std::invalid_argument.
MEMORY_API void validate_route(device from, device to);

/// Byte copy router. Callers validate arguments first (null endpoints, byte
/// count, supported endpoint kinds) and handle the CPU-to-CPU memcpy inline;
/// this routes everything that touches a GPU endpoint.
///
/// With @p track_gpu_streams every GPU endpoint must be the base of a live
/// cache allocation: both are checked before the first stream use is recorded or
/// any work is submitted, and an interior, freed or foreign pointer throws
/// std::invalid_argument (plan 4.7).
///
/// When @p track_gpu_streams is set, every GPU endpoint has the use on @p stream
/// recorded with the caching allocator BEFORE the copy is enqueued, closing the
/// window in which a free could reclaim a block an in-flight copy still names.
/// @p submitted (optional) is set to true immediately before the driver is asked
/// to move data: a throw with false means nothing started; with true, work may be
/// in flight and the caller must prove completion before reusing the endpoints.
/// Throws std::invalid_argument for an unsupported combination and
/// std::runtime_error for a driver failure.
MEMORY_API void route_copy_bytes(
    void const*       from,
    void*             to,
    std::size_t       nbytes,
    device            from_dev,
    device            to_dev,
    stream_handle_t   stream,
    bool              track_gpu_streams,
    bool*             submitted);

/// After a failed submission: true when @p ctx's stream is proven idle by a
/// successful synchronize. Never throws; false when it cannot be proven.
MEMORY_API bool stream_proven_idle(execution_context const& ctx) noexcept;

// --- Recycled small blocks (plan 3.4) ---------------------------------------

/// Small-block recycler for per-operation shared state (copy-token state, the
/// retained-endpoint holder). A released block goes on a bounded freelist
/// instead of back to the heap, so a steady-state copy does not allocate.
/// Requests above the largest class fall through to operator new. Thread-safe;
/// the pool lives for the process and is never torn down.
MEMORY_API void* recycled_block_acquire(std::size_t bytes);
MEMORY_API void  recycled_block_release(void* block, std::size_t bytes) noexcept;

/// Allocator for std::allocate_shared that draws its single block (state plus
/// control block) from the recycler.
template <typename T>
struct recycled_allocator
{
    using value_type = T;

    recycled_allocator() noexcept = default;
    template <typename U>
    recycled_allocator(recycled_allocator<U> const&) noexcept  // NOLINT(google-explicit-constructor)
    {
    }

    T* allocate(std::size_t n)
    {
        static_assert(alignof(T) <= alignof(std::max_align_t), "over-aligned shared state");
        return static_cast<T*>(recycled_block_acquire(n * sizeof(T)));
    }
    void deallocate(T* p, std::size_t n) noexcept { recycled_block_release(p, n * sizeof(T)); }

    template <typename U>
    bool operator==(recycled_allocator<U> const&) const noexcept
    {
        return true;
    }
    template <typename U>
    bool operator!=(recycled_allocator<U> const&) const noexcept
    {
        return false;
    }
};

// --- Completion-token driver operations (CUDA/HIP builds only) --------------

enum class driver_status : std::uint8_t
{
    ok,
    not_ready,
    error
};

struct driver_result
{
    driver_status status{driver_status::ok};
    int           code{0};  // raw driver error code; 0 when ok
};

/// Create the token's timing-disabled event on @p device. Throws when the device
/// cannot be activated.
MEMORY_API driver_result token_event_create(int device, void** event);
/// Record @p event on @p stream. Throws when the device cannot be activated.
MEMORY_API driver_result token_event_record(int device, void* event, stream_handle_t stream);
MEMORY_API driver_result token_event_query(int device, void* event) noexcept;
MEMORY_API driver_result token_event_synchronize(int device, void* event) noexcept;
/// Stream query for tokens without an event; clears the sticky error on not-ready.
MEMORY_API driver_result token_stream_query(int device, stream_handle_t stream) noexcept;
/// Synchronize @p stream, or the whole device when @p stream is null.
MEMORY_API driver_result token_stream_synchronize(int device, stream_handle_t stream) noexcept;
/// Returns @p event to the per-device pool (bounded; destroyed when full).
/// create takes from the pool first, so a steady-state token creates and destroys
/// no driver event (plan 3.4). A recycled event is simply re-recorded.
MEMORY_API void          token_event_destroy(int device, void* event) noexcept;
/// Destroys every pooled event. Call before the driver runtime is torn down.
MEMORY_API void          release_token_event_pool() noexcept;
/// Make @p consumer wait for @p event (cudaStreamWaitEvent): work enqueued on it
/// afterwards runs after the event's recorded work. Needs no device activation:
/// the stream's device is what matters and an event may be waited on across devices.
MEMORY_API driver_result token_stream_wait(void* event, stream_handle_t consumer);
/// Static text for a raw driver code.
MEMORY_API char const* driver_error_string(int code) noexcept;

}  // namespace memory::detail
