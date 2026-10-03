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

#include "common/device.h"
#include "common/execution_context.h"
#include "common/memory_export.h"

namespace memory::detail
{

/// Byte copy router. Callers validate arguments first (null endpoints, byte
/// count, supported endpoint kinds) and handle the CPU-to-CPU memcpy inline;
/// this routes everything that touches a GPU endpoint.
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
MEMORY_API void          token_event_destroy(int device, void* event) noexcept;
/// Static text for a raw driver code.
MEMORY_API char const* driver_error_string(int code) noexcept;

}  // namespace memory::detail
