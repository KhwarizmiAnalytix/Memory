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

// Deterministic host-only runtime for testing copy_token and allocator.h GPU paths.
// Simulates stream state (ready/not-ready/error), device tracking, and peer copies.
#include <array>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

using cudaError_t  = int;
using cudaStream_t = void*;
struct fake_event
{
    cudaStream_t stream{nullptr};
    bool         complete{false};
};
using cudaEvent_t = fake_event*;
enum cudaMemcpyKind
{
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice
};
inline constexpr int      cudaSuccess               = 0;
inline constexpr int      cudaErrorMemoryAllocation = 1;
inline constexpr int      cudaErrorNotReady         = 2;
inline constexpr int      cudaErrorInvalidValue     = 3;
inline constexpr unsigned cudaEventDisableTiming    = 2;
inline constexpr unsigned cudaHostAllocPortable     = 1;

namespace fake_runtime
{
// Per-stream state: ready, not-ready, or error.
enum stream_state : int
{
    STREAM_READY = 0,
    STREAM_NOT_READY = 2,  // Same as cudaErrorNotReady
    STREAM_ERROR = 3       // Some other error
};

inline std::map<cudaStream_t, stream_state> stream_states;
inline int                                   current_device    = 0;
inline int                                   device_count      = 4;
inline int                                   host_allocations  = 0;
inline int                                   host_frees        = 0;
inline int                                   event_creates     = 0;
inline int                                   event_destroys    = 0;
inline int                                   event_records     = 0;
inline int                                   event_queries     = 0;
inline int                                   event_syncs       = 0;
inline int                                   stream_queries    = 0;
inline int                                   stream_syncs      = 0;
inline int                                   device_syncs      = 0;
inline int                                   copies            = 0;
inline int                                   peer_copies       = 0;
inline bool                                  fail_allocations  = false;
inline bool                                  fail_stream_query = false;
// Fault injection for the copy path (task 1.5/1.6): each makes the matching
// runtime call return an error without performing it.
inline bool                                  fail_event_create = false;
inline bool                                  fail_event_record = false;
inline bool                                  fail_memcpy       = false;
// 1-based index of the cudaSetDevice call that fails (0 = never); counts all calls.
inline int                                   fail_set_device_on_call = 0;
inline int                                   set_device_calls        = 0;
inline std::set<void*>                       backing;

inline void reset()
{
    for (void* ptr : backing)
        std::free(ptr);
    backing.clear();
    stream_states.clear();
    current_device = 0;
    host_allocations = 0;
    host_frees = 0;
    event_creates = 0;
    event_destroys = 0;
    event_records = 0;
    event_queries = 0;
    event_syncs = 0;
    stream_queries = 0;
    stream_syncs = 0;
    device_syncs = 0;
    copies = 0;
    peer_copies = 0;
    fail_allocations = false;
    fail_stream_query = false;
    fail_event_create = false;
    fail_event_record = false;
    fail_memcpy = false;
    fail_set_device_on_call = 0;
    set_device_calls = 0;
}

inline void set_stream_ready(cudaStream_t stream, bool is_ready = true)
{
    if (stream == nullptr)
        stream = reinterpret_cast<cudaStream_t>(0);
    stream_states[stream] = is_ready ? STREAM_READY : STREAM_NOT_READY;
}

inline void set_stream_error(cudaStream_t stream)
{
    if (stream == nullptr)
        stream = reinterpret_cast<cudaStream_t>(0);
    stream_states[stream] = STREAM_ERROR;
}

inline stream_state get_stream_state(cudaStream_t stream)
{
    if (stream == nullptr)
        stream = reinterpret_cast<cudaStream_t>(0);
    auto it = stream_states.find(stream);
    return (it != stream_states.end()) ? it->second : STREAM_READY;
}

}  // namespace fake_runtime

inline const char* cudaGetErrorString(cudaError_t err)
{
    switch (err)
    {
        case 0: return "cudaSuccess";
        case 1: return "cudaErrorMemoryAllocation";
        case 2: return "cudaErrorNotReady";
        default: return "injected runtime failure";
    }
}

inline cudaError_t cudaGetLastError()
{
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceCount(int* count)
{
    *count = fake_runtime::device_count;
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int* device)
{
    *device = fake_runtime::current_device;
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int device)
{
    ++fake_runtime::set_device_calls;
    if (fake_runtime::fail_set_device_on_call == fake_runtime::set_device_calls)
        return cudaErrorInvalidValue;
    if (device < 0 || device >= fake_runtime::device_count)
        return cudaErrorInvalidValue;
    fake_runtime::current_device = device;
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize()
{
    ++fake_runtime::device_syncs;
    int device = fake_runtime::current_device;
    auto state = fake_runtime::get_stream_state(nullptr);
    if (state == fake_runtime::STREAM_ERROR)
        return 3;
    fake_runtime::set_stream_ready(nullptr, true);
    return cudaSuccess;
}

inline cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    ++fake_runtime::stream_queries;
    if (fake_runtime::fail_stream_query)
        return 3;
    if (stream == nullptr)
        stream = reinterpret_cast<cudaStream_t>(0);
    auto state = fake_runtime::get_stream_state(stream);
    if (state == fake_runtime::STREAM_ERROR)
        return 3;
    return (state == fake_runtime::STREAM_READY) ? cudaSuccess : cudaErrorNotReady;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    ++fake_runtime::stream_syncs;
    if (stream == nullptr)
        stream = reinterpret_cast<cudaStream_t>(0);
    auto state = fake_runtime::get_stream_state(stream);
    if (state == fake_runtime::STREAM_ERROR)
        return 3;
    fake_runtime::set_stream_ready(stream, true);
    return cudaSuccess;
}

inline cudaError_t cudaHostAlloc(void** ptr, std::size_t bytes, unsigned)
{
    if (fake_runtime::fail_allocations)
        return 1;
    *ptr = std::malloc(bytes);
    if (!*ptr)
        return 1;
    fake_runtime::backing.insert(*ptr);
    ++fake_runtime::host_allocations;
    return 0;
}

inline cudaError_t cudaFreeHost(void* ptr)
{
    fake_runtime::backing.erase(ptr);
    std::free(ptr);
    ++fake_runtime::host_frees;
    return 0;
}

inline cudaError_t cudaEventCreate(cudaEvent_t* event)
{
    if (fake_runtime::fail_event_create)
        return 3;
    *event = new fake_event;
    ++fake_runtime::event_creates;
    return 0;
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned)
{
    return cudaEventCreate(event);
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    if (fake_runtime::fail_event_record)
        return 3;
    event->stream = stream;
    event->complete =
        fake_runtime::get_stream_state(stream) == fake_runtime::STREAM_READY;
    ++fake_runtime::event_records;
    return 0;
}

inline cudaError_t cudaEventQuery(cudaEvent_t event)
{
    ++fake_runtime::event_queries;
    if (fake_runtime::fail_stream_query)
        return 3;
    if (event->complete)
        return cudaSuccess;
    auto state = fake_runtime::get_stream_state(event->stream);
    if (state == fake_runtime::STREAM_ERROR)
        return 3;
    if (state == fake_runtime::STREAM_READY)
    {
        event->complete = true;
        return cudaSuccess;
    }
    return cudaErrorNotReady;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    ++fake_runtime::event_syncs;
    if (fake_runtime::fail_stream_query)
        return 3;
    auto state = fake_runtime::get_stream_state(event->stream);
    if (state == fake_runtime::STREAM_ERROR)
        return 3;
    event->complete = true;
    fake_runtime::set_stream_ready(event->stream, true);
    return 0;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    delete event;
    ++fake_runtime::event_destroys;
    return 0;
}

inline cudaError_t cudaMemcpyAsync(
    void* to, const void* from, std::size_t bytes, cudaMemcpyKind, cudaStream_t stream)
{
    if (fake_runtime::fail_memcpy)
        return 3;
    ++fake_runtime::copies;
    std::memcpy(to, from, bytes);
    return 0;
}

inline cudaError_t cudaMemcpyPeerAsync(
    void* to, int to_device, const void* from, int from_device, std::size_t bytes, cudaStream_t stream)
{
    ++fake_runtime::peer_copies;
    if (to_device < 0 || to_device >= fake_runtime::device_count ||
        from_device < 0 || from_device >= fake_runtime::device_count)
        return cudaErrorInvalidValue;
    std::memcpy(to, from, bytes);
    return 0;
}
