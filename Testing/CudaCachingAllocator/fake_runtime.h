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

// Deterministic host-only runtime used solely to exercise
// gpu/cuda_caching_allocator.cpp's failure/budget paths without real GPU
// hardware. Separate from Testing/PinnedRuntime/fake_runtime.h: this fakes a
// materially larger call surface (driver malloc/free, device enumeration,
// per-call event-record failure injection) than the pinned pool needs, and
// coupling the two unrelated test binaries' call surfaces would make both
// harder to reason about. Does not emulate GPU execution.

#include <array>
#include <condition_variable>
#include <cstdlib>
#include <cstddef>
#include <map>
#include <mutex>
#include <set>

using cudaError_t  = int;
using cudaStream_t = void*;
struct fake_event
{
    cudaStream_t stream{nullptr};
};
using cudaEvent_t = fake_event*;
enum cudaMemcpyKind
{
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost
};
inline constexpr int      cudaSuccess               = 0;
inline constexpr int      cudaErrorMemoryAllocation = 1;
inline constexpr int      cudaErrorNotReady         = 2;
inline constexpr int      cudaErrorUnknown          = 3;
inline constexpr unsigned cudaEventDisableTiming    = 2;

namespace fake_runtime
{
inline int         current_device     = 0;
inline int         device_count       = 1;
inline std::size_t device_total_bytes = 64ULL * 1024 * 1024 * 1024;

// cudaGetDevice failure injection: decrement-countdown, like fail_malloc_calls
// below. Arm with N to fail exactly the next N calls, then succeed again.
inline int fail_get_device_calls = 0;
inline int get_device_calls      = 0;
// Fails exactly the cudaGetDevice call whose 1-based index (get_device_calls)
// equals this value (0 = never).
inline int fail_get_device_at_call = 0;

// cudaMalloc failure injection: decrement-countdown.
inline int          fail_malloc_calls = 0;
inline int          malloc_calls      = 0;
// Error returned by an injected cudaMalloc failure (OOM unless a test changes it).
inline int          malloc_error      = 1;  // cudaErrorMemoryAllocation
// Fails exactly the cudaMalloc call whose 1-based index equals this (0 = never).
inline int          fail_malloc_at_call = 0;
inline std::set<void*> device_backing;
// Live bytes behind device_backing (cudaMalloc minus cudaFree), for budget checks.
inline std::size_t  device_backing_bytes = 0;
inline std::map<void*, std::size_t> device_backing_sizes;

// Pauses the cudaMalloc call whose 1-based call index matches
// malloc_pause_on_call (0 = disabled) until release_malloc_pause() is called
// from another thread. Used to make a genuinely concurrent budget-recheck
// window deterministic in a single-process test.
inline int                     malloc_pause_on_call = 0;
inline bool                    malloc_paused         = false;
inline bool                    malloc_release_requested = false;
inline std::mutex              malloc_pause_mutex;
inline std::condition_variable malloc_pause_cv;

inline void wait_for_malloc_pause()
{
    std::unique_lock lock(malloc_pause_mutex);
    malloc_pause_cv.wait(lock, [] { return malloc_paused; });
}

inline void release_malloc_pause()
{
    std::unique_lock lock(malloc_pause_mutex);
    malloc_release_requested = true;
    malloc_pause_cv.notify_all();
}

// cudaEventRecord failure injection: fails exactly the call whose 1-based
// index equals fail_event_record_at_call (0 = never), so a multi-stream
// deallocate can be made to fail on a specific (e.g. second) stream's event
// while earlier streams succeed -- the "partial failure" case.
inline int fail_event_record_at_call = 0;
inline int event_record_calls        = 0;

inline int fail_event_create_at_call = 0;
inline int event_create_calls        = 0;

inline int event_creates  = 0;
inline int event_destroys = 0;

inline std::array<bool, 8> event_ready{
    true, true, true, true, true, true, true, true};

inline cudaStream_t stream(std::size_t i)
{
    return reinterpret_cast<void*>(i + 1);  // 0 reserved for "default stream"/nullptr
}

inline void reset()
{
    {
        std::unique_lock lock(malloc_pause_mutex);
        malloc_pause_on_call     = 0;
        malloc_paused            = false;
        malloc_release_requested = false;
    }
    for (void* ptr : device_backing)
    {
        std::free(ptr);
    }
    device_backing.clear();
    device_backing_sizes.clear();
    device_backing_bytes   = 0;
    fail_get_device_at_call = 0;
    fail_malloc_at_call    = 0;
    malloc_error           = 1;
    current_device         = 0;
    device_count           = 1;
    device_total_bytes     = 64ULL * 1024 * 1024 * 1024;
    fail_get_device_calls  = 0;
    get_device_calls       = 0;
    fail_malloc_calls      = 0;
    malloc_calls           = 0;
    fail_event_record_at_call = 0;
    event_record_calls        = 0;
    fail_event_create_at_call = 0;
    event_create_calls        = 0;
    event_creates  = 0;
    event_destroys = 0;
    event_ready.fill(true);
}
}  // namespace fake_runtime

inline const char* cudaGetErrorString(cudaError_t)
{
    return "injected fake_runtime failure";
}

inline cudaError_t cudaGetLastError()
{
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int* device)
{
    ++fake_runtime::get_device_calls;
    if (fake_runtime::fail_get_device_at_call != 0 &&
        fake_runtime::get_device_calls == fake_runtime::fail_get_device_at_call)
    {
        return cudaErrorUnknown;
    }
    if (fake_runtime::fail_get_device_calls > 0)
    {
        --fake_runtime::fail_get_device_calls;
        return cudaErrorUnknown;
    }
    *device = fake_runtime::current_device;
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int device)
{
    fake_runtime::current_device = device;
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceCount(int* count)
{
    *count = fake_runtime::device_count;
    return cudaSuccess;
}

inline cudaError_t cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes)
{
    *free_bytes  = fake_runtime::device_total_bytes;
    *total_bytes = fake_runtime::device_total_bytes;
    return cudaSuccess;
}

inline cudaError_t cudaMalloc(void** ptr, std::size_t size)
{
    ++fake_runtime::malloc_calls;
    if (fake_runtime::malloc_pause_on_call == fake_runtime::malloc_calls)
    {
        std::unique_lock lock(fake_runtime::malloc_pause_mutex);
        fake_runtime::malloc_paused = true;
        fake_runtime::malloc_pause_cv.notify_all();
        fake_runtime::malloc_pause_cv.wait(
            lock, [] { return fake_runtime::malloc_release_requested; });
    }
    if (fake_runtime::fail_malloc_at_call != 0 &&
        fake_runtime::malloc_calls == fake_runtime::fail_malloc_at_call)
    {
        return fake_runtime::malloc_error;
    }
    if (fake_runtime::fail_malloc_calls > 0)
    {
        --fake_runtime::fail_malloc_calls;
        return fake_runtime::malloc_error;
    }
    *ptr = std::malloc(size);
    if (*ptr == nullptr)
    {
        return cudaErrorMemoryAllocation;
    }
    fake_runtime::device_backing.insert(*ptr);
    fake_runtime::device_backing_sizes[*ptr] = size;
    fake_runtime::device_backing_bytes += size;
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* ptr)
{
    auto const sized = fake_runtime::device_backing_sizes.find(ptr);
    if (sized != fake_runtime::device_backing_sizes.end())
    {
        fake_runtime::device_backing_bytes -= sized->second;
        fake_runtime::device_backing_sizes.erase(sized);
    }
    fake_runtime::device_backing.erase(ptr);
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned)
{
    ++fake_runtime::event_create_calls;
    if (fake_runtime::fail_event_create_at_call != 0 &&
        fake_runtime::event_create_calls == fake_runtime::fail_event_create_at_call)
    {
        return cudaErrorUnknown;
    }
    *event = new fake_event;
    ++fake_runtime::event_creates;
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    ++fake_runtime::event_record_calls;
    if (fake_runtime::fail_event_record_at_call != 0 &&
        fake_runtime::event_record_calls == fake_runtime::fail_event_record_at_call)
    {
        return cudaErrorUnknown;
    }
    event->stream = stream;
    return cudaSuccess;
}

inline cudaError_t cudaEventQuery(cudaEvent_t event)
{
    auto const idx = reinterpret_cast<std::size_t>(event->stream);
    return fake_runtime::event_ready.at(idx) ? cudaSuccess : cudaErrorNotReady;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    auto const idx                 = reinterpret_cast<std::size_t>(event->stream);
    fake_runtime::event_ready.at(idx) = true;
    return cudaSuccess;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    delete event;
    ++fake_runtime::event_destroys;
    return cudaSuccess;
}
