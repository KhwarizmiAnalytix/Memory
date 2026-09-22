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

// Deterministic host-only runtime used solely to exercise the production pool's
// delayed-completion and failure paths. It does not emulate GPU execution.
#include <array>
#include <cstdlib>
#include <cstring>
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
inline constexpr unsigned cudaEventDisableTiming    = 2;
inline constexpr unsigned cudaHostAllocPortable     = 1;

namespace fake_runtime
{
inline std::array<bool, 4> ready{true, true, true, true};
inline int                 fail_allocations  = 0;
inline bool                fail_record       = false;
inline bool                fail_query        = false;
inline bool                fail_event_create = false;
inline bool                fail_host_free    = false;
inline int                 current_device    = 0;
inline int                 host_allocations  = 0;
inline int                 host_frees        = 0;
inline int                 event_creates     = 0;
inline int                 event_destroys    = 0;
inline int                 copies            = 0;
inline int                 synchronizations  = 0;
inline std::set<void*>     backing;
inline cudaStream_t        stream(std::size_t i)
{
    return reinterpret_cast<void*>(i);
}
inline void reset()
{
    for (void* ptr : backing)
        std::free(ptr);  // Reclaim deliberately quarantined fake backing.
    backing.clear();
    ready.fill(true);
    fail_allocations = 0;
    fail_record = fail_query = fail_event_create = fail_host_free = false;
    current_device = host_allocations = host_frees = event_creates = event_destroys = 0;
    copies = synchronizations = 0;
}
}  // namespace fake_runtime
inline const char* cudaGetErrorString(cudaError_t)
{
    return "injected runtime failure";
}
inline cudaError_t cudaGetLastError()
{
    return cudaSuccess;
}
inline cudaError_t cudaGetDevice(int* device)
{
    *device = fake_runtime::current_device;
    return 0;
}
inline cudaError_t cudaSetDevice(int device)
{
    fake_runtime::current_device = device;
    return 0;
}
inline cudaError_t cudaHostAlloc(void** ptr, std::size_t bytes, unsigned)
{
    if (fake_runtime::fail_allocations > 0)
    {
        --fake_runtime::fail_allocations;
        return 1;
    }
    *ptr = std::malloc(bytes);
    if (!*ptr)
        return 1;
    fake_runtime::backing.insert(*ptr);
    ++fake_runtime::host_allocations;
    return 0;
}
inline cudaError_t cudaFreeHost(void* ptr)
{
    if (fake_runtime::fail_host_free)
        return 3;
    fake_runtime::backing.erase(ptr);
    std::free(ptr);
    ++fake_runtime::host_frees;
    return 0;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned)
{
    if (fake_runtime::fail_event_create)
        return 3;
    *event = new fake_event;
    ++fake_runtime::event_creates;
    return 0;
}
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    if (fake_runtime::fail_record)
        return 3;
    event->stream = stream;
    return 0;
}
inline cudaError_t cudaEventQuery(cudaEvent_t event)
{
    if (fake_runtime::fail_query)
        return 3;
    return fake_runtime::ready.at(reinterpret_cast<std::size_t>(event->stream)) ? 0 : 2;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    ++fake_runtime::synchronizations;
    if (fake_runtime::fail_query)
        return 3;
    fake_runtime::ready.at(reinterpret_cast<std::size_t>(event->stream)) = true;
    return 0;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    delete event;
    ++fake_runtime::event_destroys;
    return 0;
}
inline cudaError_t cudaMemcpyAsync(
    void* to, const void* from, std::size_t bytes, cudaMemcpyKind, cudaStream_t)
{
    ++fake_runtime::copies;
    std::memcpy(to, from, bytes);
    return 0;
}
