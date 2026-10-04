#pragma once

// Thread-safe, host-only fake driver for the task 8.1 contention benchmark.
// Shadows ../fake_runtime.h (which is single-threaded: plain counters and an
// unlocked std::set). Put this directory BEFORE the parent in the include path.
// Driver calls are as cheap as possible (malloc/free + relaxed atomic counters), so
// what the benchmark measures is the allocator's own host cost and lock behaviour;
// driver latency and device-side synchronization are NOT modelled.

#include <atomic>
#include <cstddef>
#include <cstdlib>

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
inline std::atomic<size_t> malloc_calls{0};
inline std::atomic<size_t> free_calls{0};
inline std::atomic<size_t> event_creates{0};
inline std::atomic<size_t> event_destroys{0};
inline std::atomic<size_t> event_record_calls{0};
inline std::atomic<size_t> event_query_calls{0};

inline cudaStream_t stream(std::size_t i)
{
    return reinterpret_cast<void*>(i + 1);
}

inline void reset()
{
    malloc_calls       = 0;
    free_calls         = 0;
    event_creates      = 0;
    event_destroys     = 0;
    event_record_calls = 0;
    event_query_calls  = 0;
}
}  // namespace fake_runtime

inline const char* cudaGetErrorString(cudaError_t) { return "fake_runtime"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* device)
{
    *device = 0;
    return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* count)
{
    *count = 1;
    return cudaSuccess;
}
inline cudaError_t cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes)
{
    *free_bytes  = 64ULL << 30;
    *total_bytes = 64ULL << 30;
    return cudaSuccess;
}
inline cudaError_t cudaMalloc(void** ptr, std::size_t size)
{
    fake_runtime::malloc_calls.fetch_add(1, std::memory_order_relaxed);
    *ptr = std::malloc(size);
    return *ptr != nullptr ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void* ptr)
{
    fake_runtime::free_calls.fetch_add(1, std::memory_order_relaxed);
    std::free(ptr);
    return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned)
{
    *event = new fake_event;
    fake_runtime::event_creates.fetch_add(1, std::memory_order_relaxed);
    return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    fake_runtime::event_record_calls.fetch_add(1, std::memory_order_relaxed);
    event->stream = stream;
    return cudaSuccess;
}
inline cudaError_t cudaEventQuery(cudaEvent_t)
{
    fake_runtime::event_query_calls.fetch_add(1, std::memory_order_relaxed);
    return cudaSuccess;  // work is always complete: no delayed consumers (task 8.2)
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    delete event;
    fake_runtime::event_destroys.fetch_add(1, std::memory_order_relaxed);
    return cudaSuccess;
}
