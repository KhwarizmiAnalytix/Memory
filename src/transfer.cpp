/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

// Byte copy router and completion-token driver operations (plan 4.4, task 2.6).
// The only translation unit, besides the backends, that touches CUDA/HIP
// runtime calls on the copy path.

#include "common/transfer.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/device_guard.h"
#include "gpu/gpu_dispatch.h"
#include "gpu/gpu_runtime.h"
#endif

namespace memory::detail
{

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

namespace
{
constexpr bool is_cuda_or_hip(device_enum type) noexcept
{
    return type == device_enum::CUDA || type == device_enum::HIP;
}

cudaStream_t native(stream_handle_t stream) noexcept
{
    return static_cast<cudaStream_t>(stream);
}

driver_result to_result(cudaError_t r) noexcept
{
    if (r == cudaSuccess)
    {
        return {driver_status::ok, 0};
    }
    if (r == cudaErrorNotReady)
    {
        return {driver_status::not_ready, static_cast<int>(r)};
    }
    return {driver_status::error, static_cast<int>(r)};
}
}  // namespace

void route_copy_bytes(
    void const*     from,
    void*           to,
    std::size_t     nbytes,
    device          from_dev,
    device          to_dev,
    stream_handle_t stream,
    bool            track_gpu_streams,
    bool*           submitted)
{
    device_enum const from_type = from_dev.type;
    device_enum const to_type   = to_dev.type;
    int const         from_index = from_dev.index;
    int const         to_index   = to_dev.index;

    if (from_type == device_enum::CPU && to_type == device_enum::CPU)
    {
        if (submitted != nullptr)
        {
            *submitted = true;
        }
        std::memcpy(to, from, nbytes);
        return;
    }

    if (!is_cuda_or_hip(from_type) && !is_cuda_or_hip(to_type))
    {
        throw std::invalid_argument("Unsupported device combination for memory copy");
    }

    // Register stream uses on GPU endpoints BEFORE enqueuing the copy. This
    // closes the window between submission and registration: a deallocation
    // arriving between cudaMemcpyAsync and record_stream could reclaim a block
    // that the in-flight copy still references. A null stream is the default
    // CUDA stream, a valid stream identity, not "no stream".
    if (track_gpu_streams)
    {
        if (is_cuda_or_hip(from_type))
        {
            gpu::record_stream_use(const_cast<void*>(from), from_index, stream);
        }
        if (is_cuda_or_hip(to_type))
        {
            gpu::record_stream_use(to, to_index, stream);
        }
    }

    cudaError_t result = cudaSuccess;
    if (is_cuda_or_hip(from_type) && is_cuda_or_hip(to_type) && from_index != to_index)
    {
        gpu::device_guard const peer_guard(to_index);
        if (submitted != nullptr)
        {
            *submitted = true;
        }
        result = cudaMemcpyPeerAsync(to, to_index, from, from_index, nbytes, native(stream));
    }
    else
    {
        cudaMemcpyKind copy_kind;
        if (from_type == device_enum::CPU && is_cuda_or_hip(to_type))
        {
            copy_kind = cudaMemcpyHostToDevice;
        }
        else if (is_cuda_or_hip(from_type) && to_type == device_enum::CPU)
        {
            copy_kind = cudaMemcpyDeviceToHost;
        }
        else
        {
            copy_kind = cudaMemcpyDeviceToDevice;
        }

        int const          gpu_index = is_cuda_or_hip(to_type) ? to_index : from_index;
        gpu::device_guard const guard(gpu_index);
        if (submitted != nullptr)
        {
            *submitted = true;
        }
        // Always the async form so enqueuing is non-blocking; a null stream is the
        // default CUDA stream.
        result = cudaMemcpyAsync(to, from, nbytes, copy_kind, native(stream));
    }
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            "GPU memory copy failed: " + std::string(cudaGetErrorString(result)));
    }
}

bool stream_proven_idle(execution_context const& ctx) noexcept
{
    try
    {
        gpu::device_guard const guard(ctx.device_index());
        return cudaStreamSynchronize(native(ctx.stream)) == cudaSuccess;
    }
    catch (...)
    {
        return false;
    }
}

driver_result token_event_create(int device, void** event)
{
    gpu::device_guard guard(device);
    cudaEvent_t       created{};
    cudaError_t const r = cudaEventCreateWithFlags(&created, cudaEventDisableTiming);
    if (r == cudaSuccess)
    {
        *event = static_cast<void*>(created);
    }
    return to_result(r);
}

driver_result token_event_record(int device, void* event, stream_handle_t stream)
{
    gpu::device_guard guard(device);
    return to_result(cudaEventRecord(static_cast<cudaEvent_t>(event), native(stream)));
}

driver_result token_event_query(int device, void* event) noexcept
{
    gpu::device_guard guard(device, std::nothrow);
    return to_result(cudaEventQuery(static_cast<cudaEvent_t>(event)));
}

driver_result token_event_synchronize(int device, void* event) noexcept
{
    gpu::device_guard guard(device, std::nothrow);
    return to_result(cudaEventSynchronize(static_cast<cudaEvent_t>(event)));
}

driver_result token_stream_query(int device, stream_handle_t stream) noexcept
{
    gpu::device_guard guard(device, std::nothrow);
    cudaError_t const r = cudaStreamQuery(native(stream));
    if (r == cudaErrorNotReady)
    {
        // Clear sticky error state so repeated queries work.
        (void)cudaGetLastError();
    }
    return to_result(r);
}

driver_result token_stream_synchronize(int device, stream_handle_t stream) noexcept
{
    gpu::device_guard guard(device, std::nothrow);
    return to_result(stream != nullptr ? cudaStreamSynchronize(native(stream))
                                       : cudaDeviceSynchronize());
}

void token_event_destroy(int device, void* event) noexcept
{
    gpu::device_guard guard(device, std::nothrow);
    (void)cudaEventDestroy(static_cast<cudaEvent_t>(event));
}

char const* driver_error_string(int code) noexcept
{
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

#else  // no CUDA/HIP backend

void route_copy_bytes(
    void const*     from,
    void*           to,
    std::size_t     nbytes,
    device          from_dev,
    device          to_dev,
    stream_handle_t stream,
    bool            track_gpu_streams,
    bool*           submitted)
{
    (void)stream;
    (void)track_gpu_streams;
#if MEMORY_HAS_METAL
    // Shared-storage MTLBuffers are host-addressable: every METAL side is memcpy.
    bool const cpu_or_metal =
        (from_dev.type == device_enum::CPU || from_dev.type == device_enum::METAL) &&
        (to_dev.type == device_enum::CPU || to_dev.type == device_enum::METAL);
    if (cpu_or_metal)
    {
        if (submitted != nullptr)
        {
            *submitted = true;
        }
        std::memcpy(to, from, nbytes);
        return;
    }
#else
    if (from_dev.type == device_enum::CPU && to_dev.type == device_enum::CPU)
    {
        if (submitted != nullptr)
        {
            *submitted = true;
        }
        std::memcpy(to, from, nbytes);
        return;
    }
#endif
    throw std::invalid_argument("Unsupported device combination for memory copy");
}

bool stream_proven_idle(execution_context const&) noexcept
{
    return true;  // no stream can be in flight without a CUDA/HIP backend
}

#endif

}  // namespace memory::detail
