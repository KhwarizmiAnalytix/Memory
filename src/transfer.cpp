/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

// Byte copy router and completion-token driver operations (plan 4.4, task 2.6).
// The only translation unit, besides the backends, that touches CUDA/HIP
// runtime calls on the copy path.

#include "common/transfer.h"

#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/device_guard.h"
#include "gpu/gpu_dispatch.h"
#include "gpu/gpu_runtime.h"
#endif

namespace memory::detail
{

namespace
{
// Size-class freelists for recycled_block_acquire/release: 16-byte classes up
// to 256 bytes, at most kMaxFree blocks kept per class. Leaked on purpose: the
// blocks back shared state that static destructors elsewhere may still release.
class block_recycler
{
public:
    static constexpr std::size_t kGranule = 16;
    static constexpr std::size_t kClasses = 16;
    static constexpr std::size_t kMaxFree = 1024;

    static block_recycler& instance()
    {
        static block_recycler* const pool = new block_recycler;
        return *pool;
    }

    static constexpr std::size_t class_of(std::size_t bytes) noexcept
    {
        return (bytes + kGranule - 1) / kGranule - 1;
    }

    void* acquire(std::size_t bytes)
    {
        std::size_t const c = class_of(bytes);
        {
            std::lock_guard<std::mutex> const lock(mutex_);
            entry*&                           head = heads_[c];
            if (head != nullptr)
            {
                entry* const e = head;
                head           = e->next;
                --counts_[c];
                return e;
            }
        }
        return ::operator new((c + 1) * kGranule);
    }

    void release(void* block, std::size_t bytes) noexcept
    {
        std::size_t const c = class_of(bytes);
        {
            std::lock_guard<std::mutex> const lock(mutex_);
            if (counts_[c] < kMaxFree)
            {
                heads_[c] = new (block) entry{heads_[c]};
                ++counts_[c];
                return;
            }
        }
        ::operator delete(block);
    }

private:
    struct entry
    {
        entry* next;
    };
    block_recycler() = default;

    std::mutex                        mutex_;
    std::array<entry*, kClasses>      heads_{};
    std::array<std::size_t, kClasses> counts_{};
};
}  // namespace

void* recycled_block_acquire(std::size_t bytes)
{
    if (bytes == 0 || bytes > block_recycler::kGranule * block_recycler::kClasses)
    {
        return ::operator new(bytes == 0 ? 1 : bytes);
    }
    return block_recycler::instance().acquire(bytes);
}

void recycled_block_release(void* block, std::size_t bytes) noexcept
{
    if (bytes == 0 || bytes > block_recycler::kGranule * block_recycler::kClasses)
    {
        ::operator delete(block);
        return;
    }
    block_recycler::instance().release(block, bytes);
}

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

namespace
{
constexpr default_stream_mode kCompiledMode = caller_default_stream_mode();
std::atomic<default_stream_mode> g_library_stream_mode{kCompiledMode};
}  // namespace

default_stream_mode library_default_stream_mode() noexcept
{
    return g_library_stream_mode.load(std::memory_order_relaxed);
}

void set_library_default_stream_mode_for_testing(default_stream_mode mode) noexcept
{
    g_library_stream_mode.store(mode, std::memory_order_relaxed);
}

void validate_stream_handle(stream_handle_t stream, default_stream_mode caller)
{
    default_stream_mode const library = library_default_stream_mode();
    if (stream == nullptr)
    {
        if (caller != library)
        {
            throw std::invalid_argument(
                "ambiguous null stream: the caller and the Memory library disagree on whether "
                "the null stream is the legacy or the per-thread default stream; pass an "
                "explicit stream");
        }
        if (library == default_stream_mode::per_thread)
        {
            throw std::invalid_argument(
                "null stream is the per-thread default stream, which is a different stream on "
                "every thread and cannot be a cache identity; pass an explicit stream");
        }
        return;
    }
#ifdef cudaStreamPerThread
    if (stream == static_cast<stream_handle_t>(cudaStreamPerThread))
    {
        throw std::invalid_argument(
            "cudaStreamPerThread cannot be a cache identity (a different stream on every "
            "thread); pass an explicit stream");
    }
#endif
}

void validate_route(device from, device to)
{
    bool const from_gpu = is_cuda_or_hip(from.type);
    bool const to_gpu   = is_cuda_or_hip(to.type);
    if ((!from_gpu && from.type != device_enum::CPU) || (!to_gpu && to.type != device_enum::CPU))
    {
        throw std::invalid_argument("Unsupported device combination for memory copy");
    }
    if (from_gpu && to_gpu && from.index != to.index)
    {
        int               can    = 0;
        cudaError_t const status = cudaDeviceCanAccessPeer(&can, to.index, from.index);
        if (status != cudaSuccess)
        {
            (void)cudaGetLastError();
            throw std::invalid_argument(
                "peer copy rejected: peer access between devices " + std::to_string(from.index) +
                " and " + std::to_string(to.index) + " could not be queried (" +
                std::string(cudaGetErrorString(status)) + ")");
        }
        if (can == 0)
        {
            throw std::invalid_argument(
                "peer copy rejected: device " + std::to_string(to.index) +
                " cannot access device " + std::to_string(from.index) +
                "; copy through host memory instead");
        }
    }
}

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
        // Validate both endpoints before recording a use on either or submitting:
        // an interior, freed or foreign pointer cannot be tracked by the cache, so
        // the copy is refused rather than left to a failing lookup mid-way (4.7).
        if (is_cuda_or_hip(from_type) && !gpu::owns_live_allocation(from, from_index))
        {
            throw std::invalid_argument(
                "copy source is not the base of a live GPU allocation (interior, freed or "
                "foreign pointer); nothing was submitted");
        }
        if (is_cuda_or_hip(to_type) && !gpu::owns_live_allocation(to, to_index))
        {
            throw std::invalid_argument(
                "copy destination is not the base of a live GPU allocation (interior, freed "
                "or foreign pointer); nothing was submitted");
        }
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

namespace
{
// Per-device pool of idle token events (plan 3.4). Leaked on purpose, like the
// block recycler: tokens may be destroyed by static destructors.
class event_pool
{
public:
    static constexpr std::size_t kMaxDevices = 64;
    static constexpr std::size_t kMaxPooled  = 64;

    static event_pool& instance()
    {
        static event_pool* const pool = new event_pool;
        return *pool;
    }

    void* take(int device)
    {
        if (!valid(device))
        {
            return nullptr;
        }
        std::lock_guard<std::mutex> const lock(mutex_);
        auto&                             idle = idle_[static_cast<std::size_t>(device)];
        if (idle.empty())
        {
            return nullptr;
        }
        void* const event = idle.back();
        idle.pop_back();
        return event;
    }

    bool give(int device, void* event) noexcept
    {
        if (!valid(device))
        {
            return false;
        }
        std::lock_guard<std::mutex> const lock(mutex_);
        auto&                             idle = idle_[static_cast<std::size_t>(device)];
        if (idle.size() >= kMaxPooled)
        {
            return false;
        }
        try
        {
            if (idle.capacity() < kMaxPooled)
            {
                idle.reserve(kMaxPooled);
            }
            idle.push_back(event);  // capacity is reserved: cannot reallocate
        }
        catch (...)
        {
            return false;
        }
        return true;
    }

    void release_all() noexcept
    {
        for (std::size_t d = 0; d < kMaxDevices; ++d)
        {
            std::vector<void*> events;
            {
                std::lock_guard<std::mutex> const lock(mutex_);
                events.swap(idle_[d]);
            }
            if (events.empty())
            {
                continue;
            }
            gpu::device_guard guard(static_cast<int>(d), std::nothrow);
            for (void* event : events)
            {
                (void)cudaEventDestroy(static_cast<cudaEvent_t>(event));
            }
        }
    }

private:
    static bool valid(int device) noexcept
    {
        return device >= 0 && static_cast<std::size_t>(device) < kMaxDevices;
    }
    event_pool() = default;

    std::mutex                                  mutex_;
    std::array<std::vector<void*>, kMaxDevices> idle_;
};
}  // namespace

driver_result token_event_create(int device, void** event)
{
    if (void* const pooled = event_pool::instance().take(device))
    {
        *event = pooled;
        return {};
    }
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

// A failed device activation is an error, never a result: answering from the
// wrong device context could report an operation complete that was not
// (plan 4.1). The token turns the error into a published failed state.
driver_result activation_failure(gpu::device_guard const& guard) noexcept
{
    return {driver_status::error, guard.error()};
}

driver_result token_event_query(int device, void* event) noexcept
{
    gpu::device_guard guard(device, std::nothrow);
    if (!guard.active())
    {
        return activation_failure(guard);
    }
    return to_result(cudaEventQuery(static_cast<cudaEvent_t>(event)));
}

driver_result token_event_synchronize(int device, void* event) noexcept
{
    gpu::device_guard guard(device, std::nothrow);
    if (!guard.active())
    {
        return activation_failure(guard);
    }
    return to_result(cudaEventSynchronize(static_cast<cudaEvent_t>(event)));
}

driver_result token_stream_wait(void* event, stream_handle_t consumer)
{
    return to_result(cudaStreamWaitEvent(native(consumer), static_cast<cudaEvent_t>(event), 0));
}

driver_result token_stream_query(int device, stream_handle_t stream) noexcept
{
    gpu::device_guard guard(device, std::nothrow);
    if (!guard.active())
    {
        return activation_failure(guard);
    }
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
    if (!guard.active())
    {
        return activation_failure(guard);
    }
    return to_result(stream != nullptr ? cudaStreamSynchronize(native(stream))
                                       : cudaDeviceSynchronize());
}

void token_event_destroy(int device, void* event) noexcept
{
    if (event_pool::instance().give(device, event))
    {
        return;
    }
    gpu::device_guard guard(device, std::nothrow);
    (void)cudaEventDestroy(static_cast<cudaEvent_t>(event));
}

void release_token_event_pool() noexcept
{
    event_pool::instance().release_all();
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

void release_token_event_pool() noexcept {}

namespace
{
std::atomic<default_stream_mode> g_library_stream_mode{default_stream_mode::legacy};
}  // namespace

default_stream_mode library_default_stream_mode() noexcept
{
    return g_library_stream_mode.load(std::memory_order_relaxed);
}

void set_library_default_stream_mode_for_testing(default_stream_mode mode) noexcept
{
    g_library_stream_mode.store(mode, std::memory_order_relaxed);
}

// Without a CUDA/HIP backend no handle is ever interpreted as a stream.
void validate_stream_handle(stream_handle_t, default_stream_mode) {}

void validate_route(device from, device to)
{
#if MEMORY_HAS_METAL
    bool const ok = (from.type == device_enum::CPU || from.type == device_enum::METAL) &&
                    (to.type == device_enum::CPU || to.type == device_enum::METAL);
#else
    bool const ok = from.type == device_enum::CPU && to.type == device_enum::CPU;
#endif
    if (!ok)
    {
        throw std::invalid_argument("Unsupported device combination for memory copy");
    }
}

#endif

}  // namespace memory::detail
