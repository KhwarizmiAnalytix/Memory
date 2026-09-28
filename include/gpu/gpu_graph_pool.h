/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

// Graph-aware memory pool skeleton for CUDA graph capture.
//
// During CUDA graph capture (cudaStreamBeginCapture), all allocations must
// come from a cudaMemPool that participates in the capture.  The default
// caching_allocator uses cudaMalloc which is NOT capture-safe.  This header
// provides the interface for a pool that can be used inside graph capture.
//
// STATUS: Interface defined; implementation requires a fully configured
// cudaMemPool_t and validated graph capture/replay semantics.  Do not use
// in production until ORDER 6 acceptance gates are met.
//
// Ordering requirements (from NVIDIA docs):
//   - Pool allocations inside a captured region are only valid within the
//     graph execution and must be freed within the same graph.
//   - Pool configuration (access, peer access) must be set BEFORE capture.
//   - cudaGraphMemAllocNode / cudaGraphMemFreeNode must be used for
//     allocation/free nodes, not cudaMallocAsync inside captured code.
//
// See: https://docs.nvidia.com/cuda/cuda-programming-guide/
//      04-special-topics/stream-ordered-memory-allocation.html

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

#include <cstddef>
#include <stdexcept>

#include "common/memory_export.h"
#include "common/memory_macros.h"
#include "gpu/gpu_runtime.h"

namespace memory::gpu
{

// Opaque token identifying a capture scope.  The caller obtains one from
// begin_capture() and passes it to allocations within the captured region.
struct capture_scope
{
    cudaStream_t stream{nullptr};
    bool         active{false};
};

// Minimal graph-pool interface.  A concrete implementation requires a
// cudaMemPool_t and must be validated against real CUDA graphs before use.
class MEMORY_VISIBILITY gpu_graph_pool
{
public:
    explicit gpu_graph_pool(int device = 0) : device_(device) {}

    // Begin a capture scope.  All allocations inside the scope must use the
    // returned capture_scope to link them to the graph node.
    capture_scope begin_capture(cudaStream_t stream)
    {
        if (active_)
        {
            throw std::logic_error("gpu_graph_pool: nested capture not supported");
        }
        active_ = true;
        return {stream, true};
    }

    // End a capture scope.  After this call the allocations are frozen inside
    // the graph; further calls to allocate() on those nodes are invalid.
    void end_capture(capture_scope const& scope)
    {
        if (!scope.active)
        {
            return;
        }
        active_ = false;
    }

    // Returns true if a capture is currently in progress.
    bool is_capturing() const noexcept { return active_; }

    int device() const noexcept { return device_; }

private:
    int  device_{0};
    bool active_{false};
};

}  // namespace memory::gpu

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP
