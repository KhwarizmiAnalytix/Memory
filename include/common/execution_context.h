/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include "common/device.h"
#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/gpu_runtime.h"
#endif

namespace memory
{

// Canonical stream handle: cudaStream_t under CUDA/HIP, void* otherwise.
// nullptr is the real CUDA/HIP per-thread default stream, not "no stream".
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
using stream_handle_t = cudaStream_t;
#else
using stream_handle_t = void*;
#endif

// Identifies where and on which stream work executes.  Replaces scattered
// (device_enum, int, stream) triples so copy, workspace, and arena APIs
// carry a single context argument instead of three.
struct execution_context
{
    memory::device_enum    device_type{memory::device_enum::CPU};
    int            device_index{0};
    stream_handle_t stream{nullptr};  // nullptr = CUDA/HIP per-thread default

    static execution_context cpu() noexcept
    {
        return {memory::device_enum::CPU, 0, nullptr};
    }

    static execution_context cuda(int index = 0, stream_handle_t s = nullptr) noexcept
    {
        return {memory::device_enum::CUDA, index, s};
    }

    static execution_context hip(int index = 0, stream_handle_t s = nullptr) noexcept
    {
        return {memory::device_enum::HIP, index, s};
    }

    static execution_context metal(int index = 0) noexcept
    {
        return {memory::device_enum::METAL, index, nullptr};
    }

    bool is_gpu() const noexcept
    {
        return device_type == memory::device_enum::CUDA || device_type == memory::device_enum::HIP ||
               device_type == memory::device_enum::METAL;
    }

    bool operator==(execution_context const& o) const noexcept
    {
        return device_type == o.device_type && device_index == o.device_index &&
               stream == o.stream;
    }
    bool operator!=(execution_context const& o) const noexcept { return !(*this == o); }
};

}  // namespace memory
