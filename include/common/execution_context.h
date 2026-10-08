/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstdint>

#include "common/device.h"
#include "common/memory_macros.h"

namespace memory
{

// Opaque stream handle (plan 4.2). CUDA/HIP streams are pointer types, so a
// cudaStream_t / hipStream_t converts to it implicitly; the backend converts back
// with static_cast inside the library, which keeps vendor runtime headers out of
// every public header. Metal and CPU ignore it.
// nullptr is the real CUDA/HIP default stream of the build's compile mode (legacy
// or per-thread), not "no stream".
using stream_handle_t = void*;

// Identifies where and on which stream work executes (plan §4.2).
//
// Stream semantics: on CUDA/HIP a null `stream` is the real default stream of
// the build's compile mode (legacy or per-thread), not "no stream"; it is a
// valid submission target and a valid recorded use. CPU and Metal ignore it.
//
// `dev` is the identity; device_type()/device_index() are compatibility
// accessors for code written against the former separate fields.
struct execution_context
{
    memory::device  dev{};
    stream_handle_t stream{nullptr};

    constexpr execution_context() noexcept = default;
    constexpr explicit execution_context(memory::device d, stream_handle_t s = nullptr) noexcept
        : dev(d), stream(s)
    {
    }
    constexpr execution_context(memory::device_enum type, int index, stream_handle_t s) noexcept
        : dev{type, static_cast<std::int16_t>(index)}, stream(s)
    {
    }

    constexpr memory::device_enum device_type() const noexcept { return dev.type; }
    constexpr int                 device_index() const noexcept { return dev.index; }

    static execution_context cpu() noexcept
    {
        return execution_context(memory::device::cpu(), nullptr);
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

    constexpr bool is_gpu() const noexcept { return dev.is_gpu(); }

    constexpr bool operator==(execution_context const& o) const noexcept
    {
        return dev == o.dev && stream == o.stream;
    }
    constexpr bool operator!=(execution_context const& o) const noexcept { return !(*this == o); }
};

}  // namespace memory
