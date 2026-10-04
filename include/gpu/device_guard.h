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

// RAII current-device switch for CUDA/HIP. Include only when
// MEMORY_HAS_CUDA || MEMORY_HAS_HIP.

#include <new>
#include <stdexcept>
#include <string>

#include "common/memory_export.h"
#include "gpu/gpu_runtime.h"

namespace memory::gpu
{

inline void throw_on_cuda_error(cudaError_t result, char const* what)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(result));
    }
}

class MEMORY_VISIBILITY device_guard
{
public:
    explicit device_guard(int device)
    {
        int current = 0;
        throw_on_cuda_error(cudaGetDevice(&current), "cudaGetDevice");
        prev_ = current;
        if (current != device)
        {
            throw_on_cuda_error(cudaSetDevice(device), "cudaSetDevice");
            changed_ = true;
        }
        active_ = true;
    }

    // Best-effort variant for noexcept teardown (allocator destructors): the
    // runtime may already be unloading (cudaErrorCudartUnloading). Skip the
    // switch rather than throw.
    //
    // Check active() before relying on the device: when activation failed the
    // current device is NOT @p device, and a caller that goes on to answer a
    // question about device-local state (an event query, say) must report the
    // failure instead of a result from the wrong context (plan 4.1).
    device_guard(int device, std::nothrow_t) noexcept
    {
        int         current = 0;
        cudaError_t status  = cudaGetDevice(&current);
        if (status == cudaSuccess)
        {
            prev_ = current;
            if (current == device)
            {
                active_ = true;
            }
            else
            {
                status = cudaSetDevice(device);
                if (status == cudaSuccess)
                {
                    changed_ = true;
                    active_  = true;
                }
            }
        }
        error_ = static_cast<int>(status);
    }

    /// True when @p device is the current device for this guard's lifetime.
    bool active() const noexcept { return active_; }
    /// Raw driver code of the failed activation step; 0 when active().
    int error() const noexcept { return error_; }

    device_guard(device_guard const&)            = delete;
    device_guard& operator=(device_guard const&) = delete;

    ~device_guard()
    {
        if (changed_)
        {
            (void)cudaSetDevice(prev_);
        }
    }

private:
    int  prev_{0};
    int  error_{0};
    bool changed_{false};
    bool active_{false};
};

}  // namespace memory::gpu
