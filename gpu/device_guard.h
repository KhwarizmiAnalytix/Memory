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
    }

    // Best-effort variant for noexcept teardown (allocator destructors): the
    // runtime may already be unloading (cudaErrorCudartUnloading). Skip the
    // switch rather than throw.
    device_guard(int device, std::nothrow_t) noexcept
    {
        int current = 0;
        if (cudaGetDevice(&current) == cudaSuccess)
        {
            prev_ = current;
            if (current != device && cudaSetDevice(device) == cudaSuccess)
            {
                changed_ = true;
            }
        }
    }

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
    bool changed_{false};
};

}  // namespace memory::gpu
