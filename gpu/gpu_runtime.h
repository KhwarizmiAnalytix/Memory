/*
 * Quarisma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of Quarisma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@quarisma.co.uk
 * Website: https://www.quarisma.co.uk
 */

#pragma once

// CUDA / HIP runtime portability for the caching allocator and allocator<T>
// copy path. Call sites keep CUDA API spellings; HIP builds map them to hip*.
//
// Include only when MEMORY_HAS_CUDA || MEMORY_HAS_HIP.

#include "common/memory_macros.h"

#if MEMORY_HAS_CUDA

#include <cuda_runtime.h>

#elif MEMORY_HAS_HIP

#include <hip/hip_runtime.h>

using cudaError_t    = hipError_t;
using cudaStream_t   = hipStream_t;
using cudaEvent_t    = hipEvent_t;
using cudaMemcpyKind = hipMemcpyKind;

// HIP historically used hipErrorOutOfMemory; hipErrorMemoryAllocation is an
// alias on recent ROCm. Prefer OutOfMemory for broader toolkit coverage.
#define cudaSuccess hipSuccess
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorNotReady hipErrorNotReady
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventDestroy hipEventDestroy
#define cudaStreamSynchronize hipStreamSynchronize

#else
#error "gpu/gpu_runtime.h requires MEMORY_HAS_CUDA or MEMORY_HAS_HIP"
#endif
