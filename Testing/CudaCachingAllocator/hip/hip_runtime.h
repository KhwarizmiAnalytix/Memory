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
#include "fake_runtime.h"

// Aliases only the hip* spellings gpu/gpu_runtime.h's MEMORY_HAS_HIP branch
// #defines cuda* call sites to, and only the subset gpu/cuda_caching_allocator.cpp
// actually calls (see that file's own #include of <hip/hip_runtime.h> via
// gpu_runtime.h). This suite deliberately builds only the HIP-labeled variant
// (MEMORY_HAS_CUDA=0, MEMORY_HAS_HIP=1): the CUDA driver-API expandable-VM path
// (try_cu_vm_alloc, <cuda.h>) is out of scope to fake, and the HIP VM path is
// gated behind HIP_VERSION >= 50600000, which this fake never defines, so it
// does not compile in either -- the C++ logic under test (event insertion,
// budget checks, retry chain, block pooling) is shared between backends via
// gpu_runtime.h's aliasing, so exercising it under HIP labels is equally valid.
using hipError_t                          = cudaError_t;
using hipStream_t                         = cudaStream_t;
using hipEvent_t                          = cudaEvent_t;
// gpu/gpu_runtime.h's HIP branch unconditionally declares
// `using cudaMemcpyKind = hipMemcpyKind;` regardless of whether this TU
// actually calls a memcpy function, so the alias target must exist here too.
using hipMemcpyKind                       = cudaMemcpyKind;
inline constexpr auto hipSuccess          = cudaSuccess;
inline constexpr auto hipErrorOutOfMemory = cudaErrorMemoryAllocation;
inline constexpr auto hipErrorNotReady    = cudaErrorNotReady;
inline constexpr auto hipEventDisableTiming = cudaEventDisableTiming;
inline auto hipGetErrorString       = cudaGetErrorString;
inline auto hipGetLastError         = cudaGetLastError;
inline auto hipGetDevice            = cudaGetDevice;
inline auto hipSetDevice            = cudaSetDevice;
inline auto hipGetDeviceCount       = cudaGetDeviceCount;
inline auto hipMalloc               = cudaMalloc;
inline auto hipFree                 = cudaFree;
inline auto hipMemGetInfo           = cudaMemGetInfo;
inline auto hipEventCreateWithFlags = cudaEventCreateWithFlags;
inline auto hipEventRecord          = cudaEventRecord;
inline auto hipEventQuery           = cudaEventQuery;
inline auto hipEventSynchronize     = cudaEventSynchronize;
inline auto hipEventDestroy         = cudaEventDestroy;
