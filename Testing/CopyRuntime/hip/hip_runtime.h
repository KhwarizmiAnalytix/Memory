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
#include "../fake_runtime.h"

// The real gpu/gpu_runtime.h HIP branch #defines cuda* call sites to hip*
// spellings; alias each one it maps (and this suite reaches) back to the fake
// CUDA runtime. No gpu/gpu_runtime.h shadow is used: MSVC searches a dep's
// include path before this package's, so a shadow is not portable.
using hipError_t                              = cudaError_t;
using hipStream_t                             = cudaStream_t;
using hipEvent_t                              = cudaEvent_t;
using hipMemcpyKind                           = cudaMemcpyKind;
using hipDeviceProp_t                         = cudaDeviceProp;
inline constexpr auto hipSuccess              = cudaSuccess;
inline constexpr auto hipErrorOutOfMemory     = cudaErrorMemoryAllocation;
inline constexpr auto hipErrorNotReady        = cudaErrorNotReady;
inline constexpr auto hipEventDisableTiming   = cudaEventDisableTiming;
inline constexpr auto hipHostMallocPortable   = cudaHostAllocPortable;
inline constexpr auto hipMemcpyHostToDevice   = cudaMemcpyHostToDevice;
inline constexpr auto hipMemcpyDeviceToHost   = cudaMemcpyDeviceToHost;
inline constexpr auto hipMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
inline auto           hipGetErrorString       = cudaGetErrorString;
inline auto           hipGetLastError         = cudaGetLastError;
inline auto           hipGetDeviceCount       = cudaGetDeviceCount;
inline auto           hipGetDevice            = cudaGetDevice;
inline auto           hipGetDeviceProperties  = cudaGetDeviceProperties;
inline auto           hipRuntimeGetVersion    = cudaRuntimeGetVersion;
inline auto           hipDriverGetVersion     = cudaDriverGetVersion;
inline auto           hipSetDevice            = cudaSetDevice;
inline auto           hipDeviceSynchronize    = cudaDeviceSynchronize;
inline auto           hipStreamQuery          = cudaStreamQuery;
inline auto           hipStreamSynchronize    = cudaStreamSynchronize;
inline auto           hipStreamWaitEvent      = cudaStreamWaitEvent;
inline auto           hipDeviceCanAccessPeer  = cudaDeviceCanAccessPeer;
inline auto           hipHostMalloc           = cudaHostAlloc;
inline auto           hipHostFree             = cudaFreeHost;
inline auto           hipEventCreateWithFlags = cudaEventCreateWithFlags;
inline auto           hipEventRecord          = cudaEventRecord;
inline auto           hipEventQuery           = cudaEventQuery;
inline auto           hipEventSynchronize     = cudaEventSynchronize;
inline auto           hipEventDestroy         = cudaEventDestroy;
inline auto           hipMemcpyAsync          = cudaMemcpyAsync;
inline auto           hipMemcpyPeerAsync      = cudaMemcpyPeerAsync;
