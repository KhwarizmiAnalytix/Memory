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

#include <atomic>
#include <cstddef>

// Unified entry for the process-wide GPU caching allocator.
//
// CUDA, HIP, and Metal backends are compile-time exclusive (MEMORY_GPU_BACKEND).
// This header exposes a single type alias and registry name so allocator<T>
// (and other callers) do not fork on backend-specific symbols.
//
// Metal kernel binding still uses memory::metal::mtl_buffer_handle/offset —
// those stay in gpu/metal/metal_buffer_allocator.h (ObjC++ boundary).

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

#include "gpu/cuda_caching_allocator.h"

namespace memory::gpu
{
using caching_allocator = cuda_caching_allocator;
// caching_allocator_for_device(int) is declared in cuda_caching_allocator.h
}  // namespace memory::gpu

#elif MEMORY_HAS_METAL

#include "common/memory_macros.h"
#include "gpu/metal/metal_caching_allocator.h"

namespace memory::gpu
{
using caching_allocator = metal_caching_allocator;

// Same registry name as CUDA/HIP so allocate/free dispatch is identical.
MEMORY_FORCE_INLINE caching_allocator& caching_allocator_for_device(int device_index)
{ return metal_caching_allocator_for_device(device_index); }
}  // namespace memory::gpu

#endif

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
namespace memory::gpu
{
// Process-wide cache API (PyTorch torch.cuda.memory). Operates on the shared
// per-device allocator that allocator<T> / data_ptr / tensor already use.

MEMORY_FORCE_INLINE void empty_cache(int device_index = 0)
{ caching_allocator_for_device(device_index).empty_cache(); }

MEMORY_FORCE_INLINE size_t memory_allocated(int device_index = 0)
{
    return caching_allocator_for_device(device_index)
        .stats()
        .bytes_allocated.load(std::memory_order_relaxed);
}

MEMORY_FORCE_INLINE size_t max_memory_allocated(int device_index = 0)
{
    return caching_allocator_for_device(device_index)
        .stats()
        .peak_bytes_allocated.load(std::memory_order_relaxed);
}

MEMORY_FORCE_INLINE size_t memory_reserved(int device_index = 0)
{
    return caching_allocator_for_device(device_index)
        .stats()
        .bytes_reserved.load(std::memory_order_relaxed);
}

MEMORY_FORCE_INLINE size_t max_memory_reserved(int device_index = 0)
{
    return caching_allocator_for_device(device_index)
        .stats()
        .peak_bytes_reserved.load(std::memory_order_relaxed);
}

MEMORY_FORCE_INLINE unified_cache_stats memory_stats(int device_index = 0)
{ return caching_allocator_for_device(device_index).stats(); }

MEMORY_FORCE_INLINE void reset_peak_memory_stats(int device_index = 0)
{ caching_allocator_for_device(device_index).reset_peak_stats(); }

MEMORY_FORCE_INLINE void set_memory_fraction(double fraction, int device_index = 0)
{ caching_allocator_for_device(device_index).set_memory_fraction(fraction); }

MEMORY_FORCE_INLINE double memory_fraction(int device_index = 0)
{ return caching_allocator_for_device(device_index).memory_fraction(); }

MEMORY_FORCE_INLINE void set_max_cached_bytes(size_t bytes, int device_index = 0)
{ caching_allocator_for_device(device_index).set_max_cached_bytes(bytes); }

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
MEMORY_FORCE_INLINE void set_expandable_segments(bool enabled, int device_index = 0)
{ caching_allocator_for_device(device_index).set_expandable_segments(enabled); }

MEMORY_FORCE_INLINE bool expandable_segments(int device_index = 0)
{ return caching_allocator_for_device(device_index).expandable_segments(); }
#endif

MEMORY_FORCE_INLINE size_t device_total_memory(int device_index = 0)
{ return caching_allocator_for_device(device_index).device_total_memory(); }

MEMORY_FORCE_INLINE void record_memory_history(
    bool enabled, size_t max_entries = kDefaultMemoryHistoryEntries, int device_index = 0)
{ caching_allocator_for_device(device_index).record_memory_history(enabled, max_entries); }

MEMORY_FORCE_INLINE gpu_memory_snapshot memory_snapshot(int device_index = 0)
{ return caching_allocator_for_device(device_index).snapshot(); }
}  // namespace memory::gpu
#endif
