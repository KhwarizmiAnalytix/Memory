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
}

#elif MEMORY_HAS_METAL

#include "gpu/metal/metal_caching_allocator.h"

#include "common/memory_macros.h"

namespace memory::gpu
{
using caching_allocator = metal_caching_allocator;

// Same registry name as CUDA/HIP so allocate/free dispatch is identical.
MEMORY_FORCE_INLINE caching_allocator& caching_allocator_for_device(int device_index)
{
    return metal_caching_allocator_for_device(device_index);
}
}

#endif
