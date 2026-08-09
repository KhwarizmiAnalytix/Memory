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

// Plain C++ surface for Metal kernel binding — no Objective-C types cross this
// boundary. Allocation/deallocation/copy go through memory::allocator<T> →
// gpu::caching_allocator_for_device (see allocator.h / metal_caching_allocator).
// These helpers only resolve live allocations for setBuffer:offset:.

#include <cstddef>

#include "common/memory_export.h"

namespace memory::metal
{
// Owning id<MTLBuffer> for a live allocation, bridged to void* (not retained).
// Returns nullptr if host_ptr is not a live METAL allocation.
MEMORY_API void* mtl_buffer_handle(void* host_ptr);

// Byte offset of host_ptr within its owning MTLBuffer (0 at segment base).
MEMORY_API std::size_t mtl_buffer_offset(void* host_ptr);

// True if MTLCreateSystemDefaultDevice() is available.
MEMORY_API bool device_available();

}  // namespace memory::metal
