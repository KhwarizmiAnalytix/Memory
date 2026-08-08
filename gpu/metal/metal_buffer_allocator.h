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

// Plain C++ surface over Metal buffer allocation — no Objective-C types cross this
// boundary, so ordinary (non-.mm) translation units can allocate/free/copy Metal
// storage without becoming Objective-C++ themselves.
//
// Allocation/deallocation forward to memory::gpu::metal_caching_allocator (segment
// cache with Shared-storage MTLBuffers). The pointer returned by allocate() is
// host-dereferenceable and may be an offset into a larger packed segment; use
// mtl_buffer_handle() + mtl_buffer_offset() when binding as a kernel argument.

#include <cstddef>

#include "common/memory_export.h"

namespace memory::metal
{
// Allocates via the process-wide Metal caching allocator (device 0) and returns a
// host-visible pointer (buffer.contents + block offset). Throws std::bad_alloc.
MEMORY_API void* allocate(std::size_t bytes);

// Returns the block to the caching allocator (may be retained for reuse).
// No-op on nullptr. Throws if ptr is not a live allocation of the cache.
MEMORY_API void deallocate(void* host_ptr);

// memcpy-equivalent copy between any combination of Metal/CPU pointers — valid
// because MTLResourceStorageModeShared buffers are plain host-addressable memory.
MEMORY_API void copy(const void* src, void* dst, std::size_t bytes);

// Returns the owning id<MTLBuffer> for a live allocation, bridged to void*
// (not retained — keep the allocation alive). Returns nullptr if untracked.
MEMORY_API void* mtl_buffer_handle(void* host_ptr);

// Byte offset of host_ptr within its owning MTLBuffer (0 at segment base).
// Returns 0 if host_ptr is not a live allocation.
MEMORY_API std::size_t mtl_buffer_offset(void* host_ptr);

// True if a Metal device (MTLCreateSystemDefaultDevice) is available.
MEMORY_API bool device_available();

}  // namespace memory::metal
