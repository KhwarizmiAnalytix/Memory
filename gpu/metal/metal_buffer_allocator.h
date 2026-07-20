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
// storage without becoming Objective-C++ themselves. The .mm implementation targets
// Apple Silicon unified memory (MTLResourceStorageModeShared): the pointer returned
// by allocate() is directly host-dereferenceable and is also what the GPU sees.

#include <cstddef>

#include "common/memory_export.h"

namespace memory::metal
{
// Allocates a shared-storage MTLBuffer and returns its `.contents` pointer.
// Throws std::bad_alloc on failure (no Metal device, or allocation failure).
MEMORY_API void* allocate(std::size_t bytes);

// Releases the MTLBuffer previously returned by allocate(). No-op on nullptr.
MEMORY_API void deallocate(void* host_ptr);

// memcpy-equivalent copy between any combination of Metal/CPU pointers — valid
// because MTLResourceStorageModeShared buffers are plain host-addressable memory.
MEMORY_API void copy(const void* src, void* dst, std::size_t bytes);

// Returns the owning id<MTLBuffer> for a pointer previously returned by allocate(),
// bridged to void* (ARC-retained) for Vectorization's dispatch layer to bind as a
// compute-kernel argument. Returns nullptr if host_ptr is not a tracked allocation.
MEMORY_API void* mtl_buffer_handle(void* host_ptr);

// True if a Metal device (MTLCreateSystemDefaultDevice) is available on this machine.
MEMORY_API bool device_available();

}  // namespace memory::metal
