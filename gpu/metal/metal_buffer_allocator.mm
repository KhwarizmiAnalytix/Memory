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

#include "gpu/metal/metal_buffer_allocator.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu/metal/metal_caching_allocator.h"

namespace memory::metal
{
namespace
{
id<MTLDevice> system_device()
{
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}
}  // namespace

void* mtl_buffer_handle(void* host_ptr)
{
    void*  handle = nullptr;
    size_t offset = 0;
    if (!gpu::metal_caching_allocator_for_device(0).resolve_live_allocation(
            host_ptr, &handle, &offset))
    {
        return nullptr;
    }
    return handle;
}

std::size_t mtl_buffer_offset(void* host_ptr)
{
    void*  handle = nullptr;
    size_t offset = 0;
    if (!gpu::metal_caching_allocator_for_device(0).resolve_live_allocation(
            host_ptr, &handle, &offset))
    {
        return 0;
    }
    return offset;
}

bool device_available()
{
    return system_device() != nil;
}

}  // namespace memory::metal
