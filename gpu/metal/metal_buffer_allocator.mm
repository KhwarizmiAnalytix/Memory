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

#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>

namespace memory::metal
{
namespace
{
// Process-wide default device — every Mac has at least an integrated GPU, so this
// is realistically always non-nil once Metal.framework is linked.
id<MTLDevice> device()
{
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}

// buffer.contents (host pointer) -> owning id<MTLBuffer>. Pointer-keyed bookkeeping,
// chosen rather than adding a backing-handle field to data_ptr.h — keeps data_ptr's
// shape, and the CUDA/HIP paths through it, completely untouched.
std::mutex&                                  buffer_map_mutex()
{
    static std::mutex m;
    return m;
}
std::unordered_map<void*, id<MTLBuffer>>& buffer_map()
{
    static std::unordered_map<void*, id<MTLBuffer>> m;
    return m;
}
}  // namespace

void* allocate(std::size_t bytes)
{
    id<MTLDevice> dev = device();
    if (dev == nil)
    {
        throw std::bad_alloc();
    }

    id<MTLBuffer> buffer = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer == nil)
    {
        throw std::bad_alloc();
    }

    void* ptr = [buffer contents];

    std::lock_guard<std::mutex> lock(buffer_map_mutex());
    buffer_map()[ptr] = buffer;
    return ptr;
}

void deallocate(void* host_ptr)
{
    if (host_ptr == nullptr)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(buffer_map_mutex());
    buffer_map().erase(host_ptr);  // erasing the id<MTLBuffer> releases it under ARC
}

void copy(const void* src, void* dst, std::size_t bytes)
{
    if (src == nullptr || dst == nullptr || bytes == 0)
    {
        return;
    }
    std::memcpy(dst, src, bytes);
}

void* mtl_buffer_handle(void* host_ptr)
{
    std::lock_guard<std::mutex> lock(buffer_map_mutex());
    auto                        it = buffer_map().find(host_ptr);
    if (it == buffer_map().end())
    {
        return nullptr;
    }
    return (__bridge void*)it->second;
}

bool device_available()
{
    return device() != nil;
}

}  // namespace memory::metal
