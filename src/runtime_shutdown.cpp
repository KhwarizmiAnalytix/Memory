/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <chrono>
#include <cstddef>

#include "common/memory_macros.h"
#include "common/retained_operation_service.h"
#include "common/transfer.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/cuda_caching_allocator.h"
#elif MEMORY_HAS_METAL
#include "gpu/metal/metal_caching_allocator.h"
#endif

namespace memory
{

size_t shutdown_runtime(std::chrono::milliseconds timeout)
{
    // 1. Stop admission and wait for retained operations: their owners may hold
    //    cache blocks and their events live in the token pool.
    size_t const remaining = retained_operation_service::instance().shutdown(timeout);
    if (remaining != 0)
    {
        return remaining;
    }
    // 2. Tokens are gone: drop the pooled events.
    detail::release_token_event_pool();
    // 3. Caches last: return cached segments to the driver.
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP || MEMORY_HAS_METAL
    gpu::shutdown();
#endif
    return 0;
}

}  // namespace memory
