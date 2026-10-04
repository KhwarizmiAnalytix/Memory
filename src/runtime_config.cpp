/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include "runtime_config.h"

#include <cstddef>
#include <string>

#include "common/memory_macros.h"
#include "common/retained_operation_service.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/cuda_caching_allocator.h"
#endif

namespace memory
{

std::string effective_config_json([[maybe_unused]] int device)
{
    auto&             svc = retained_operation_service::instance();
    std::string       out = "{\"retained_service\":{\"max_pending\":";
    out += std::to_string(svc.max_pending());
    out += ",\"max_quarantined\":" + std::to_string(svc.max_quarantined());
    out += ",\"max_quarantined_bytes\":" + std::to_string(svc.max_quarantined_bytes());
    out += "}";
#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP  // Metal has no fraction accessor yet
    auto& cache = gpu::caching_allocator_for_device(device);
    out += ",\"gpu_cache\":{\"device\":" + std::to_string(device);
    out += ",\"memory_fraction\":" + std::to_string(cache.memory_fraction());
    out += ",\"max_cached_bytes\":" + std::to_string(cache.max_cached_bytes());
    out += std::string(",\"expandable_segments\":") + (cache.expandable_segments() ? "true" : "false");
    out += "}";
#endif
    out += "}";
    return out;
}

}  // namespace memory
