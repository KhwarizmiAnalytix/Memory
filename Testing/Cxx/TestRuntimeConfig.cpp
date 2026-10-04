/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

// Plan 8.5: effective limits are reported as set, and invalid values are rejected.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "MemoryTest.h"
#include "common/memory_macros.h"
#include "common/retained_operation_service.h"
#include "runtime_config.h"

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
#include "gpu/cuda_caching_allocator.h"
#endif

using memory::retained_operation_service;

MEMORYTEST(RuntimeConfig, reports_service_limits_as_set)
{
    auto& svc          = retained_operation_service::instance();
    auto const old_p   = svc.max_pending();
    auto const old_q   = svc.max_quarantined();
    auto const old_qb  = svc.max_quarantined_bytes();

    svc.set_max_pending(7);
    svc.set_max_quarantined(3, 4096);
    std::string const json = memory::effective_config_json(0);
    EXPECT_NE(json.find("\"max_pending\":7"), std::string::npos) << json;
    EXPECT_NE(json.find("\"max_quarantined\":3"), std::string::npos) << json;
    EXPECT_NE(json.find("\"max_quarantined_bytes\":4096"), std::string::npos) << json;

    svc.set_max_pending(old_p);
    svc.set_max_quarantined(old_q, old_qb);
    END_TEST();
}

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP
MEMORYTEST(RuntimeConfig, invalid_memory_fraction_is_rejected_and_effective_value_reported)
{
    auto& cache = memory::gpu::caching_allocator_for_device(0);
    double const original = cache.memory_fraction();
    cache.set_memory_fraction(0.5);
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), 0.0, -0.1, 1.5,
                       std::numeric_limits<double>::infinity()})
    {
        EXPECT_THROW(cache.set_memory_fraction(bad), std::invalid_argument) << bad;
        EXPECT_EQ(0.5, cache.memory_fraction()) << "a rejected value must not change the limit";
    }
    std::string const json = memory::effective_config_json(0);
    EXPECT_NE(json.find("\"memory_fraction\":0.5"), std::string::npos) << json;
    cache.set_memory_fraction(original);
    END_TEST();
}
#endif
