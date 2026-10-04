/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <string>

#include "common/memory_export.h"

namespace memory
{

/**
 * @brief Effective runtime limits as one JSON object (plan 8.5).
 *
 * Reports the values in force now, after any setter calls, so a benchmark or
 * test manifest records what actually ran rather than what was requested:
 * `retained_service` (admission and quarantine limits; 0 = unlimited) and, in a
 * GPU build, `gpu_cache` for @p device (memory fraction, byte budget derived
 * from it, cached-bytes limit, expandable segments). Setters validate on entry
 * (`set_memory_fraction` throws `std::invalid_argument` for NaN, <= 0 and > 1
 * and leaves the old value); this function only reads.
 */
MEMORY_API std::string effective_config_json(int device = 0);

}  // namespace memory
