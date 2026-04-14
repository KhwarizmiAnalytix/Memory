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

#include "../../Logging/util/exception.h"

namespace memory
{
using exception_mode     = logging::exception_mode;
using exception_category = logging::exception_category;
using source_location    = logging::source_location;
using exception          = logging::exception;

inline exception_mode get_exception_mode() noexcept
{
    return logging::get_exception_mode();
}
inline void set_exception_mode(exception_mode mode) noexcept
{
    logging::set_exception_mode(mode);
}
inline void init_exception_mode_from_env() noexcept
{
    logging::init_exception_mode_from_env();
}
namespace details = logging::details;
}  // namespace memory

// ============================================================================
// MEMORY_THROW(format_str, ...)
//
// Throws (or fatally logs, depending on the global exception mode) a
// Memory-scoped error with an fmt-style formatted message.
//
// Delegates to LOGGING_THROW so that exception mode, category, and
// source-location capture are handled consistently by the Logging module.
//
// Usage:
//   MEMORY_THROW("allocating {} bytes failed", num_bytes);
//   MEMORY_THROW("CUDA error: {}", cuda_error_string(err));
// ============================================================================
#ifndef MEMORY_THROW
#define MEMORY_THROW(format_str, ...) LOGGING_THROW(format_str, ##__VA_ARGS__)
#endif
