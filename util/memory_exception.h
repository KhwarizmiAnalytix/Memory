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

#include <fmt/format.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#include "common/memory_macros.h"

// ============================================================================
// memory::details::format_check_msg
//
// Formats a check-failure message with an optional fmt-style user message.
// Used by MEMORY_CHECK and MEMORY_CHECK_DEBUG.
// ============================================================================
namespace memory
{
namespace details
{
template <typename... Args>
inline std::string format_check_msg(
    const char* cond_str, fmt::format_string<Args...> fmt_str, Args&&... args)
{
    return fmt::format(
        "Check failed: {} - {}", cond_str, fmt::format(fmt_str, std::forward<Args>(args)...));
}

inline std::string format_check_msg(const char* cond_str)
{
    return fmt::format("Check failed: {}", cond_str);
}
}  // namespace details
}  // namespace memory

// ============================================================================
// MEMORY_THROW(format_str, ...)
//
// Throws std::runtime_error with an fmt-style formatted message.
//
// Usage:
//   MEMORY_THROW("allocating {} bytes failed", num_bytes);
//   MEMORY_THROW("CUDA error: {}", cuda_error_string(err));
// ============================================================================
#ifndef MEMORY_THROW
#define MEMORY_THROW(format_str, ...) \
    throw std::runtime_error(fmt::format(fmt::runtime(format_str), ##__VA_ARGS__))
#endif

#ifndef MEMORY_CHECK
#define MEMORY_CHECK(cond, ...)                                                               \
    if MEMORY_UNLIKELY (!(cond))                                                              \
    {                                                                                         \
        std::string _mem_check_msg = memory::details::format_check_msg(#cond, ##__VA_ARGS__); \
        MEMORY_THROW("{}", _mem_check_msg);                                                   \
    }
#endif

#ifndef MEMORY_LOG_WARNING
#define MEMORY_LOG_WARNING(format_str, ...) \
    std::fprintf(                           \
        stderr,                             \
        "[MEMORY WARNING] %s\n",            \
        fmt::format(fmt::runtime(format_str), ##__VA_ARGS__).c_str())
#endif

#ifndef MEMORY_LOG_INFO
#define MEMORY_LOG_INFO(format_str, ...) \
    std::fprintf(                        \
        stdout,                          \
        "[MEMORY INFO] %s\n",            \
        fmt::format(fmt::runtime(format_str), ##__VA_ARGS__).c_str())
#endif

#ifndef MEMORY_LOG_ERROR
#define MEMORY_LOG_ERROR(format_str, ...) \
    std::fprintf(                         \
        stderr,                           \
        "[MEMORY ERROR] %s\n",            \
        fmt::format(fmt::runtime(format_str), ##__VA_ARGS__).c_str())
#endif

#ifdef NDEBUG
#define MEMORY_CHECK_DEBUG(condition, ...)
#else
#define MEMORY_CHECK_DEBUG(condition, ...)      \
    do                                          \
    {                                           \
        MEMORY_CHECK(condition, ##__VA_ARGS__); \
    } while (0)
#endif

#ifdef NDEBUG
#define MEMORY_LOG_INFO_DEBUG(format_str, ...)
#else
#define MEMORY_LOG_INFO_DEBUG(format_str, ...)      \
    do                                              \
    {                                               \
        MEMORY_LOG_INFO(format_str, ##__VA_ARGS__); \
    } while (0)
#endif