/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#pragma once
#include <atomic>  // for allocation statistics

#if __cplusplus >= 202002L
#include <bit>  // for std::has_single_bit (C++20)
#endif

#include <cstdlib>

#include "common/memory_export.h"
#include "common/memory_macros.h"

namespace memory
{
namespace cpu
{
namespace memory_allocator
{
// Memory initialization options
enum class init_policy_enum : uint8_t
{
    UNINITIALIZED = 0,  // Don't initialize memory (fastest)
    ZERO          = 1,  // Zero-fill memory
    PATTERN       = 2   // Fill with debug pattern (debug builds only)
};

// Validate alignment is power of 2 and >= sizeof(void*)
MEMORY_FORCE_INLINE bool is_valid_alignment(std::size_t alignment) noexcept
{
    return alignment >= sizeof(void*) &&
#if __cplusplus >= 202002L
           std::has_single_bit(alignment);  // C++20
#else
           (alignment & (alignment - 1)) == 0;  // Power of 2 check
#endif
}

// Get default alignment for the platform
MEMORY_FORCE_INLINE MEMORY_FUNCTION_CONSTEXPR std::size_t default_alignment() noexcept
{
    return MEMORY_ALIGNMENT;
}

MEMORY_API void* allocate(
    std::size_t      nbytes,
    std::size_t      alignment = default_alignment(),
    init_policy_enum init      = init_policy_enum::UNINITIALIZED);

MEMORY_API void free(void* ptr, std::size_t nbytes = 0) noexcept;

/**
 * @brief Returns the usable size of a block previously returned by allocate().
 *
 * Queries the underlying allocator (mimalloc, TBB, or the platform malloc)
 * for the actual size of the allocated block, which may exceed the size
 * originally requested.
 *
 * @param ptr Pointer returned by allocate(), or nullptr
 * @return Usable size in bytes, or 0 if ptr is nullptr or the underlying
 *         allocator cannot report block sizes (e.g. the MSVC _aligned_malloc
 *         fallback, which would require the original alignment).
 */
MEMORY_API std::size_t usable_size(const void* ptr) noexcept;

// TBB-specific allocation and deallocation
MEMORY_API void* allocate_tbb(std::size_t nbytes, std::size_t alignment = default_alignment());

MEMORY_API void free_tbb(void* ptr, std::size_t nbytes = 0) noexcept;

// mimalloc-specific allocation and deallocation
MEMORY_API void* allocate_mi(std::size_t nbytes, std::size_t alignment = default_alignment());

MEMORY_API void free_mi(void* ptr, std::size_t nbytes = 0) noexcept;

// Zero-initialized allocation
MEMORY_FORCE_INLINE void* allocate_zero(
    std::size_t nbytes, std::size_t alignment = default_alignment())
{
    return allocate(nbytes, alignment, init_policy_enum::ZERO);
}

// ---------------------------------------------------------------------------
// mimalloc statistics — available only when the library is configured with
// MEMORY_ENABLE_MIMALLOC_STATS=ON (compiles the vendored mimalloc with
// MI_STAT=1; mimalloc release builds otherwise compile all counters out).
// Because mimalloc is linked with MI_OVERRIDE=OFF, the statistics cover only
// allocations that went through this namespace — not the process's malloc.
// ---------------------------------------------------------------------------

/** OS-level process memory counters as reported by mimalloc. */
struct MEMORY_VISIBILITY process_memory_info
{
    std::size_t elapsed_msecs{0};
    std::size_t user_msecs{0};
    std::size_t system_msecs{0};
    std::size_t current_rss{0};
    std::size_t peak_rss{0};
    std::size_t current_commit{0};
    std::size_t peak_commit{0};
    std::size_t page_faults{0};
};

/** True when mimalloc statistics are compiled in (MEMORY_ENABLE_MIMALLOC_STATS=ON). */
MEMORY_API bool has_stats() noexcept;

/**
 * @brief Log mimalloc allocator statistics via MEMORY_LOG_INFO.
 *
 * Merges the calling thread's counters into the process totals first
 * (mi_stats_merge), then dumps line-by-line through mi_stats_print_out into
 * the Memory logger. No-op when statistics are not compiled in. Equivalent
 * env-var route: MIMALLOC_SHOW_STATS=1 prints at process exit (to stderr,
 * outside this API).
 */
MEMORY_API void stats_print() noexcept;

/**
 * @brief Fill @p info with process memory counters (RSS, commit, page faults).
 * @return false — with @p info zeroed — when statistics are not compiled in.
 */
MEMORY_API bool process_info(process_memory_info& info) noexcept;

}  // namespace memory_allocator
}  // namespace cpu
}  // namespace memory
