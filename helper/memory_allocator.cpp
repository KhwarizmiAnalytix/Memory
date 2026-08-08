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

#include "memory_allocator.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>  // for std::memset

#include "common/memory_macros.h"
#include "util/memory_exception.h"

#if MEMORY_HAS_TBB

#ifdef _MSC_VER
#pragma push_macro("__TBB_NO_IMPLICIT_LINKAGE")
#define __TBB_NO_IMPLICIT_LINKAGE 1  // NOLINT(bugprone-reserved-identifier)
#endif

#include <tbb/scalable_allocator.h>

#ifdef _MSC_VER
#pragma pop_macro("__TBB_NO_IMPLICIT_LINKAGE")
#endif
#endif

#if MEMORY_HAS_NUMA
#include "common/numa.h"
#endif  // MEMORY_HAS_NUMA

#if defined(__APPLE__)
#include <malloc/malloc.h>  // for malloc_size
#elif !defined(_MSC_VER) && !defined(__MINGW32__) && !defined(__MINGW64__)
#include <malloc.h>  // for malloc_usable_size
#endif

#if MEMORY_HAS_MIMALLOC
#include <mimalloc.h>
#endif

//#include "logger/logger.h"

namespace memory::cpu::memory_allocator
{

void* allocate(std::size_t nbytes, std::size_t alignment, init_policy_enum init)
{
    MEMORY_CHECK(
        static_cast<std::ptrdiff_t>(nbytes) > 0,
        "cpu allocate() called with negative or zero size: {}",
        nbytes);

    MEMORY_CHECK_DEBUG(
        is_valid_alignment(alignment),
        "cpu allocate() called with invalid alignment: {} (must be power of 2 >= {})",
        alignment,
        sizeof(void*));

    void* ptr = nullptr;

    // Platform-specific allocation
#if MEMORY_HAS_MIMALLOC
    ptr = mi_aligned_alloc(alignment, nbytes);
#elif MEMORY_HAS_TBB
    ptr = scalable_aligned_malloc(nbytes, alignment);
#elif defined(__ANDROID__)
    ptr = memalign(alignment, nbytes);
#elif defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    ptr = _aligned_malloc(nbytes, alignment);  // Fixed syntax error
#else
    // POSIX systems
    if (alignment < sizeof(void*))
    {
        ptr = malloc(nbytes);
    }
    else
    {
        // cppcheck-suppress syntaxError
        if MEMORY_UNLIKELY (posix_memalign(&ptr, alignment, nbytes) != 0)
        {
            return nullptr;
        }
    }
#endif
    // cppcheck-suppress syntaxError
    if MEMORY_UNLIKELY (ptr == nullptr)
    {
        return nullptr;
    }

    // NUMA optimization
#if MEMORY_HAS_NUMA
    NUMAMove(ptr, nbytes, GetCurrentNUMANode());
#endif

    // Memory initialization
    switch (init)
    {
    case init_policy_enum::ZERO:
        std::memset(ptr, 0, nbytes);
        break;
#ifndef NDEBUG
    case init_policy_enum::PATTERN:
        std::memset(ptr, 0xCC, nbytes);
        break;
#endif
    case init_policy_enum::UNINITIALIZED:
    default:
        // Do nothing - fastest option
        break;
    }

    return ptr;
}

void free(void* ptr, MEMORY_UNUSED std::size_t nbytes) noexcept
{
    if MEMORY_LIKELY (ptr != nullptr)
    {
#if MEMORY_HAS_MIMALLOC
        mi_free(ptr);
#elif MEMORY_HAS_TBB
        scalable_aligned_free(ptr);
#elif defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
        _aligned_free(ptr);  // Fixed syntax error
#else
        ::free(ptr);
#endif
    }
}

std::size_t usable_size(const void* ptr) noexcept
{
    if (ptr == nullptr)
    {
        return 0;
    }

    // Must mirror the backend selection order used by allocate()/free().
#if MEMORY_HAS_MIMALLOC
    return mi_usable_size(ptr);
#elif MEMORY_HAS_TBB
    return scalable_msize(const_cast<void*>(ptr));
#elif defined(__ANDROID__)
    return malloc_usable_size(const_cast<void*>(ptr));
#elif defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    // _aligned_malloc blocks require _aligned_msize with the original
    // alignment, which is not available here.
    return 0;
#elif defined(__APPLE__)
    return malloc_size(ptr);
#else
    return malloc_usable_size(const_cast<void*>(ptr));
#endif
}

void* allocate_tbb(MEMORY_UNUSED std::size_t nbytes, MEMORY_UNUSED std::size_t alignment)
{
#if MEMORY_HAS_TBB
    return scalable_aligned_malloc(nbytes, alignment);
#else
    return nullptr;
#endif
}

void free_tbb(MEMORY_UNUSED void* ptr, MEMORY_UNUSED std::size_t nbytes) noexcept
{
#if MEMORY_HAS_TBB
    scalable_aligned_free(ptr);
#endif
}

void* allocate_mi(MEMORY_UNUSED std::size_t nbytes, MEMORY_UNUSED std::size_t alignment)
{
#if MEMORY_HAS_MIMALLOC
    return mi_malloc_aligned(nbytes, alignment);
#else
    return nullptr;
#endif
}

void free_mi(MEMORY_UNUSED void* ptr, MEMORY_UNUSED std::size_t nbytes) noexcept
{
#if MEMORY_HAS_MIMALLOC
    mi_free(ptr);
#endif
}

bool has_stats() noexcept
{
#if MEMORY_HAS_MIMALLOC && MEMORY_HAS_MIMALLOC_STATS
    return true;
#else
    return false;
#endif
}

void stats_print() noexcept
{
#if MEMORY_HAS_MIMALLOC && MEMORY_HAS_MIMALLOC_STATS
    mi_stats_merge();        // fold this thread's counters into the process totals
    mi_stats_print(nullptr); // dumps to stderr
#endif
}

bool process_info(process_memory_info& info) noexcept
{
#if MEMORY_HAS_MIMALLOC && MEMORY_HAS_MIMALLOC_STATS
    mi_process_info(
        &info.elapsed_msecs,
        &info.user_msecs,
        &info.system_msecs,
        &info.current_rss,
        &info.peak_rss,
        &info.current_commit,
        &info.peak_commit,
        &info.page_faults);
    return true;
#else
    info = process_memory_info{};
    return false;
#endif
}
}  // namespace memory::cpu::memory_allocator
