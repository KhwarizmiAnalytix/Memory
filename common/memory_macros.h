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

// Standalone Memory module: use this header (memory_*.h) instead of Core common/macros.h.
#pragma once

#ifndef MEMORY_PORTABLE_MACROS_INCLUDED_
#define MEMORY_PORTABLE_MACROS_INCLUDED_

#include <cstddef>

// ============================================================================
// Minimum compiler version (Memory targets C++17+)
// ============================================================================
#if defined(_MSC_VER) && _MSC_VER < 1910
#error MEMORY requires MSVC++ 15.0 (Visual Studio 2017) or newer for C++17 support
#endif

#if !defined(__clang__) && defined(__GNUC__) && \
    (__GNUC__ < 7 || (__GNUC__ == 7 && __GNUC_MINOR__ < 1))
#error MEMORY requires GCC 7.1 or newer for C++17 support
#endif

#if defined(__clang__) && (__clang_major__ < 5)
#error MEMORY requires Clang 5.0 or newer for C++17 support
#endif

// ============================================================================
// Default heap / SIMD buffer alignment
// ============================================================================
#ifdef MEMORY_MOBILE
inline constexpr size_t MEMORY_ALIGNMENT = 16;
#else
inline constexpr size_t MEMORY_ALIGNMENT = 64;
#endif

// Export macros live in common/memory_export.h (MEMORY_API, MEMORY_VISIBILITY, etc.).

// ============================================================================
// Forced inline
// ============================================================================
#if defined(_MSC_VER)
#define MEMORY_FORCE_INLINE __forceinline
#elif defined(__INTEL_COMPILER)
#define MEMORY_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(__clang__)
#define MEMORY_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(__GNUC__)
#define MEMORY_FORCE_INLINE inline __attribute__((always_inline))
#else
#define MEMORY_FORCE_INLINE inline
#endif

// ============================================================================
// Attribute feature tests (used by branch hints and thread-safety annotations)
// ============================================================================
#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define MEMORY_HAVE_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define MEMORY_HAVE_CPP_ATTRIBUTE(x) 0
#endif

#ifdef __has_attribute
#define MEMORY_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define MEMORY_HAVE_ATTRIBUTE(x) 0
#endif

// ============================================================================
// Branch prediction hints
// ============================================================================
#if __cplusplus >= 202002L && MEMORY_HAVE_CPP_ATTRIBUTE(likely) && \
    MEMORY_HAVE_CPP_ATTRIBUTE(unlikely)
#define MEMORY_LIKELY(expr) (expr) [[likely]]
#define MEMORY_UNLIKELY(expr) (expr) [[unlikely]]
#elif defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define MEMORY_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define MEMORY_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define MEMORY_LIKELY(expr) (expr)
#define MEMORY_UNLIKELY(expr) (expr)
#endif

// ============================================================================
// Conditional constexpr (C++20+)
// ============================================================================
#if __cplusplus >= 202002L
#define MEMORY_FUNCTION_CONSTEXPR constexpr
#else
#define MEMORY_FUNCTION_CONSTEXPR
#endif

// ============================================================================
// [[nodiscard]] (C++17+)
// ============================================================================
#if __cplusplus >= 201703L
#define MEMORY_NODISCARD [[nodiscard]]
#else
#define MEMORY_NODISCARD
#endif

// ============================================================================
// Suppress unused parameters / variables
// ============================================================================
#if __cplusplus >= 201703L
#define MEMORY_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__)
#define MEMORY_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
#define MEMORY_UNUSED __pragma(warning(suppress : 4100))
#else
#define MEMORY_UNUSED
#endif

// ============================================================================
// __cxa_demangle availability (diagnostics)
// ============================================================================
#if defined(__ANDROID__) && (defined(__i386__) || defined(__x86_64__))
#define MEMORY_HAS_CXA_DEMANGLE 0
#elif (__GNUC__ >= 4 || (__GNUC__ >= 3 && __GNUC_MINOR__ >= 4)) && !defined(__mips__)
#define MEMORY_HAS_CXA_DEMANGLE 1
#elif defined(__clang__) && !defined(_MSC_VER)
#define MEMORY_HAS_CXA_DEMANGLE 1
#else
#define MEMORY_HAS_CXA_DEMANGLE 0
#endif

// ============================================================================
// Deleted special members
// ============================================================================
#define MEMORY_DELETE_CLASS(type)          \
    type()                         = delete; \
    type(const type&)              = delete; \
    type& operator=(const type& a) = delete; \
    type(type&&)                   = delete; \
    type& operator=(type&&)        = delete; \
    ~type()                        = delete;

#define MEMORY_DELETE_COPY_AND_MOVE(type)  \
private:                                     \
    type(const type&)              = delete; \
    type& operator=(const type& a) = delete; \
    type(type&&)                   = delete; \
    type& operator=(type&&)        = delete; \
                                             \
public:

#define MEMORY_DELETE_COPY(type)           \
    type(const type&)              = delete; \
    type& operator=(const type& a) = delete;

// ============================================================================
// Clang thread-safety analysis (no-op when unsupported)
// ============================================================================
#if MEMORY_HAVE_ATTRIBUTE(no_thread_safety_analysis)
#define MEMORY_NO_THREAD_SAFETY_ANALYSIS __attribute__((no_thread_safety_analysis))
#else
#define MEMORY_NO_THREAD_SAFETY_ANALYSIS
#endif

#if MEMORY_HAVE_ATTRIBUTE(guarded_by)
#define MEMORY_GUARDED_BY(x) __attribute__((guarded_by(x)))
#else
#define MEMORY_GUARDED_BY(x)
#endif

#if MEMORY_HAVE_ATTRIBUTE(exclusive_locks_required)
#define MEMORY_EXCLUSIVE_LOCKS_REQUIRED(...) \
    __attribute__((exclusive_locks_required(__VA_ARGS__)))
#else
#define MEMORY_EXCLUSIVE_LOCKS_REQUIRED(...)
#endif

#endif  // MEMORY_PORTABLE_MACROS_INCLUDED_
