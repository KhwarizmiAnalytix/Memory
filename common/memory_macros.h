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

// Shared with logging_macros.h and Core common/macros.h — one portable body per translation unit.
#ifndef MEMORY_PORTABLE_MACROS_INCLUDED_
#define MEMORY_PORTABLE_MACROS_INCLUDED_

#include <cstring>
#include <string>
#include <type_traits>  // for std::underlying_type
#include <typeinfo>

// ============================================================================
// Section 1: Minimum Compiler Version Guards
//
// Memory requires C++17 features (if-constexpr, structured bindings,
// std::optional, etc.). These #error directives abort compilation immediately
// when a known-too-old compiler is detected, giving a clear diagnostic
// instead of cryptic template errors later.
//
//   MSVC  : _MSC_VER >= 1910  →  Visual Studio 2017 (15.0)
//   GCC   : >= 7.1            →  first GCC with full C++17 core language
//   Clang : >= 5.0            →  first Clang with full C++17 core language
// ============================================================================
#if defined(_MSC_VER) && _MSC_VER < 1910  // VS2017 15.0 minimum for C++17
#error MEMORY requires MSVC++ 15.0 (Visual Studio 2017) or newer for C++17 support
#endif

#if !defined(__clang__) && defined(__GNUC__) && \
    (__GNUC__ < 7 || (__GNUC__ == 7 && __GNUC_MINOR__ < 1))  // GCC 7.1+ for C++17
#error MEMORY requires GCC 7.1 or newer for C++17 support
#endif

#if defined(__clang__) && (__clang_major__ < 5)  // Clang 5.0+ for C++17
#error MEMORY requires Clang 5.0 or newer for C++17 support
#endif

// ============================================================================
// Section 2: Memory Alignment Constant
//
// MEMORY_ALIGNMENT controls the default byte boundary used for heap
// allocations and SIMD buffers.
//
//   Mobile builds (MEMORY_MOBILE defined):
//     16 bytes — satisfies ARM NEON (AArch32 / AArch64) and x86 SSE.
//
//   Desktop / server builds (default):
//     64 bytes — matches a typical cache line and covers every SIMD ISA up
//     to AVX-512 (which needs 64-byte alignment for aligned loads/stores).
//
// The value is an inline constexpr so it participates in constant-expression
// evaluation (e.g. static_assert, template arguments) without ODR issues.
// ============================================================================
#ifdef MEMORY_MOBILE
// Use 16-byte alignment on mobile
// - ARM NEON AArch32 and AArch64
// - x86[-64] < AVX
inline constexpr size_t MEMORY_ALIGNMENT = 16;
#else
// Use 64-byte alignment should be enough for computation up to AVX512.
inline constexpr size_t MEMORY_ALIGNMENT = 64;
#endif

// ============================================================================
// Section 3: DLL Export / Import Macros
//
// The Memory library uses common/memory_export.h (MEMORY_API, etc.), not Core export.h.
// ============================================================================

// ============================================================================
// Section 4: Cross-Compiler alignas Portability
//
// C++11 introduced the standard 'alignas' keyword. MSVC versions before 2015
// (< 1900) do not support it and require __declspec(align(N)) instead.
//
// MEMORY_ALIGN(N) abstracts this difference:
//   - MSVC < 2015      : __declspec(align(N))
//   - MSVC >= 2015     : standard alignas(N)  (preferred path)
//   - GCC              : __attribute__((aligned(N)))
//   - Intel ICC        : (no definition — ICC supports GCC syntax but the
//                         branch is intentionally left empty; ICC users are
//                         expected to pass the GCC-compatible flags)
//
// Usage:
//   MEMORY_ALIGN(64) float buffer[1024];
// ============================================================================
#if defined(_MSC_VER)
#if (_MSC_VER < 1900)
// Visual studio until 2015 is not supporting standard 'alignas' keyword
#ifdef alignas
// This check can be removed when verified that for all other versions alignas
// works as requested
#error "MEMORY error: alignas already defined"
#else
#define alignas(alignment) __declspec(align(alignment))
#endif
#endif

#ifdef alignas
#define MEMORY_ALIGN(alignment) alignas(alignment)
#else
#define MEMORY_ALIGN(alignment) __declspec(align(alignment))
#endif
#elif defined(__GNUC__)
#define MEMORY_ALIGN(alignment) __attribute__((aligned(alignment)))
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#endif

// ============================================================================
// Section 5: Restrict Pointer Qualifier
//
// The 'restrict' keyword (C99) and its compiler extensions tell the optimizer
// that two pointers passed to a function do not alias each other, enabling
// more aggressive auto-vectorization and load/store reordering.
//
// C++ has no standard 'restrict', so each compiler exposes its own spelling:
//   - MSVC            : __restrict
//   - GCC / Clang     : __restrict__
//   - C99 compiler    : restrict  (standard keyword)
//   - Unknown         : empty    (safe fallback, no optimization hint)
//
// Usage:
//   void copy(float* MEMORY_RESTRICT dst, const float* MEMORY_RESTRICT src, int n);
// ============================================================================
#if defined(_MSC_VER)
#define MEMORY_RESTRICT __restrict
#elif defined(__cplusplus)
// C++ doesn't have standard restrict, use compiler extensions
#if defined(__GNUC__) || defined(__clang__)
#define MEMORY_RESTRICT __restrict__
#else
#define MEMORY_RESTRICT
#endif
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
// C99 and later
#define MEMORY_RESTRICT restrict
#else
// Fallback for older compilers
#define MEMORY_RESTRICT
#endif

// ============================================================================
// Section 6: GPU Count Constant
//
// MEMORY_COMPILE_TIME_MAX_GPUS sets the upper bound on how many GPU devices
// the library will ever manage in a single process. It is used to size
// fixed-length arrays and bitmasks that track per-device state, avoiding
// dynamic allocation in hot paths.
//
// 16 covers all current multi-GPU workstations and most HPC nodes; raise it
// if a target system has more devices.
// ============================================================================
inline constexpr int MEMORY_COMPILE_TIME_MAX_GPUS = 16;

// ============================================================================
// Section 7: Calling Convention and Forced-Inline Hints
//
// MEMORY_VECTORCALL
//   On MSVC, __vectorcall passes up to six floating-point / SIMD arguments in
//   XMM/YMM registers instead of on the stack, reducing prologue overhead for
//   SIMD-heavy kernels.  On every other compiler the macro is empty because
//   the ABI already passes vector arguments in registers by default.
//
// MEMORY_FORCE_INLINE
//   Instructs the compiler to always inline the marked function, even when
//   normal heuristics would reject it (e.g. in debug or at -O1).
//   - MSVC            : __forceinline
//   - GCC / Clang / ICC : inline __attribute__((always_inline))
//   - Fallback        : plain inline  (hint only, compiler may still reject)
//
// NOTE: Forced inlining can hurt compile times and I-cache pressure.  Apply
//       only to functions that are provably hot and small.
//
// Compiler priority: MSVC → ICC → Clang → GCC → generic.
// ============================================================================
#if defined(_MSC_VER)
#define MEMORY_VECTORCALL __vectorcall
#define MEMORY_FORCE_INLINE __forceinline

#elif defined(__INTEL_COMPILER)
#define MEMORY_VECTORCALL
#define MEMORY_FORCE_INLINE inline __attribute__((always_inline))

#elif defined(__clang__)
#define MEMORY_VECTORCALL
#define MEMORY_FORCE_INLINE inline __attribute__((always_inline))

#elif defined(__GNUC__)
#define MEMORY_VECTORCALL
#define MEMORY_FORCE_INLINE inline __attribute__((always_inline))

#else
#define MEMORY_VECTORCALL
#define MEMORY_FORCE_INLINE inline
#endif

// ============================================================================
// Section 8: Code-Flow Hint Attributes
//
// These attributes help the compiler (and the CPU's branch predictor) lay out
// code more efficiently by marking functions that are rarely or never reached.
//
// MEMORY_NORETURN
//   Marks a function that never returns (e.g. fatal error handlers, functions
//   that always throw).  Allows the compiler to omit the return path and
//   suppress false "control reaches end of non-void function" warnings.
//   - GCC / Clang / Apple Clang : __attribute__((noreturn))
//   - MSVC                      : __declspec(noreturn)
//   - Fallback                  : empty (safe but loses the optimization)
//
// MEMORY_NOINLINE
//   Prevents inlining of the marked function.  Useful for error-handling
//   paths that should not bloat hot-path code, or for functions that must
//   appear as distinct symbols in profiler output.
//   - GCC / Clang : __attribute__((noinline))
//   - Others      : empty
//
// MEMORY_COLD
//   Tells the compiler the function is called rarely (e.g. init/teardown,
//   error paths).  The optimizer moves its code away from hot-path code,
//   improving I-cache utilization and branch-prediction for the common case.
//   - GCC / Clang : __attribute__((cold))
//   - Others      : empty
//
// The SWIG guard ensures the attributes are hidden from the SWIG parser,
// which does not understand GCC extension syntax.
// ============================================================================
#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
// Compiler supports GCC-style attributes
#define MEMORY_NORETURN __attribute__((noreturn))
#define MEMORY_NOINLINE __attribute__((noinline))
#define MEMORY_COLD __attribute__((cold))
#elif defined(_MSC_VER)
// Non-GCC equivalents
#define MEMORY_NORETURN __declspec(noreturn)
#define MEMORY_NOINLINE
#define MEMORY_COLD
#else
// Non-GCC equivalents
#define MEMORY_NORETURN
#define MEMORY_NOINLINE
#define MEMORY_COLD
#endif

// ============================================================================
// Section 9: SIMD Kernel Return-Type Helper
//
// MEMORY_SIMD_RETURN_TYPE is a convenience macro for declaring SIMD
// processing functions that return void.
//
//   Release builds (NDEBUG defined):
//     Expands to:  MEMORY_FORCE_INLINE static void MEMORY_VECTORCALL
//     The function is inlined and uses the vectorcall ABI (MSVC) so that
//     SIMD arguments are passed in registers.
//
//   Debug builds:
//     Expands to:  static void
//     No forced inlining, making the function debuggable step-by-step.
//
// Usage:
//   MEMORY_SIMD_RETURN_TYPE AddPackets(const __m256& a, const __m256& b, __m256& out);
// ============================================================================
#ifdef NDEBUG
#define MEMORY_SIMD_RETURN_TYPE MEMORY_FORCE_INLINE static void MEMORY_VECTORCALL
#else
#define MEMORY_SIMD_RETURN_TYPE static void
#endif

// ============================================================================
// Section 10: CUDA / HIP Device-Function Attributes
//
// When compiling with nvcc (CUDA) or hipcc (HIP), functions that run on the
// GPU must be annotated so the device compiler generates GPU code for them.
//
// MEMORY_CUDA_DEVICE        : __device__  — callable only from GPU code
// MEMORY_CUDA_HOST          : __host__    — callable only from CPU code
// MEMORY_CUDA_FUNCTION_TYPE : __host__ __device__  — compiled for both;
//                               lets a single implementation be shared
//                               between CPU unit tests and GPU kernels.
//
// Outside CUDA/HIP all three expand to nothing, so the same source compiles
// as plain C++ without any changes.
//
// MEMORY_FUNCTION_ATTRIBUTE combines MEMORY_CUDA_FUNCTION_TYPE with
// MEMORY_FORCE_INLINE (release) or just MEMORY_CUDA_FUNCTION_TYPE
// (debug), giving the standard decoration for math utility functions that
// must work on both host and device.
// ============================================================================
#if defined(__CUDACC__) || defined(__HIPCC__)
#define MEMORY_CUDA_DEVICE __device__
#define MEMORY_CUDA_HOST __host__
#define MEMORY_CUDA_FUNCTION_TYPE __host__ __device__
#else
#define MEMORY_CUDA_DEVICE
#define MEMORY_CUDA_HOST
#define MEMORY_CUDA_FUNCTION_TYPE
#endif

#ifdef NDEBUG
#define MEMORY_FUNCTION_ATTRIBUTE MEMORY_FORCE_INLINE MEMORY_CUDA_FUNCTION_TYPE
#else
#define MEMORY_FUNCTION_ATTRIBUTE MEMORY_CUDA_FUNCTION_TYPE
#endif

// ============================================================================
// Section 11: Inline Assembly Comment
//
// MEMORY_ASM_COMMENT(X) injects a comment string directly into the
// compiler's assembly output.  This is purely a developer/profiling aid:
//
//   - It acts as a zero-cost "fence" that prevents the compiler from
//     reordering surrounding instructions across the comment.
//   - It makes it easy to locate a specific code region when reading
//     disassembly or perf annotate output.
//
// Only defined on GCC for x86-32, x86-64, ARM, and AArch64 — the only
// combinations that support the inline-asm syntax used here.
// On all other platforms it expands to nothing.
//
// Usage:
//   MEMORY_ASM_COMMENT("begin matrix multiply kernel");
// ============================================================================
#if defined(__GNUC__) && \
    (defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__))
#define MEMORY_ASM_COMMENT(X) __asm__("#" X)
#else
#define MEMORY_ASM_COMMENT(X)
#endif

// ============================================================================
// Section 12: Token-Pasting / Unique-Name Helpers
//
// MEMORY_CONCATENATE_IMPL(s1, s2)
//   The inner macro that performs the actual ## token paste.  It must be a
//   separate macro from MEMORY_CONCATENATE so that macro arguments are
//   fully expanded before pasting (standard C preprocessor two-step).
//
// MEMORY_CONCATENATE(s1, s2)
//   Public API — paste two already-expanded tokens into one identifier.
//   Example:  MEMORY_CONCATENATE(my_, var)  →  my_var
//
// MEMORY_UID
//   A unique integer token within a translation unit.
//   Prefers __COUNTER__ (GCC/Clang/MSVC extension; increments each time it
//   is used) over __LINE__ (same value for two macros on the same line).
//
// MEMORY_ANONYMOUS_VARIABLE(str)
//   Creates a unique identifier by appending MEMORY_UID to a given prefix.
//   Typical use: declaring RAII guard variables that must not clash:
//
//     auto MEMORY_ANONYMOUS_VARIABLE(guard) = makeScopedLock(mutex);
// ============================================================================
#define MEMORY_CONCATENATE_IMPL(s1, s2) s1##s2
#define MEMORY_CONCATENATE(s1, s2) MEMORY_CONCATENATE_IMPL(s1, s2)

#ifdef __COUNTER__
#define MEMORY_UID __COUNTER__
#define MEMORY_ANONYMOUS_VARIABLE(str) MEMORY_CONCATENATE(str, __COUNTER__)
#else
#define MEMORY_UID __LINE__
#define MEMORY_ANONYMOUS_VARIABLE(str) MEMORY_CONCATENATE(str, __LINE__)
#endif

// ============================================================================
// Section 13: C++20 Three-Way Comparison Detection
//
// MEMORY_HAS_THREE_WAY_COMPARISON evaluates to 1 when both the compiler
// and the standard library fully support the spaceship operator (<=>), and
// to 0 otherwise.
//
// Two feature-test macros must both be present and at the required level:
//   __cpp_impl_three_way_comparison  >= 201907L  (core-language support)
//   __cpp_lib_three_way_comparison   >= 201907L  (std::strong_ordering etc.)
//
// The outer #if guard prevents redefining the macro if calling code has
// already set it (e.g. via a build-system flag).
//
// Usage:
//   #if MEMORY_HAS_THREE_WAY_COMPARISON
//     auto operator<=>(const Foo&) const = default;
//   #endif
// ============================================================================
#if !MEMORY_HAS_THREE_WAY_COMPARISON
#if defined(__cpp_impl_three_way_comparison) && __cpp_impl_three_way_comparison >= 201907L && \
    defined(__cpp_lib_three_way_comparison) && __cpp_lib_three_way_comparison >= 201907L
#define MEMORY_HAS_THREE_WAY_COMPARISON 1
#else
#define MEMORY_HAS_THREE_WAY_COMPARISON 0
#endif
#endif

// ============================================================================
// Section 14: MemorySanitizer Suppression
//
// MEMORY_NO_SANITIZE_MEMORY suppresses MemorySanitizer (MSan) reports for
// the marked function.  MSan intercepts memory reads and flags use of
// uninitialised memory; some low-level or hardware-interfacing routines
// intentionally read "uninitialised" bytes (e.g. reading padding, working
// with memory-mapped I/O), producing false positives.
//
//   Clang with MSan enabled : __attribute__((no_sanitize_memory))
//   All other builds        : empty  (no runtime cost, no behavioural change)
//
// Usage:
//   MEMORY_NO_SANITIZE_MEMORY void ReadHardwareBuffer(void* buf, size_t n);
// ============================================================================
#if defined(__clang__)
#if __has_feature(memory_sanitizer)
#define MEMORY_NO_SANITIZE_MEMORY __attribute__((no_sanitize_memory))
#else
#define MEMORY_NO_SANITIZE_MEMORY
#endif  // __has_feature(memory_sanitizer)
#else
#define MEMORY_NO_SANITIZE_MEMORY
#endif  // __clang__

// ============================================================================
// Section 15: Type Utilities
//
// MACRO_CORE_TYPE_ID_NAME(x)
//   Returns the implementation-defined mangled name of type x via typeid.
//   Useful for debug logging and diagnostic messages.
//   Note: the returned string is compiler-specific (e.g. "i" for int on GCC).
//   Use __cxa_demangle (see MEMORY_HAS_CXA_DEMANGLE below) for a
//   human-readable name.
//   Example: MACRO_CORE_TYPE_ID_NAME(float)  →  "f"  (GCC/Clang)
//
// MEMORY_DELETE_CLASS(type)
//   Deletes all six special member functions of a class:
//     default constructor, copy constructor, copy assignment,
//     move constructor, move assignment, destructor.
//   Use for classes that should never be instantiated at all (pure
//   namespace-like utility classes, tag types, etc.).
//
// MEMORY_DELETE_COPY_AND_MOVE(type)
//   Deletes the four copy/move special members while leaving constructor and
//   destructor intact.  Switches to private access before the deletions and
//   back to public afterwards, enforcing the common pattern where
//   non-copyable/non-movable types keep their access specifiers tidy.
//   Use for singleton-like resources (device handles, loggers, allocators).
//
// MEMORY_DELETE_COPY(type)
//   Deletes only the copy constructor and copy assignment operator, leaving
//   move semantics available.  Use for move-only resource wrappers.
// ============================================================================
#define MACRO_CORE_TYPE_ID_NAME(x) typeid(x).name()

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
// Section 16: Printf-Format Attribute
//
// MACRO_CORE_PRINTF_FORMAT(a, b) applies GCC's format(printf, a, b)
// attribute to a variadic function, enabling the compiler to type-check
// the format string against the variadic arguments at compile time —
// the same checking applied to printf itself.
//
//   a : 1-based index of the format-string argument
//   b : 1-based index of the first variadic argument to check
//
// The outer #if guard means calling code can define the macro to empty
// (disabling the check) without a redefinition warning.
//
// Example:
//   void Log(const char* fmt, ...) MACRO_CORE_PRINTF_FORMAT(1, 2);
// ============================================================================
#if !defined(MACRO_CORE_PRINTF_FORMAT)
#if defined(__GNUC__)
#define MACRO_CORE_PRINTF_FORMAT(a, b) __attribute__((format(printf, a, b)))
#else
#define MACRO_CORE_PRINTF_FORMAT(a, b)
#endif
#endif

// ============================================================================
// Section 17: void_t Utility (C++17 Detection Idiom)
//
// memory::void_t<Ts...> maps any list of well-formed types to void.
// It is the standard building block for SFINAE-based type-trait detection.
//
// Although std::void_t is available in C++17, the alias is kept here so
// that internal templates can use the memory:: namespace consistently
// without needing a std:: qualifier.
//
// Example — detect whether T has a ::value_type member:
//   template <typename T, typename = void>
//   struct has_value_type : std::false_type {};
//
//   template <typename T>
//   struct has_value_type<T, memory::void_t<typename T::value_type>>
//       : std::true_type {};
// ============================================================================
namespace memory
{
template <typename...>
using void_t = std::void_t<>;
}  // namespace memory

// ============================================================================
// Section 18: Portable Integer Type Aliases
//
// memory_int  : a signed 64-bit integer
// memory_long : an unsigned 64-bit integer
//
// On MSVC (excluding Intel ICC which mimics GCC) the standard typedefs
// for 64-bit integers use the Microsoft-specific __int64 / unsigned __int64
// keywords.  On all other compilers (GCC, Clang, ICC) the standard C/C++
// long long / unsigned long long spellings are used.
//
// These aliases avoid scattering #ifdef _MSC_VER throughout arithmetic code.
// ============================================================================
#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define memory_int __int64
#define memory_long unsigned __int64
#else
#define memory_int long long int
#define memory_long unsigned long long int
#endif

// ============================================================================
// Section 19: Compiler Attribute Detection Helpers
//
// MEMORY_HAVE_CPP_ATTRIBUTE(x)
//   Wraps __has_cpp_attribute(x) — the standard C++20 way to test for a
//   specific [[attribute]].  Falls back to 0 on compilers that do not
//   support __has_cpp_attribute.
//
// MEMORY_HAVE_ATTRIBUTE(x)
//   Wraps __has_attribute(x) — the GCC/Clang extension for testing
//   __attribute__((x)) support.  Falls back to 0 on other compilers.
//
// Both macros are used internally (see sections below) to guard attribute
// definitions so that unsupported attributes silently expand to nothing
// rather than causing a compiler error.
//
// Example:
//   #if MEMORY_HAVE_ATTRIBUTE(noinline)
//   #define MY_NOINLINE __attribute__((noinline))
//   #endif
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
// Section 20: Branch Prediction Hints
//
// MEMORY_LIKELY(expr)   — expr is expected to be true most of the time
// MEMORY_UNLIKELY(expr) — expr is expected to be false most of the time
//
// Wrapping a condition with these macros lets the compiler lay out the hot
// path as fall-through code and push the cold path into a separate basic
// block, improving instruction-cache utilisation and branch-predictor
// training.
//
// Three tiers of implementation:
//   C++20 [[likely]] / [[unlikely]] attributes (standard, no cast needed):
//     if (MEMORY_LIKELY(x > 0)) { ... }
//     expands to: if ((x > 0) [[likely]]) { ... }
//
//   GCC / Clang / ICC __builtin_expect (most common path):
//     Casts expr to bool before passing to __builtin_expect so that the
//     macro works correctly for non-integer expressions.
//
//   Fallback (MSVC without C++20, unknown compilers):
//     No-op — the condition is passed through unchanged.
//
// Usage:
//   if (MEMORY_LIKELY(ptr != nullptr)) { fast_path(); }
//   if (MEMORY_UNLIKELY(error))        { handle_error(); }
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
// Section 21: Conditional constexpr for C++20
//
// MEMORY_FUNCTION_CONSTEXPR marks a function as constexpr only when
// compiling in C++20 or later mode.
//
// Motivation: some functions cannot be constexpr in C++17 (e.g. they call
// functions that became constexpr only in C++20) but should be when the
// compiler supports it.  This macro lets the same source compile cleanly
// under both standard versions without #ifdef clutter at every declaration.
//
// Usage:
//   MEMORY_FUNCTION_CONSTEXPR int Clamp(int v, int lo, int hi);
// ============================================================================
#if __cplusplus >= 202002L
#define MEMORY_FUNCTION_CONSTEXPR constexpr
#else
#define MEMORY_FUNCTION_CONSTEXPR
#endif

// ============================================================================
// Section 22: [[nodiscard]] Portability
//
// MEMORY_NODISCARD applies the [[nodiscard]] attribute on C++17 and later,
// causing the compiler to emit a warning when the return value of a marked
// function is silently discarded.
//
// Typical use cases: factory functions, error codes, memory allocations,
// and any function whose return value is the only channel for communicating
// success or failure.
//
// On pre-C++17 compilers the macro expands to nothing (silent no-op).
//
// Usage:
//   MEMORY_NODISCARD Status Allocate(size_t bytes, void** out);
// ============================================================================
#if __cplusplus >= 201703L
#define MEMORY_NODISCARD [[nodiscard]]
#else
#define MEMORY_NODISCARD
#endif

// ============================================================================
// Section 23: Suppress Unused-Variable / Unused-Parameter Warnings
//
// MEMORY_UNUSED suppresses compiler warnings about variables or function
// parameters that are intentionally left unreferenced (e.g. in debug-only
// code guarded by #ifdef, or in virtual overrides that ignore some params).
//
//   C++17 and later : [[maybe_unused]]  (standard attribute)
//   GCC / Clang     : __attribute__((unused))
//   MSVC            : __pragma(warning(suppress : 4100))
//   Fallback        : empty
//
// MEMORY_USED is the complementary macro: it forces the linker to retain
// a symbol even if it appears unreferenced.
//   GCC / Clang with __has_attribute(used) : __attribute__((__used__))
//   MSVC / others                          : empty
//   Typical use: static arrays or callback tables that are referenced only
//   from assembly or loaded at runtime.
//
// Usage:
//   void Foo(int x, MEMORY_UNUSED int debug_hint) { ... }
//   static MEMORY_USED CallbackTable kTable[] = { ... };
// ============================================================================
#if __cplusplus >= 201703L
#define MEMORY_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__)
// For GCC or Clang: use __attribute__
#define MEMORY_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
// For MSVC
#define MEMORY_UNUSED __pragma(warning(suppress : 4100))
#else
// Fallback for other compilers
#define MEMORY_UNUSED
#endif

// Check for MSVC first, then Clang/GCC
#if defined(_MSC_VER)
// MSVC doesn't support __attribute__((used))
// Use __pragma(comment(linker, "/include:symbol")) or just leave empty
#define MEMORY_USED
#elif defined(__has_attribute)
#if __has_attribute(used)
#define MEMORY_USED __attribute__((__used__))
#else
#define MEMORY_USED
#endif
#else
#define MEMORY_USED
#endif

// ============================================================================
// Section 24: C++ Symbol Demangling Availability
//
// MEMORY_HAS_CXA_DEMANGLE is 1 when the platform provides
// __cxa_demangle() (from <cxxabi.h>), which converts a compiler-mangled
// C++ symbol name (e.g. "_ZN9memory3FooEv") into a human-readable form
// ("memory::Foo()").  Used by stack-trace and diagnostic utilities.
//
// Rules:
//   0 — Android on x86/x86-64 (known to be missing)
//   1 — GCC >= 3.4 on non-MIPS targets
//   1 — Clang on non-MSVC platforms
//   0 — everything else (MSVC, unknown compilers, MIPS)
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
// Section 25: Printf-Format Attribute (MEMORY naming convention)
//
// MEMORY_PRINTF_ATTRIBUTE(format_index, first_to_check) is the
// MEMORY-prefixed counterpart to MACRO_CORE_PRINTF_FORMAT (Section 16).
// It applies __attribute__((format(printf, ...))) only when the compiler
// actually supports it (checked via MEMORY_HAVE_ATTRIBUTE from Section 19),
// making it safer than an unconditional __attribute__ use.
//
//   format_index    : 1-based position of the format string argument
//   first_to_check  : 1-based position of the first variadic argument
//
// Example:
//   void Warn(int level, const char* fmt, ...)
//       MEMORY_PRINTF_ATTRIBUTE(2, 3);
// ============================================================================
#if MEMORY_HAVE_ATTRIBUTE(format)
#define MEMORY_PRINTF_ATTRIBUTE(format_index, first_to_check) \
    __attribute__((format(printf, format_index, first_to_check)))
#else
#define MEMORY_PRINTF_ATTRIBUTE(format_index, first_to_check)
#endif

// ============================================================================
// Section 26: Thread-Safety Analysis Annotations
//
// These macros integrate with Clang's Thread Safety Analysis (TSA) — a
// static analysis pass that verifies locks are held when accessing guarded
// data.  When TSA is not available (GCC, MSVC, older Clang) all macros
// expand to nothing, so the annotations are zero-cost at runtime.
//
// MEMORY_NO_THREAD_SAFETY_ANALYSIS
//   Opt a function out of TSA entirely.  Use sparingly for lock-free
//   primitives or functions where TSA produces false positives.
//
// MEMORY_GUARDED_BY(x)
//   Declares that the annotated variable must only be accessed while
//   holding mutex x.
//   Example:  int count_ MEMORY_GUARDED_BY(mutex_);
//
// MEMORY_EXCLUSIVE_LOCKS_REQUIRED(...)
//   Documents that the caller must hold the listed mutexes exclusively
//   (write locks) before calling the annotated function.
//
// MEMORY_LOCKS_EXCLUDED(...)
//   Documents that the caller must NOT hold the listed mutexes when calling
//   the annotated function (prevents self-deadlock for non-reentrant locks).
//
// Usage:
//   void Increment() MEMORY_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
//   void Reset()     MEMORY_LOCKS_EXCLUDED(mutex_);
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

#if MEMORY_HAVE_ATTRIBUTE(locks_excluded)
#define MEMORY_LOCKS_EXCLUDED(...) __attribute__((locks_excluded(__VA_ARGS__)))
#else
#define MEMORY_LOCKS_EXCLUDED(...)
#endif

// ============================================================================
// Section 27: Lifetime-Bound Annotation
//
// MEMORY_LIFETIME_BOUND marks a function parameter (or implicit *this) to
// indicate that the returned reference / pointer must not outlive the
// argument.  This lets the compiler warn when the caller stores the result
// in a variable that will outlast the temporary argument.
//
// Priority order (most specific first):
//   [[clang::lifetimebound]]  — Clang attribute (C++11 extension)
//   [[msvc::lifetimebound]]   — MSVC attribute  (C++11 extension)
//   __attribute__((lifetimebound)) — raw attribute form
//   empty                     — all other compilers
//
// Example:
//   std::string_view AsView(const std::string& s MEMORY_LIFETIME_BOUND);
//   // Warn if caller does: auto sv = AsView(std::string("tmp"));
// ============================================================================
#if MEMORY_HAVE_CPP_ATTRIBUTE(clang::lifetimebound)
#define MEMORY_LIFETIME_BOUND [[clang::lifetimebound]]
#elif MEMORY_HAVE_CPP_ATTRIBUTE(msvc::lifetimebound)
#define MEMORY_LIFETIME_BOUND [[msvc::lifetimebound]]
#elif MEMORY_HAVE_ATTRIBUTE(lifetimebound)
#define MEMORY_LIFETIME_BOUND __attribute__((lifetimebound))
#else
#define MEMORY_LIFETIME_BOUND
#endif

// ============================================================================
// Section 28: Constant-Initialisation Annotation
//
// MEMORY_CONST_INIT applies [[clang::require_constant_initialization]]
// to a variable, ensuring it is initialised at compile time (i.e. its
// initialiser is a constant expression).  This prevents the "static
// initialisation order fiasco" for global / static-local variables and
// guarantees the variable is ready before any dynamic initialisation runs.
//
// Only Clang supports this attribute; on all other compilers the macro
// expands to nothing (the variable is still initialised, just without the
// compile-time guarantee).
//
// Usage:
//   MEMORY_CONST_INIT static Config kDefaultConfig = { ... };
// ============================================================================
#if MEMORY_HAVE_CPP_ATTRIBUTE(clang::require_constant_initialization)
#define MEMORY_CONST_INIT [[clang::require_constant_initialization]]
#else
#define MEMORY_CONST_INIT
#endif

// ============================================================================
// Section 29: Darwin (Apple) Missing Type Definitions
//
// Some versions of Homebrew Clang on macOS do not ship the Darwin SDK
// typedef aliases (__uint32_t, __int64_t, etc.) that Apple's own Clang
// provides.  Third-party headers (e.g. certain BLAS or LAPACK wrappers)
// assume these types exist.
//
// When building on Apple platforms without these definitions, this block
// injects them as using aliases to the corresponding <cstdint> types.
// The __DEFINED_DARWIN_TYPES guard prevents double-definition if the SDK
// header that normally defines them is also included.
// ============================================================================
#if defined(__APPLE__) && !defined(__DEFINED_DARWIN_TYPES)
#define __DEFINED_DARWIN_TYPES
using __uint32_t = std::uint32_t;
using __uint64_t = std::uint64_t;
using __int32_t  = std::int32_t;
using __int64_t  = std::int64_t;
using __uint8_t  = std::uint8_t;
using __uint16_t = std::uint16_t;
using __int8_t   = std::int8_t;
using __int16_t  = std::int16_t;
#endif

#endif  // MEMORY_PORTABLE_MACROS_INCLUDED_
