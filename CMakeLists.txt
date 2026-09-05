# =============================================================================
# Library/Memory —
# CMakeLists.txt
#
# CPU/GPU allocators, mimalloc integration, CUDA/HIP device memory, NUMA-aware allocation, and
# per-allocator statistics.
#
# Public compile definitions: MEMORY_HAS_TBB=1/0       — TBB scalable allocator active
# MEMORY_HAS_CUDA=1/0      — CUDA device-memory support compiled in MEMORY_HAS_HIP=1/0       —
# HIP/ROCm device-memory support compiled in MEMORY_HAS_MIMALLOC=1/0  — mimalloc high-perf allocator
# active MEMORY_HAS_NUMA=1/0      — NUMA-aware allocation available MEMORY_HAS_MEMKIND=1/0   —
# memkind extended-memory library available. Logging is a required sibling
# module (always links Logging::Logging).
# =============================================================================
cmake_minimum_required(VERSION 3.16)

# --- Section 1: Feature flags (MEMORY_ENABLE_*) ---

# Link-Time Optimization — disabled when MEMORY_ENABLE_COVERAGE or MEMORY_ENABLE_SANITIZER is ON
# (LTO conflicts with profiling instrumentation and typical sanitizer workflows).
include(lto)
xsigma_lto_compute_default(_lto_default PROFILE PERFORMANCE)
set(MEMORY_LTO_MODE "${_lto_default}" CACHE STRING
                                            "LTO mode for Memory: off | thin | full | ipo | auto"
)
set_property(CACHE MEMORY_LTO_MODE PROPERTY STRINGS off thin full ipo auto)

# Coverage instrumentation — activates -fprofile-instr-generate / --coverage compiler flags for
# this module only. LTO is automatically skipped when coverage is enabled.
option(MEMORY_ENABLE_COVERAGE "Enable coverage instrumentation for Memory" OFF)

# Testing — gates add_subdirectory(Testing/Cxx).
option(MEMORY_ENABLE_TESTING "Build Memory test suite" ON)

# Examples — gates add_subdirectory(Examples) when Memory examples exist.
option(MEMORY_ENABLE_EXAMPLES "Build Memory example programs" OFF)

# Google Test integration for Memory tests.
option(MEMORY_ENABLE_GTEST "Enable GoogleTest for Memory" ON)

# When ON and target Profiler::Profiler exists: link Profiler and set MEMORY_HAS_PROFILER=1, so
# the CPU allocator and GPU caching allocators can report allocation/deallocation/OOM events via
# profiler::report_memory_usage() / report_out_of_memory() (Docs/profiler/profiler.md,
# Instrumentation). CPU allocate/free compile those calls out when this is 0; when it is 1 they
# run only after profiler::memory_profiling_active() (predicted-false). Safe to leave ON.
option(MEMORY_ENABLE_PROFILER "Link Memory with the Profiler library when available" ON)

# Microsoft mimalloc — high-performance drop-in malloc replacement. ON by default to match the CMake
# project default.
option(MEMORY_ENABLE_MIMALLOC "Enable Microsoft mimalloc high-performance memory allocator" ON)
mark_as_advanced(MEMORY_ENABLE_MIMALLOC)

# mimalloc statistics — OFF by default: mimalloc release builds compile stats out (MI_STAT=0).
# ON compiles the vendored mimalloc with MI_STAT=1 so MIMALLOC_SHOW_STATS / mi_stats_print() and
# the cpu::memory_allocator stats helpers report real counters.
option(MEMORY_ENABLE_MIMALLOC_STATS "Compile mimalloc statistics support (MI_STAT=1)" OFF)
mark_as_advanced(MEMORY_ENABLE_MIMALLOC_STATS)

# memkind — Linux-only extended-memory allocator (pmem, HBM, etc.). Forced OFF on non-Linux
# platforms via the platform-constraint block below.
option(MEMORY_ENABLE_MEMKIND "Enable memkind extended-memory allocator (Linux only)" OFF)

# Google Benchmark integration for micro-benchmarking allocator paths.
option(MEMORY_ENABLE_BENCHMARK "Enable Google Benchmark for Memory" ON)

# Icecream distributed compilation launcher (rarely needed; hidden by default).
option(MEMORY_ENABLE_ICECC "Use Icecream (icecc) distributed compiler for Memory" OFF)
mark_as_advanced(MEMORY_ENABLE_ICECC)

# Per-target compiler cache — wraps every compile command with ccache / sccache etc. to speed up
# incremental and CI rebuilds. ON by default.
option(MEMORY_ENABLE_CACHE "Enable compiler caching for the Memory target" ON)
mark_as_advanced(MEMORY_ENABLE_CACHE)

option(MEMORY_ENABLE_CLANGTIDY "Enable clang-tidy static analysis for Memory" OFF)
mark_as_advanced(MEMORY_ENABLE_CLANGTIDY)

option(MEMORY_ENABLE_FIX "Enable clang-tidy fix-errors and fix options for Memory" OFF)
mark_as_advanced(MEMORY_ENABLE_FIX)

option(MEMORY_ENABLE_IWYU "Enable include-what-you-use analysis for Memory" OFF)
mark_as_advanced(MEMORY_ENABLE_IWYU)

option(MEMORY_ENABLE_SANITIZER "Build Memory with sanitizer support (Clang/GCC only)" OFF)
set(MEMORY_SANITIZER_TYPE "address"
    CACHE STRING "Sanitizer type for Memory. Options: address, undefined, thread, memory, leak"
)

set_property(CACHE MEMORY_SANITIZER_TYPE PROPERTY STRINGS address undefined thread memory leak)
mark_as_advanced(MEMORY_ENABLE_SANITIZER MEMORY_SANITIZER_TYPE)

option(MEMORY_ENABLE_SPELL "Enable check-only spell checking for Memory" OFF)
mark_as_advanced(MEMORY_ENABLE_SPELL)

option(MEMORY_ENABLE_VALGRIND "Execute Memory test suite with Valgrind" OFF)
mark_as_advanced(MEMORY_ENABLE_VALGRIND)
set(MEMORY_LINKER_CHOICE "default" CACHE STRING
                                         "Linker for Memory: default, lld, mold, gold, lld-link"
)

set_property(CACHE MEMORY_LINKER_CHOICE PROPERTY STRINGS default lld mold gold lld-link)
mark_as_advanced(MEMORY_LINKER_CHOICE)
set(MEMORY_CACHE_BACKEND "none"
    CACHE STRING "Compiler cache backend for Memory. Options: none, ccache, sccache, buildcache"
)

set_property(CACHE MEMORY_CACHE_BACKEND PROPERTY STRINGS none ccache sccache buildcache)
mark_as_advanced(MEMORY_CACHE_BACKEND)

# GPU backend selector — drives CUDA/HIP toolchain detection and compile definitions.
# Can be set from the command line: -DMEMORY_GPU_BACKEND=cuda|hip|none
set(MEMORY_GPU_BACKEND "none" CACHE STRING "GPU backend for Memory: none, cuda, hip, metal")
set_property(CACHE MEMORY_GPU_BACKEND PROPERTY STRINGS none cuda hip metal)
if(NOT MEMORY_GPU_BACKEND MATCHES "^(none|cuda|hip|metal)$")
  message(
    FATAL_ERROR
      "MEMORY_GPU_BACKEND must be one of: none, cuda, hip, metal (got '${MEMORY_GPU_BACKEND}')"
  )
endif()
if(MEMORY_GPU_BACKEND STREQUAL "metal" AND NOT APPLE)
  message(FATAL_ERROR "MEMORY_GPU_BACKEND=metal is only supported on Apple platforms.")
endif()

# C++ standard — applied to every TU in this module via CMAKE_CXX_STANDARD.
set(MEMORY_CXX_STANDARD "20" CACHE STRING "C++ standard for Memory. Options: 11 14 17 20 23")
set_property(CACHE MEMORY_CXX_STANDARD PROPERTY STRINGS 11 14 17 20 23)
if(NOT MEMORY_CXX_STANDARD MATCHES "^(11|14|17|20|23)$")
  message(FATAL_ERROR "MEMORY_CXX_STANDARD must be one of: 11 14 17 20 23")
endif()

# Platform constraints — NUMA and memkind are Linux-only kernel/library interfaces. Silently force
# them OFF on Windows and macOS so downstream callers do not need to guard every reference to
# MEMORY_ENABLE_NUMA / MEMORY_ENABLE_MEMKIND.
if(NOT UNIX)
  set(MEMORY_ENABLE_NUMA OFF CACHE BOOL "Enable NUMA node allocation (Linux only)" FORCE)
endif()
if(NOT UNIX OR APPLE)
  set(MEMORY_ENABLE_MEMKIND OFF CACHE BOOL "Enable memkind allocator (Linux only)" FORCE)
endif()

# --- Section 3: CMake module includes --- Order: introspection → toolchain helpers → Memory-owned
# finders → coverage/sanitizer → LTO/linker.

# a) Standard CMake introspection
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)
include(CheckSymbolExists)

# b) Shared toolchain helpers
include(compiler_checks) # Detects compiler capabilities; sets PROJECT_* flags
include(checks) # Additional feature-check macros
include(build_type) # Validates / normalises CMAKE_BUILD_TYPE
include(platform) # Platform flags (macOS rpath, Windows /utf-8, …)
include(CMakeDependentOption)

include(cache) # Configures per-target compiler cache launcher
include(memory_utils) # Memory-specific compiler checks (NUMA availability)

# c) Memory-owned feature finders (Library/Memory/Cmake/)
if(MEMORY_GPU_BACKEND STREQUAL "cuda")
  include(cuda) # Detects/configures CUDA
elseif(MEMORY_GPU_BACKEND STREQUAL "hip")
  include(hip) # Detects/configures HIP/ROCm
elseif(MEMORY_GPU_BACKEND STREQUAL "metal")
  include(metal) # Detects/configures Apple Metal
endif()
include(tbb_memory) # Finds/builds TBB scalable allocator when MEMORY_ENABLE_TBB=ON
include(numa) # Finds NUMA library when MEMORY_ENABLE_NUMA=ON
if(MEMORY_ENABLE_SANITIZER)
  set(MEMORY_ENABLE_MIMALLOC OFF)
  include(sanitize)
endif()

# Valgrind and mimalloc do not play nicely together: mimalloc can reserve very large
# virtual address ranges which triggers Valgrind internal errors/crashes.
if(MEMORY_ENABLE_VALGRIND)
  set(MEMORY_ENABLE_MIMALLOC OFF CACHE BOOL "Disable mimalloc when running under Valgrind" FORCE)
endif()

# Statistics only make sense when mimalloc itself is compiled in.
if(NOT MEMORY_ENABLE_MIMALLOC AND MEMORY_ENABLE_MIMALLOC_STATS)
  message(WARNING "MEMORY_ENABLE_MIMALLOC_STATS requires MEMORY_ENABLE_MIMALLOC — disabling")
  set(MEMORY_ENABLE_MIMALLOC_STATS OFF CACHE BOOL "Requires MEMORY_ENABLE_MIMALLOC" FORCE)
endif()
if(MEMORY_ENABLE_SPELL)
  include(spell)
endif()
if(MEMORY_ENABLE_VALGRIND)
  include(valgrind)
endif()
include(helper_macros)

# d) Coverage gate — appends -fprofile-instr-generate / --coverage; LTO already cleared above.
if(MEMORY_ENABLE_COVERAGE)
  include(coverage)
elseif(MSVC)
  string(APPEND CMAKE_EXE_LINKER_FLAGS " /DEBUG:FULL")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " /DEBUG:FULL")
endif()

# --- Third-party dependencies owned by Memory ---
set(_tpdir "${CMAKE_SOURCE_DIR}/ThirdParty")

set(BUILD_TESTING OFF CACHE BOOL "Disable testing for third-party libraries" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable examples for third-party libraries" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable benchmarks for third-party libraries" FORCE)

# fmt — Logging is configured first and registers Fmt::fmt; this block is a
# fallback if Logging is absent.
if(NOT TARGET Fmt::fmt)
  set(FMT_TEST OFF CACHE BOOL "Disable fmt tests" FORCE)
  set(FMT_DOC OFF CACHE BOOL "Disable fmt documentation" FORCE)
  set(FMT_INSTALL OFF CACHE BOOL "Disable fmt install" FORCE)
  set(FMT_HEADER_ONLY ON CACHE BOOL "Use header-only fmt library" FORCE)

  add_third_party_library(
    fmt
    LIBRARY_TYPE
    STATIC
    SOURCE_DIR
    "${_tpdir}/fmt"
    ALIASES
    "fmt::fmt-header-only"
    XSIGMA_TARGETS
    "Fmt::fmt=fmt::fmt-header-only|fmt::fmt|fmt"
    INCLUDE_DIR
    "${_tpdir}/fmt/include"
    AUTO_ADD_TO_DEPENDENCIES
    OFF
  )
endif()

if(MEMORY_ENABLE_MIMALLOC)
  # mimalloc-specific options
  set(MI_BUILD_TESTS OFF CACHE BOOL "Disable mimalloc tests" FORCE)
  set(MI_BUILD_OBJECT OFF CACHE BOOL "Disable mimalloc object library" FORCE)
  set(MI_INSTALL_TOPLEVEL OFF CACHE BOOL "Disable mimalloc top-level install" FORCE)
  set(MI_OVERRIDE OFF CACHE BOOL "Disable mimalloc malloc override" FORCE)
  set(MI_BUILD_SHARED OFF CACHE BOOL "Build mimalloc as static library" FORCE)
  set(MI_BUILD_STATIC ON CACHE BOOL "Build mimalloc as static library" FORCE)

  add_third_party_library(
    mimalloc
    CONDITION
    MEMORY_ENABLE_MIMALLOC
    LIBRARY_TYPE
    STATIC
    SOURCE_DIR
    "${_tpdir}/mimalloc"
    LIBRARY_TYPE_VAR
    MI_BUILD_STATIC
    # Do not list -Wno-strict-prototypes here: FILTER_FLAGS removes flags from CMAKE_*_FLAGS, which
    # would drop the toolchain's prototype warning suppression for mimalloc C sources.
    FILTER_FLAGS
    "-flto=thin"
    "-flto=full"
    XSIGMA_TARGETS
    "Mimalloc::mimalloc=mimalloc|mimalloc-static"
    INCLUDE_DIR
    "${_tpdir}/mimalloc/include"
    AUTO_ADD_TO_DEPENDENCIES
    OFF
  )

  list(APPEND MEMORY_DEPENDENCY_INCLUDE_DIRS "${_tpdir}/mimalloc/include")

  # mimalloc release builds compile statistics out (MI_STAT=0); re-enable them on the
  # vendored target only — no ThirdParty source edits.
  if(MEMORY_ENABLE_MIMALLOC_STATS AND TARGET mimalloc-static)
    target_compile_definitions(mimalloc-static PRIVATE MI_STAT=1)
  endif()

  # MinGW: mimalloc links Windows system libs as bare names; patch to -l<name>.
  if(WIN32 AND (MINGW OR CMAKE_CXX_COMPILER MATCHES "msys64") AND TARGET mimalloc-static)
    set(_mi_win_syslibs psapi shell32 user32 advapi32 bcrypt)
    get_target_property(_mi_ll mimalloc-static LINK_LIBRARIES)
    if(_mi_ll)
      set(_changed FALSE)
      foreach(_lib IN LISTS _mi_win_syslibs)
        if("${_lib}" IN_LIST _mi_ll)
          list(REMOVE_ITEM _mi_ll "${_lib}")
          list(APPEND _mi_ll "-l${_lib}")
          set(_changed TRUE)
        endif()
      endforeach()
      if(_changed)
        set_property(TARGET mimalloc-static PROPERTY LINK_LIBRARIES "${_mi_ll}")
        message(STATUS "mimalloc: patched LINK_LIBRARIES for MinGW (-l prefix)")
      endif()
    endif()
    get_target_property(_mi_il mimalloc-static INTERFACE_LINK_LIBRARIES)
    if(_mi_il)
      set(_changed FALSE)
      foreach(_lib IN LISTS _mi_win_syslibs)
        if("${_lib}" IN_LIST _mi_il)
          list(REMOVE_ITEM _mi_il "${_lib}")
          list(APPEND _mi_il "-l${_lib}")
          set(_changed TRUE)
        endif()
        set(_ge "$<LINK_ONLY:${_lib}>")
        set(_ge2 "$<LINK_ONLY:-l${_lib}>")
        if("${_ge}" IN_LIST _mi_il)
          list(REMOVE_ITEM _mi_il "${_ge}")
          list(APPEND _mi_il "${_ge2}")
          set(_changed TRUE)
        endif()
      endforeach()
      if(_changed)
        set_property(TARGET mimalloc-static PROPERTY INTERFACE_LINK_LIBRARIES "${_mi_il}")
        message(STATUS "mimalloc: patched INTERFACE_LINK_LIBRARIES for MinGW (-l prefix)")
      endif()
    endif()
  endif()

endif()
# --- Section 2: MEMORY_ENABLE_* → MEMORY_HAS_* compile definitions --- TBB/CUDA/HIP/NUMA flags come
# from Section 3 finders; mapped after those includes.
include(compile_definitions)
# Per-allocator call-site tracking (opt-in; adds overhead).
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_MIMALLOC)
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_MIMALLOC_STATS)
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_MEMKIND)
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_NUMA)
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_TBB)
if(MEMORY_GPU_BACKEND STREQUAL "cuda")
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_CUDA=1 MEMORY_HAS_HIP=0 MEMORY_HAS_METAL=0)
elseif(MEMORY_GPU_BACKEND STREQUAL "hip")
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_CUDA=0 MEMORY_HAS_HIP=1 MEMORY_HAS_METAL=0)
elseif(MEMORY_GPU_BACKEND STREQUAL "metal")
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_CUDA=0 MEMORY_HAS_HIP=0 MEMORY_HAS_METAL=1)
else()
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_CUDA=0 MEMORY_HAS_HIP=0 MEMORY_HAS_METAL=0)
endif()
list(APPEND MEMORY_DEPENDENCY_COMPILE_DEFINITIONS ${MEMORY_COMPILE_DEFINITIONS})

# Required Logging sibling — Memory uses Library/Logging's LOGGING_CHECK/LOGGING_THROW/
# LOGGING_LOG_* macros directly (util/exception.h, logger/logger.h).
if(NOT TARGET Logging::Logging)
  message(
    FATAL_ERROR
      "Memory requires Logging::Logging. Configure Logging in this tree (full build or --project.memory, which now enables Logging)."
  )
endif()
list(APPEND MEMORY_DEPENDENCY_LIBS Logging::Logging)

# Optional Profiler sibling module — only wire link when MEMORY_ENABLE_PROFILER and target exists.
set(MEMORY_HAS_PROFILER_BUILD 0)
if(MEMORY_ENABLE_PROFILER AND TARGET Profiler::Profiler)
  set(MEMORY_HAS_PROFILER_BUILD 1)
  list(APPEND MEMORY_DEPENDENCY_LIBS Profiler::Profiler)
endif()
list(APPEND MEMORY_DEPENDENCY_COMPILE_DEFINITIONS MEMORY_HAS_PROFILER=${MEMORY_HAS_PROFILER_BUILD})

# --- Runtime library wiring ---
if(TARGET Fmt::fmt)
  list(APPEND MEMORY_DEPENDENCY_LIBS Fmt::fmt)
endif()
if(MEMORY_GPU_BACKEND STREQUAL "cuda" AND DEFINED PROJECT_CUDA_LIBRARIES)
  list(APPEND MEMORY_DEPENDENCY_LIBS ${PROJECT_CUDA_LIBRARIES})
endif()
if(MEMORY_GPU_BACKEND STREQUAL "hip" AND DEFINED PROJECT_HIP_LIBRARIES)
  list(APPEND MEMORY_DEPENDENCY_LIBS ${PROJECT_HIP_LIBRARIES})
endif()
if(MEMORY_ENABLE_TBB AND TARGET Tbb::tbbmalloc)
  list(APPEND MEMORY_DEPENDENCY_LIBS Tbb::tbbmalloc)
endif()
if(MEMORY_ENABLE_NUMA AND TARGET Numa::numa)
  list(APPEND MEMORY_DEPENDENCY_LIBS Numa::numa)
endif()
if(MEMORY_ENABLE_MIMALLOC AND TARGET Mimalloc::mimalloc)
  list(APPEND MEMORY_DEPENDENCY_LIBS Mimalloc::mimalloc)
endif()

# --- Source discovery ---
set(memory_headers "")
set(memory_sources "")
set(CMAKE_CXX_STANDARD ${MEMORY_CXX_STANDARD})

file(GLOB_RECURSE memory_headers LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
                                                        "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx"
)
file(GLOB_RECURSE memory_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Objective-C++ sources (Metal backend only) are globbed separately since they require
# OBJCXX language support, only enabled when MEMORY_GPU_BACKEND=metal (see below).
file(GLOB_RECURSE memory_objcxx_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/*.mm")

# Keep test sources out of the Memory library target.
list(FILTER memory_sources EXCLUDE REGEX ".*/Testing/.*")
list(FILTER memory_headers EXCLUDE REGEX ".*/Testing/.*")
list(FILTER memory_objcxx_sources EXCLUDE REGEX ".*/Testing/.*")

# GPU sources are only compiled when a GPU backend is active.
if(MEMORY_GPU_BACKEND STREQUAL "none")
  file(GLOB_RECURSE GPU_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cpp")
  file(GLOB_RECURSE GPU_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.h"
       "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.hxx"
  )
  list(REMOVE_ITEM memory_headers ${GPU_HEADER_FILES})
  list(REMOVE_ITEM memory_sources ${GPU_SOURCE_FILES})
endif()

# .mm sources only compile (and only link Metal/Foundation) when the Metal backend is active;
# on every other backend they're excluded, matching the gpu/*.cpp none-backend exclusion above.
if(MEMORY_GPU_BACKEND STREQUAL "metal")
  list(APPEND memory_sources ${memory_objcxx_sources})
  set_source_files_properties(${memory_objcxx_sources} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

# --- Target definition ---
add_library(Memory ${memory_sources} ${memory_headers})
xsigma_target_apply_lto(Memory MEMORY)
if(MEMORY_ENABLE_SANITIZER)
  xsigma_configure_sanitizer(MEMORY)
endif()

xsigma_find_linker(${MEMORY_LINKER_CHOICE} Memory MEMORY_LTO_MODE)
xsigma_target_apply_cache(Memory MEMORY_ENABLE_CACHE MEMORY_CACHE_BACKEND)
xsigma_target_optional_icecc(Memory MEMORY_ENABLE_ICECC)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(Memory PUBLIC MEMORY_SHARED_DEFINE PRIVATE MEMORY_BUILDING_DLL)
  if(COMMAND xsigma_set_target_output_directories)
    xsigma_set_target_output_directories(Memory)
  endif()
else()
  target_compile_definitions(Memory PUBLIC MEMORY_STATIC_DEFINE)
endif()

add_library(Memory::Memory ALIAS Memory)

if(MEMORY_GPU_BACKEND STREQUAL "cuda" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # CUDA is enabled in Memory's directory scope. Pass the detected toolkit to
  # CUDA consumers in sibling directories too, where CMake's language rules
  # may omit --cuda-path (notably Clang + Ninja on Windows).
  target_compile_options(
    Memory
    PUBLIC "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:--cuda-path=${CUDAToolkit_LIBRARY_ROOT}>>"
  )
endif()

xsigma_module_create_filters("${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(
  Memory PROPERTIES CXX_STANDARD ${MEMORY_CXX_STANDARD} CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
)

if(MSVC)
  target_compile_options(Memory PRIVATE /WX)
else()
  # Some toolchains implicitly provide standard headers; others (notably Clang 20.x in Debug builds)
  # have exhibited compiler instability with forced `-include cstdlib` on large translation units.
  # The project includes <cstdlib> where needed, so keep this workaround off for Clang.
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(Memory PRIVATE -include cstdlib)
  endif()
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Testing/MemoryTest.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/Testing/MemoryTest.h" @ONLY
)

# Include roots: Library/Memory → "allocator.h", "backend/…", "common/memory_export.h"
target_include_directories(
  Memory
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Testing>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/Testing>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# native/ (traceme/xplane/host_tracer/profiler_session) is always compiled into Profiler now,
# independent of PROFILER_BACKEND (Kineto/ITT) -- gate this on Profiler being available at all,
# the same condition MEMORY_HAS_PROFILER_BUILD already captures above.
if(MEMORY_HAS_PROFILER_BUILD)
  target_include_directories(Memory PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Library/Profiler>)
endif()

if(MEMORY_DEPENDENCY_INCLUDE_DIRS)
  target_include_directories(Memory PUBLIC ${MEMORY_DEPENDENCY_INCLUDE_DIRS})
endif()

if(MEMORY_ENABLE_CLANGTIDY)
  include(clang_tidy)
  xsigma_target_clang_tidy(Memory ${MEMORY_ENABLE_FIX})
endif()

if(MEMORY_ENABLE_IWYU)
  include(iwyu)
  xsigma_apply_iwyu(Memory)
endif()

if(MEMORY_DEPENDENCY_LIBS)
  target_link_libraries(Memory PUBLIC ${MEMORY_DEPENDENCY_LIBS})
endif()

if(MEMORY_DEPENDENCY_COMPILE_DEFINITIONS)
  target_compile_definitions(Memory PUBLIC ${MEMORY_DEPENDENCY_COMPILE_DEFINITIONS})
endif()

if(WIN32 AND NOT MSVC)
  target_link_libraries(Memory PRIVATE -ldbghelp)
endif()

if(MEMORY_ENABLE_MIMALLOC AND TARGET Memory::mimalloc)
  add_dependencies(Memory Memory::mimalloc)
endif()

# --- Installation ---
install(
  TARGETS Memory
  EXPORT MemoryTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
)
install(FILES ${memory_headers} DESTINATION include/Memory/Memory COMPONENT Development)

# --- Testing ---
if(MEMORY_ENABLE_TESTING)
  add_subdirectory(Testing/Cxx)
endif()

if(MEMORY_ENABLE_VALGRIND AND MEMORY_ENABLE_TESTING)
  xsigma_apply_valgrind_timeouts()
endif()

message("")
message("******** Memory module (CMake flags) ********")
message("  Mimalloc         : ${MEMORY_ENABLE_MIMALLOC}")
message("  Mimalloc stats   : ${MEMORY_ENABLE_MIMALLOC_STATS}")
message("  Logging           : required (Logging::Logging)")
message("  Profiler enable  : ${MEMORY_ENABLE_PROFILER}")
message("  Profiler (HAS)   : ${MEMORY_HAS_PROFILER_BUILD}")
message("  Memkind          : ${MEMORY_ENABLE_MEMKIND}")
message("  Numa             : ${MEMORY_ENABLE_NUMA}")
message("  Tbb              : ${MEMORY_ENABLE_TBB}")
message("  GPU backend      : ${MEMORY_GPU_BACKEND}")
message("  Cache            : ${MEMORY_ENABLE_CACHE}")
message("  Cache backend    : ${MEMORY_CACHE_BACKEND}")
message("  Icecc            : ${MEMORY_ENABLE_ICECC}")
xsigma_print_lto_linker_summary(MEMORY)
message("  Coverage         : ${MEMORY_ENABLE_COVERAGE}")
message("  Testing          : ${MEMORY_ENABLE_TESTING}")
message("  Examples         : ${MEMORY_ENABLE_EXAMPLES}")
message("  Gtest            : ${MEMORY_ENABLE_GTEST}")
message("  Benchmark        : ${MEMORY_ENABLE_BENCHMARK}")
message("  Clang-tidy       : ${MEMORY_ENABLE_CLANGTIDY}")
message("  Fix              : ${MEMORY_ENABLE_FIX}")
message("  Iwyu             : ${MEMORY_ENABLE_IWYU}")
message("  Sanitizer        : ${MEMORY_ENABLE_SANITIZER}")
message("  Sanitizer type   : ${MEMORY_SANITIZER_TYPE}")
message("  Spell            : ${MEMORY_ENABLE_SPELL}")
message("  Valgrind         : ${MEMORY_ENABLE_VALGRIND}")
if(MEMORY_GPU_BACKEND STREQUAL "cuda")
  message("CUDA (Memory):")
  message("  CUDA_INCLUDE_DIRS                      : ${CUDA_INCLUDE_DIRS}")
  message("  CUDA_NVCC_EXECUTABLE                   : ${CUDA_NVCC_EXECUTABLE}")
  message("  CMAKE_CUDA_FLAGS                       : ${CMAKE_CUDA_FLAGS}")
  message("  CUDA_HOST_COMPILER                     : ${CUDA_HOST_COMPILER}")
  message("  CUDA_SEPARABLE_COMPILATION             : ${CUDA_SEPARABLE_COMPILATION}")
  message("  CUDA_VERSION                           : ${CUDA_VERSION}")
  message("  PROJECT_CUDA_LIBRARIES                 : ${PROJECT_CUDA_LIBRARIES}")
  message("  CMAKE_CUDA_COMPILER                    : ${CMAKE_CUDA_COMPILER}")
  message("  CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES : ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()
message("")
message("Build:")
message("  Default C++ standard     : ${CMAKE_CXX_STANDARD}")
message("  C flags                  : ${CMAKE_C_FLAGS}")
message("  CXX flags                : ${CMAKE_CXX_FLAGS}")
message("  Shared linker flags      : ${CMAKE_SHARED_LINKER_FLAGS}")
message("  Static linker flags      : ${CMAKE_STATIC_LINKER_FLAGS}")
message("  Module linker flags      : ${CMAKE_MODULE_LINKER_FLAGS}")
get_directory_property(_summary_compile_defs DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
message("  Top-level compile defs   : ${_summary_compile_defs}")
message("")
