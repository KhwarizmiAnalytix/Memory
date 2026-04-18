# =============================================================================
# Library/Memory — CMakeLists.txt
#
# Provides the Memory library: CPU/GPU allocators (BFC, pool, tracking, retry),
# mimalloc integration, CUDA/HIP device memory, NUMA-aware allocation, and
# per-allocator statistics.
#
# Public compile definitions emitted by this file:
#   MEMORY_HAS_TBB=1/0       — TBB scalable allocator active
#   MEMORY_HAS_CUDA=1/0      — CUDA device-memory support compiled in
#   MEMORY_HAS_HIP=1/0       — HIP/ROCm device-memory support compiled in
#   MEMORY_HAS_MIMALLOC=1/0  — mimalloc high-perf allocator active
#   MEMORY_HAS_NUMA=1/0      — NUMA-aware allocation available
#   MEMORY_HAS_MEMKIND=1/0   — memkind extended-memory library available
# =============================================================================
cmake_minimum_required(VERSION 3.16)

# =============================================================================
# SECTION 1 — Feature flags (MEMORY_ENABLE_*) and backend selectors
#
# These cache variables are the public configuration API for this module.
# Pass them on the CMake command line: -DMEMORY_ENABLE_CUDA=ON
#
# Note: MEMORY_ENABLE_CUDA is declared inside include(cuda) and
#       MEMORY_ENABLE_HIP inside include(hip) — both are module-owned finders
#       located in Library/Memory/Cmake/ and included in Section 3.
#       MEMORY_ENABLE_TBB and MEMORY_ENABLE_NUMA are likewise owned by
#       include(tbb_memory) and include(numa) respectively.
# =============================================================================

# Link-Time Optimization — automatically disabled when MEMORY_ENABLE_COVERAGE=ON
# because LTO and LLVM profiling instrumentation are mutually incompatible.
option(MEMORY_ENABLE_LTO
  "Enable Link Time Optimization for Memory" OFF)

# Coverage instrumentation — forces MEMORY_ENABLE_LTO=OFF and activates
# -fprofile-instr-generate / --coverage compiler flags for this module only.
option(MEMORY_ENABLE_COVERAGE
  "Enable coverage instrumentation for Memory" OFF)

# Testing — gates add_subdirectory(Testing/Cxx).
option(MEMORY_ENABLE_TESTING
  "Build Memory test suite" ON)

# Examples — gates add_subdirectory(Examples) when Memory examples exist.
option(MEMORY_ENABLE_EXAMPLES
  "Build Memory example programs" OFF)

# Google Test integration for Memory tests.
option(MEMORY_ENABLE_GTEST
  "Enable GoogleTest for Memory" ON)
  

# Microsoft mimalloc — high-performance drop-in malloc replacement.
# ON by default to match the CMake project default.
option(MEMORY_ENABLE_MIMALLOC
  "Enable Microsoft mimalloc high-performance memory allocator" ON)
mark_as_advanced(MEMORY_ENABLE_MIMALLOC)

# memkind — Linux-only extended-memory allocator (pmem, HBM, etc.).
# Forced OFF on non-Linux platforms via the platform-constraint block below.
option(MEMORY_ENABLE_MEMKIND
  "Enable memkind extended-memory allocator (Linux only)" OFF)

# Google Benchmark integration for micro-benchmarking allocator paths.
option(MEMORY_ENABLE_BENCHMARK
  "Enable Google Benchmark for Memory" OFF)

# Icecream distributed compilation launcher (rarely needed; hidden by default).
option(MEMORY_ENABLE_ICECC
  "Use Icecream (icecc) distributed compiler for Memory" OFF)
mark_as_advanced(MEMORY_ENABLE_ICECC)

# Per-target compiler cache — wraps every compile command with ccache / sccache
# etc. to speed up incremental and CI rebuilds. ON by default.
option(MEMORY_ENABLE_CACHE
  "Enable compiler caching for the Memory target" ON)
mark_as_advanced(MEMORY_ENABLE_CACHE)
set(MEMORY_CACHE_BACKEND "none"
    CACHE STRING
    "Compiler cache backend for Memory. Options: none, ccache, sccache, buildcache")
set_property(CACHE MEMORY_CACHE_BACKEND PROPERTY STRINGS none ccache sccache buildcache)
mark_as_advanced(MEMORY_CACHE_BACKEND)

# GPU allocation strategy — controls cudaMallocAsync / cudaMemPool behaviour.
#   SYNC       — synchronous cudaMalloc (safest, slowest)
#   ASYNC      — asynchronous cudaMallocAsync
#   POOL_ASYNC — asynchronous pool-backed cudaMallocAsync (default, fastest)
set(MEMORY_GPU_ALLOC "POOL_ASYNC"
    CACHE STRING "GPU allocation strategy. Options: SYNC, ASYNC, POOL_ASYNC")
set_property(CACHE MEMORY_GPU_ALLOC PROPERTY STRINGS SYNC ASYNC POOL_ASYNC)
if(NOT MEMORY_GPU_ALLOC MATCHES "^(SYNC|ASYNC|POOL_ASYNC)$")
  message(FATAL_ERROR
    "Invalid MEMORY_GPU_ALLOC '${MEMORY_GPU_ALLOC}'. Valid options: SYNC, ASYNC, POOL_ASYNC")
endif()

# C++ standard — applied to every TU in this module via CMAKE_CXX_STANDARD.
set(MEMORY_CXX_STANDARD "20"
    CACHE STRING "C++ standard for Memory. Options: 11 14 17 20 23")
set_property(CACHE MEMORY_CXX_STANDARD PROPERTY STRINGS 11 14 17 20 23)
if(NOT MEMORY_CXX_STANDARD MATCHES "^(11|14|17|20|23)$")
  message(FATAL_ERROR "MEMORY_CXX_STANDARD must be one of: 11 14 17 20 23")
endif()

# Platform constraints — NUMA and memkind are Linux-only kernel/library interfaces.
# Silently force them OFF on Windows and macOS so downstream callers do not need
# to guard every reference to MEMORY_ENABLE_NUMA / MEMORY_ENABLE_MEMKIND.
if(NOT UNIX)
  set(MEMORY_ENABLE_NUMA    OFF CACHE BOOL "Enable NUMA node allocation (Linux only)" FORCE)
endif()
if(NOT UNIX OR APPLE)
  set(MEMORY_ENABLE_MEMKIND OFF CACHE BOOL "Enable memkind allocator (Linux only)"   FORCE)
endif()

# =============================================================================
# SECTION 2 — MEMORY_ENABLE_* → MEMORY_HAS_* compile definitions
#
# User-declared option flags (mimalloc, memkind) are mapped here because they
# are set in Section 1 and do not depend on any include.
#
# IMPORTANT: MEMORY_ENABLE_TBB, MEMORY_ENABLE_CUDA, MEMORY_ENABLE_HIP, and
# MEMORY_ENABLE_NUMA are set by the module-owned finders inside Section 3
# (include(tbb_memory), include(cuda), include(hip), include(numa)).
# Those flags DO NOT exist yet here — their HAS_ mappings are placed AFTER
# Section 3 in the "Section 2 — detection-based HAS" block below.
# =============================================================================
set(MEMORY_DEPENDENCY_COMPILE_DEFINITIONS ${PROJECT_DEPENDENCY_COMPILE_DEFINITIONS})

# mimalloc — declared as an option in Section 1; safe to map here.
if(MEMORY_ENABLE_MIMALLOC)
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_MIMALLOC=1)
else()
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_MIMALLOC=0)
endif()

# memkind — declared as an option in Section 1; forced OFF on non-Linux above.
if(MEMORY_ENABLE_MEMKIND)
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_MEMKIND=1)
else()
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_MEMKIND=0)
endif()

# Per-allocator call-site tracking (opt-in; adds overhead).
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_ALLOCATION_STATS)

# =============================================================================
# SECTION 3 — CMake module includes
#
# Inclusion order:
#   a) Standard CMake introspection helpers
#   b) Shared toolchain helpers  (Cmake/tools/, Cmake/flags/)
#   c) Memory-owned feature finders (Library/Memory/Cmake/) — sole owners of
#      their MEMORY_ENABLE_* flags; may override user choices based on
#      platform availability
#   d) Coverage gate — must follow all feature finders so that
#      MEMORY_ENABLE_LTO is in scope and CUDA/HIP/TBB flags are resolved
# =============================================================================

# a) Standard CMake introspection
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)
include(CheckSymbolExists)

# b) Shared toolchain helpers
include(compiler_checks)    # Detects compiler capabilities; sets PROJECT_* flags
include(cache)              # Configures per-target compiler cache launcher
include(build_type)         # Validates / normalises CMAKE_BUILD_TYPE
include(checks)             # Additional feature-check macros
include(utils)              # Utility macros shared across all modules
include(platform)           # Platform flags (macOS rpath, Windows /utf-8, …)
include(CMakeDependentOption)

# c) Memory-owned feature finders (Library/Memory/Cmake/)
include(cuda)          # Detects/configures CUDA when MEMORY_ENABLE_CUDA=ON
include(hip)           # Detects/configures HIP/ROCm when MEMORY_ENABLE_HIP=ON
include(tbb_memory)    # Finds/builds TBB scalable allocator when MEMORY_ENABLE_TBB=ON
include(numa)          # Finds NUMA library when MEMORY_ENABLE_NUMA=ON
include(sanitize)
include(clang_tidy)
include(spell)
include(iwyu)
include(valgrind)
include(helper_macros)
include(linker)
# d) Coverage gate
#    When enabled, appends -fprofile-instr-generate / --coverage to global
#    CMAKE_CXX_FLAGS for this module.  LTO is forced OFF here because LTO
#    can move or eliminate profile-counter stores, corrupting the .profraw data.
if(MEMORY_ENABLE_COVERAGE)
  set(MEMORY_ENABLE_LTO OFF)
  include(coverage)
endif()

# =============================================================================
# SECTION 2 (continued) — detection-based MEMORY_HAS_* compile definitions
#
# MEMORY_ENABLE_TBB, MEMORY_ENABLE_CUDA, MEMORY_ENABLE_HIP, and
# MEMORY_ENABLE_NUMA are set by the module-owned finders in Section 3 above.
# This block MUST appear after those includes so the flags are readable.
# =============================================================================

# TBB scalable allocator — set by include(tbb_memory).
if(MEMORY_ENABLE_TBB)
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_TBB=1)
else()
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_TBB=0)
endif()

# CUDA device memory — set by include(cuda).
if(MEMORY_ENABLE_CUDA)
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_CUDA=1)
else()
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_CUDA=0)
endif()

# HIP/ROCm device memory — set by include(hip).
if(MEMORY_ENABLE_HIP)
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_HIP=1)
else()
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_HIP=0)
endif()

# NUMA-aware allocation — set by include(numa); Linux only.
if(MEMORY_ENABLE_NUMA)
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_NUMA=1)
else()
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_NUMA=0)
endif()

list(APPEND MEMORY_DEPENDENCY_COMPILE_DEFINITIONS ${MEMORY_COMPILE_DEFINITIONS})

# =============================================================================
# Runtime library wiring
# Collect third-party targets found by the Section 3 finders into
# MEMORY_DEPENDENCY_LIBS for the single target_link_libraries() call below.
# =============================================================================
if(MEMORY_ENABLE_CUDA AND DEFINED PROJECT_CUDA_LIBRARIES)
  list(APPEND MEMORY_DEPENDENCY_LIBS ${PROJECT_CUDA_LIBRARIES})
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

# =============================================================================
# Source discovery
# =============================================================================
set(memory_headers "")
set(memory_sources "")

# Per-module override — propagates to all targets in this directory scope,
# including test targets in Testing/Cxx.  CMAKE_CXX_STANDARD_REQUIRED and
# CMAKE_CXX_EXTENSIONS are set globally in the root CMakeLists.txt.
set(CMAKE_CXX_STANDARD ${MEMORY_CXX_STANDARD})

file(GLOB_RECURSE memory_headers LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx"
)
file(GLOB_RECURSE memory_sources LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Keep test sources out of the Memory library target.
list(FILTER memory_sources EXCLUDE REGEX ".*/Testing/.*")
list(FILTER memory_headers EXCLUDE REGEX ".*/Testing/.*")

# GPU sources are only compiled when at least one GPU backend is active.
if(NOT MEMORY_ENABLE_CUDA AND NOT MEMORY_ENABLE_HIP)
  file(GLOB_RECURSE GPU_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cpp")
  file(GLOB_RECURSE GPU_HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.hxx")
  list(REMOVE_ITEM memory_headers ${GPU_HEADER_FILES})
  list(REMOVE_ITEM memory_sources ${GPU_SOURCE_FILES})
  list(REMOVE_ITEM memory_sources "${CMAKE_CURRENT_SOURCE_DIR}/cpu/allocator_device.cpp")
  list(REMOVE_ITEM memory_headers "${CMAKE_CURRENT_SOURCE_DIR}/cpu/allocator_device.h")
endif()

# =============================================================================
# Target definition
# =============================================================================
if(MEMORY_ENABLE_LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  message(STATUS "Link Time Optimization (LTO) enabled via CMAKE_INTERPROCEDURAL_OPTIMIZATION")
else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
  message(STATUS "Link Time Optimization (LTO) disabled")
endif()

add_library(Memory ${memory_sources} ${memory_headers})
quarisma_target_apply_cache(Memory MEMORY_ENABLE_CACHE MEMORY_CACHE_BACKEND)
quarisma_target_optional_icecc(Memory MEMORY_ENABLE_ICECC)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(Memory
    PUBLIC  MEMORY_SHARED_DEFINE
    PRIVATE MEMORY_BUILDING_DLL)
  if(COMMAND memory_set_target_output_directories)
    memory_set_target_output_directories(Memory)
  endif()
else()
  target_compile_definitions(Memory PUBLIC MEMORY_STATIC_DEFINE)
endif()

add_library(Memory::Memory ALIAS Memory)

if(DEFINED MEMORY_LINKER_FLAGS_COMMON)
  target_link_options(Memory PRIVATE ${MEMORY_LINKER_FLAGS_COMMON})
endif()


if(COMMAND memory_apply_iwyu)
  memory_apply_iwyu(Memory)
endif()

if(COMMAND memory_module_create_filters)
  memory_module_create_filters("${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set_target_properties(Memory PROPERTIES
  CXX_STANDARD          ${MEMORY_CXX_STANDARD}
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS        OFF
)

if(MSVC)
  target_compile_options(Memory PRIVATE /WX)
else()
  target_compile_options(Memory PRIVATE -include cstdlib)
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Testing/MemoryTest.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/Testing/MemoryTest.h"
  @ONLY
)

# Include roots: Library/Memory → "allocator.h", "backend/…", "common/memory_export.h"
target_include_directories(Memory
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Testing>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/Testing>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

if(PROFILER_BACKEND STREQUAL "NATIVE")
  target_include_directories(Memory PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Library/Profiler>)
endif()

if(MEMORY_DEPENDENCY_INCLUDE_DIRS)
  target_include_directories(Memory PUBLIC ${MEMORY_DEPENDENCY_INCLUDE_DIRS})
endif()

target_link_libraries(Memory PUBLIC Memory::build)

if(COMMAND memory_target_clang_tidy)
  memory_target_clang_tidy(Memory)
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

# =============================================================================
# Installation
# =============================================================================
install(
  TARGETS Memory
  EXPORT  MemoryTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(FILES ${memory_headers}
  DESTINATION include/Memory/Memory
  COMPONENT Development)

# =============================================================================
# Testing
# =============================================================================
if(MEMORY_ENABLE_TESTING)
  add_subdirectory(Testing/Cxx)
endif()

if(PROJECT_ENABLE_VALGRIND AND MEMORY_ENABLE_TESTING)
  quarisma_apply_valgrind_timeouts()
endif()
