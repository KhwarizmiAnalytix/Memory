# =============================================================================
# Library/Memory — CMakeLists.txt
#
# CPU/GPU allocators, mimalloc integration, CUDA/HIP device memory, NUMA-aware
# allocation, and per-allocator statistics.
#
# Public compile definitions:
#   MEMORY_HAS_TBB=1/0       — TBB scalable allocator active
#   MEMORY_HAS_CUDA=1/0      — CUDA device-memory support compiled in
#   MEMORY_HAS_HIP=1/0       — HIP/ROCm device-memory support compiled in
#   MEMORY_HAS_MIMALLOC=1/0  — mimalloc high-perf allocator active
#   MEMORY_HAS_NUMA=1/0      — NUMA-aware allocation available
#   MEMORY_HAS_MEMKIND=1/0   — memkind extended-memory library available
#   MEMORY_HAS_LOGGING=1/0   — Logging linked and logging headers preferred (see MEMORY_ENABLE_LOGGING)
# =============================================================================
cmake_minimum_required(VERSION 3.16)

# --- Section 1: Feature flags (MEMORY_ENABLE_*) ---
# MEMORY_ENABLE_CUDA/HIP/TBB/NUMA are set by module-owned finders in Section 3.

# Link-Time Optimization — disabled when MEMORY_ENABLE_COVERAGE or MEMORY_ENABLE_SANITIZER is ON
# (LTO conflicts with profiling instrumentation and typical sanitizer workflows).
option(MEMORY_ENABLE_LTO "Enable Link Time Optimization for Memory" OFF)

# Coverage instrumentation — forces MEMORY_ENABLE_LTO=OFF and activates -fprofile-instr-generate /
# --coverage compiler flags for this module only.
option(MEMORY_ENABLE_COVERAGE "Enable coverage instrumentation for Memory" OFF)

# Testing — gates add_subdirectory(Testing/Cxx).
option(MEMORY_ENABLE_TESTING "Build Memory test suite" ON)

# Examples — gates add_subdirectory(Examples) when Memory examples exist.
option(MEMORY_ENABLE_EXAMPLES "Build Memory example programs" OFF)

# Google Test integration for Memory tests.
option(MEMORY_ENABLE_GTEST "Enable GoogleTest for Memory" ON)

# When ON and target Logging::Logging exists: link Logging, set MEMORY_HAS_LOGGING=1, and prepend
# Logging include dirs so "logger/logger.h" and other Logging headers resolve in Memory TUs.
option(MEMORY_ENABLE_LOGGING "Link Memory with the Logging library when available" ON)

# Microsoft mimalloc — high-performance drop-in malloc replacement. ON by default to match the CMake
# project default.
option(MEMORY_ENABLE_MIMALLOC "Enable Microsoft mimalloc high-performance memory allocator" ON)
mark_as_advanced(MEMORY_ENABLE_MIMALLOC)

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

option(
  MEMORY_ENABLE_SPELL
  "Enable spell checking with automatic corrections for Memory (WARNING: modifies source files)"
  OFF
)
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

# GPU allocation strategy — controls cudaMallocAsync / cudaMemPool behaviour. SYNC       —
# synchronous cudaMalloc (safest, slowest) ASYNC      — asynchronous cudaMallocAsync POOL_ASYNC —
# asynchronous pool-backed cudaMallocAsync (default, fastest)
set(MEMORY_GPU_ALLOC "POOL_ASYNC" CACHE STRING
                                        "GPU allocation strategy. Options: SYNC, ASYNC, POOL_ASYNC"
)
set_property(CACHE MEMORY_GPU_ALLOC PROPERTY STRINGS SYNC ASYNC POOL_ASYNC)
if(NOT MEMORY_GPU_ALLOC MATCHES "^(SYNC|ASYNC|POOL_ASYNC)$")
  message(
    FATAL_ERROR
      "Invalid MEMORY_GPU_ALLOC '${MEMORY_GPU_ALLOC}'. Valid options: SYNC, ASYNC, POOL_ASYNC"
  )
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

# --- Section 3: CMake module includes ---
# Order: introspection → toolchain helpers → Memory-owned finders → coverage/sanitizer → LTO/linker.

# LTO is incompatible with coverage/sanitizer — clear before any include().
if(MEMORY_ENABLE_COVERAGE OR MEMORY_ENABLE_SANITIZER)
  set(MEMORY_ENABLE_LTO OFF)
endif()

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
include(cuda) # Detects/configures CUDA when MEMORY_ENABLE_CUDA=ON
include(hip) # Detects/configures HIP/ROCm when MEMORY_ENABLE_HIP=ON
include(tbb_memory) # Finds/builds TBB scalable allocator when MEMORY_ENABLE_TBB=ON
include(numa) # Finds NUMA library when MEMORY_ENABLE_NUMA=ON
if(MEMORY_ENABLE_SANITIZER)
  set(MEMORY_ENABLE_MIMALLOC OFF)
  include(sanitize)
  quarisma_configure_sanitizer(MEMORY)
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

if(MEMORY_ENABLE_LTO AND NOT MEMORY_ENABLE_COVERAGE AND NOT MEMORY_ENABLE_SANITIZER)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  message(STATUS "Link Time Optimization (LTO) enabled via CMAKE_INTERPROCEDURAL_OPTIMIZATION")
else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
  if(MEMORY_ENABLE_LTO AND (MEMORY_ENABLE_COVERAGE OR MEMORY_ENABLE_SANITIZER))
    message(
      STATUS "Link Time Optimization (LTO) disabled (incompatible with coverage and/or sanitizer)"
    )
  else()
    message(STATUS "Link Time Optimization (LTO) disabled")
  endif()
endif()

include(linker)

# --- Third-party dependencies owned by Memory ---
set(_tpdir "${CMAKE_SOURCE_DIR}/ThirdParty")

set(BUILD_TESTING OFF CACHE BOOL "Disable testing for third-party libraries" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable examples for third-party libraries" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable benchmarks for third-party libraries" FORCE)

if(MEMORY_ENABLE_MIMALLOC)
  # mimalloc-specific options
  set(MI_BUILD_TESTS OFF CACHE BOOL "Disable mimalloc tests" FORCE)
  set(MI_BUILD_OBJECT OFF CACHE BOOL "Disable mimalloc object library" FORCE)
  set(MI_INSTALL_TOPLEVEL OFF CACHE BOOL "Disable mimalloc top-level install" FORCE)
  set(MI_OVERRIDE OFF CACHE BOOL "Disable mimalloc malloc override" FORCE)
  set(MI_BUILD_SHARED OFF CACHE BOOL "Build mimalloc as static library" FORCE)
  set(MI_BUILD_STATIC ON CACHE BOOL "Build mimalloc as static library" FORCE)

  add_third_party_library(mimalloc
      CONDITION MEMORY_ENABLE_MIMALLOC
      LIBRARY_TYPE STATIC
      SOURCE_DIR "${_tpdir}/mimalloc"
      LIBRARY_TYPE_VAR MI_BUILD_STATIC
      # Do not list -Wno-strict-prototypes here: FILTER_FLAGS removes flags from CMAKE_*_FLAGS,
      # which would drop the toolchain's prototype warning suppression for mimalloc C sources.
      FILTER_FLAGS "-flto=thin" "-flto=full"
      QUARISMA_TARGETS "Mimalloc::mimalloc=mimalloc|mimalloc-static"
      INCLUDE_DIR "${_tpdir}/mimalloc/include"
      AUTO_ADD_TO_DEPENDENCIES OFF
  )

  list(APPEND MEMORY_DEPENDENCY_INCLUDE_DIRS "${_tpdir}/mimalloc/include")


  # MinGW: mimalloc links Windows system libs as bare names; patch to -l<name>.
  if(WIN32 AND (MINGW OR CMAKE_CXX_COMPILER MATCHES "msys64") AND TARGET mimalloc-static)
    set(_mi_win_syslibs psapi shell32 user32 advapi32 bcrypt)
    get_target_property(_mi_ll mimalloc-static LINK_LIBRARIES)
    if(_mi_ll)
      set(_changed FALSE)
      foreach(_lib IN LISTS _mi_win_syslibs)
        if("${_lib}" IN_LIST _mi_ll)
          list(REMOVE_ITEM _mi_ll "${_lib}")
          list(APPEND     _mi_ll "-l${_lib}")
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
          list(APPEND     _mi_il "-l${_lib}")
          set(_changed TRUE)
        endif()
        set(_ge  "$<LINK_ONLY:${_lib}>")
        set(_ge2 "$<LINK_ONLY:-l${_lib}>")
        if("${_ge}" IN_LIST _mi_il)
          list(REMOVE_ITEM _mi_il "${_ge}")
          list(APPEND     _mi_il "${_ge2}")
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
# --- Section 2: MEMORY_ENABLE_* → MEMORY_HAS_* compile definitions ---
# TBB/CUDA/HIP/NUMA flags come from Section 3 finders; mapped after those includes.
include(compile_definitions)
# Per-allocator call-site tracking (opt-in; adds overhead).
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_ALLOCATION_STATS)
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_MIMALLOC)
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_MEMKIND)
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_NUMA)
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_TBB)
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_CUDA)
compile_definition(MEMORY_COMPILE_DEFINITIONS MEMORY_ENABLE_HIP)
list(APPEND MEMORY_DEPENDENCY_COMPILE_DEFINITIONS ${MEMORY_COMPILE_DEFINITIONS})

# Optional Logging sibling module — only wire includes/link when MEMORY_ENABLE_LOGGING and target exists.
set(MEMORY_HAS_LOGGING_BUILD 0)
if(MEMORY_ENABLE_LOGGING AND TARGET Logging::Logging)
  set(MEMORY_HAS_LOGGING_BUILD 1)
  list(APPEND MEMORY_DEPENDENCY_LIBS Logging::Logging)
endif()
list(APPEND MEMORY_DEPENDENCY_COMPILE_DEFINITIONS MEMORY_HAS_LOGGING=${MEMORY_HAS_LOGGING_BUILD})

# --- Runtime library wiring ---
if(TARGET Fmt::fmt)
  list(APPEND MEMORY_DEPENDENCY_LIBS Fmt::fmt)
endif()
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

# --- Source discovery ---
set(memory_headers "")
set(memory_sources "")
set(CMAKE_CXX_STANDARD ${MEMORY_CXX_STANDARD})

file(GLOB_RECURSE memory_headers LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
                                                        "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx"
)
file(GLOB_RECURSE memory_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Keep test sources out of the Memory library target.
list(FILTER memory_sources EXCLUDE REGEX ".*/Testing/.*")
list(FILTER memory_headers EXCLUDE REGEX ".*/Testing/.*")

# GPU sources are only compiled when at least one GPU backend is active.
if(NOT MEMORY_ENABLE_CUDA AND NOT MEMORY_ENABLE_HIP)
  file(GLOB_RECURSE GPU_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cpp")
  file(GLOB_RECURSE GPU_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.h"
       "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.hxx"
  )
  list(REMOVE_ITEM memory_headers ${GPU_HEADER_FILES})
  list(REMOVE_ITEM memory_sources ${GPU_SOURCE_FILES})
  list(REMOVE_ITEM memory_sources "${CMAKE_CURRENT_SOURCE_DIR}/cpu/allocator_device.cpp")
  list(REMOVE_ITEM memory_headers "${CMAKE_CURRENT_SOURCE_DIR}/cpu/allocator_device.h")
endif()

# --- Target definition ---
add_library(Memory ${memory_sources} ${memory_headers})
quarisma_find_linker(${MEMORY_LINKER_CHOICE} Memory)
quarisma_target_apply_cache(Memory MEMORY_ENABLE_CACHE MEMORY_CACHE_BACKEND)
quarisma_target_optional_icecc(Memory MEMORY_ENABLE_ICECC)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(Memory PUBLIC MEMORY_SHARED_DEFINE PRIVATE MEMORY_BUILDING_DLL)
  if(COMMAND quarisma_set_target_output_directories)
    quarisma_set_target_output_directories(Memory)
  endif()
else()
  target_compile_definitions(Memory PUBLIC MEMORY_STATIC_DEFINE)
endif()

add_library(Memory::Memory ALIAS Memory)

quarisma_module_create_filters("${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(
  Memory PROPERTIES CXX_STANDARD ${MEMORY_CXX_STANDARD} CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
)

if(MSVC)
  target_compile_options(Memory PRIVATE /WX)
else()
  target_compile_options(Memory PRIVATE -include cstdlib)
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

if(PROFILER_BACKEND STREQUAL "NATIVE")
  target_include_directories(Memory PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Library/Profiler>)
endif()

if(MEMORY_DEPENDENCY_INCLUDE_DIRS)
  target_include_directories(Memory PUBLIC ${MEMORY_DEPENDENCY_INCLUDE_DIRS})
endif()

# Prepend Logging roots only when MEMORY_HAS_LOGGING=1 for logger and related headers without
# pulling Logging into the include path when logging integration is disabled.
if(MEMORY_HAS_LOGGING_BUILD)
  target_include_directories(
    Memory
    BEFORE
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Library/Logging>"
      "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Library/Logging/logger>"
      "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/Library/Logging>"
      "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Library/Logging/Testing>"
      "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/Library/Logging/Testing>"
  )
endif()

if(MEMORY_ENABLE_CLANGTIDY)
  include(clang_tidy)
  quarisma_target_clang_tidy(Memory ${MEMORY_ENABLE_FIX})
endif()

if(MEMORY_ENABLE_IWYU)
  include(iwyu)
  quarisma_apply_iwyu(Memory)
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
  quarisma_apply_valgrind_timeouts()
endif()

message("")
message("******** Memory module (CMake flags) ********")
message("  Mimalloc         : ${MEMORY_ENABLE_MIMALLOC}")
message("  Logging enable   : ${MEMORY_ENABLE_LOGGING}")
message("  Logging (HAS)    : ${MEMORY_HAS_LOGGING_BUILD}")
message("  Memkind          : ${MEMORY_ENABLE_MEMKIND}")
message("  Numa             : ${MEMORY_ENABLE_NUMA}")
message("  Tbb              : ${MEMORY_ENABLE_TBB}")
message("  Cuda             : ${MEMORY_ENABLE_CUDA}")
message("  Hip              : ${MEMORY_ENABLE_HIP}")
message("  Gpu alloc        : ${MEMORY_GPU_ALLOC}")
message("  Cache            : ${MEMORY_ENABLE_CACHE}")
message("  Cache backend    : ${MEMORY_CACHE_BACKEND}")
message("  Icecc            : ${MEMORY_ENABLE_ICECC}")
message("  Linker           : ${MEMORY_LINKER_CHOICE}")
message("  Lto              : ${MEMORY_ENABLE_LTO}")
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
if(MEMORY_ENABLE_CUDA)
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
message("  IPO / LTO (CMake)        : ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
get_directory_property(_summary_compile_defs DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
message("  Top-level compile defs   : ${_summary_compile_defs}")
message("")
