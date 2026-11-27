# =============================================================================
# XSigma CUDA
# Configuration Module

# This module handles CUDA compilation support for GPU acceleration. It manages CUDA toolkit
# detection, architecture configuration, and allocation strategy selection for GPU memory
# management.

# Include guard to prevent multiple inclusions
include_guard(GLOBAL)

# Default SM list shared with Vectorization (Turing+). sm_70 (Volta) was removed
# in CUDA 12.8; CUDA 13.x starts at sm_75. Keep Memory and Vectorization in sync.
set(XSIGMA_CUDA_ARCHITECTURES_DEFAULT "75;80;86;89;90")

# CUDA is not supported with the MinGW/MSYS2 toolchain in this project.
if(WIN32 AND (MINGW OR CMAKE_CXX_COMPILER MATCHES "msys64"))
  message(STATUS "CUDA: disabled on Windows+MinGW/MSYS2 toolchains")
  return()
endif()

# A parent project that already enabled CUDA owns the compiler choice.
if(CMAKE_CUDA_COMPILER_LOADED)
  # Nothing to select.
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CUDA_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "CUDA compiler (Clang)")
  message(STATUS "CUDA: using Clang ${CMAKE_CXX_COMPILER_VERSION} as CUDA compiler")
else()
  # With nvcc as CMAKE_CUDA_COMPILER, CMAKE_CXX_COMPILER only selects the compiler
  # CMake uses for plain CXX-language sources -- nvcc's own host-side codegen is
  # controlled by the separate CMAKE_CUDA_HOST_COMPILER variable (nvcc's -ccbin),
  # which CMake does NOT automatically default to CMAKE_CXX_COMPILER in this
  # project's configuration. Left unset, nvcc silently falls back to whatever
  # gcc/g++ resolves to on PATH, ignoring any g++-<N> version explicitly selected
  # via setup.py's compiler token (e.g. `gcc-12`). Confirmed: with
  # CMAKE_CXX_COMPILER=g++-12 selected but this unset, nvcc still invoked the
  # system default g++-13 for host compilation -- the resulting "internal
  # compiler error in try_forward_edges, cfgcleanup.cc" crash's bug-report
  # message pointed at gcc-13's README.Bugs, not gcc-12's, and reproduced the
  # exact GCC-13 ICE this file documents further below even though GCC 13 was
  # never the version requested.
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()
endif()

find_package(CUDAToolkit REQUIRED)

# CMake identifies Clang as a CUDA compiler by compiling a probe for a fixed GPU
# architecture. Releases before 3.31.10 only probe sm_52 and older, which
# CUDA 13 no longer supports, so identification fails.
if(NOT CMAKE_CUDA_COMPILER_LOADED
   AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
   AND CMAKE_CUDA_COMPILER STREQUAL CMAKE_CXX_COMPILER
   AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0
   AND CMAKE_VERSION VERSION_LESS 3.31.10
)
  message(
    FATAL_ERROR
      "Clang as the CUDA compiler with CUDA ${CUDAToolkit_VERSION} requires CMake 3.31.10 or "
      "newer (found ${CMAKE_VERSION})."
  )
endif()

# A parent that uses CUDA must enable it in its top-level directory before
# adding Memory: a language enabled here is not visible to sibling directories.
enable_language(CUDA)

# Version checks using consistent CUDAToolkit variables
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
  message(FATAL_ERROR "XSigma requires CUDA 12.0 or above. Found: ${CUDAToolkit_VERSION}")
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.2")
  message(FATAL_ERROR "XSigma CUDA support requires compiler version 9.2+")
endif()

# Check for version conflicts (nvcc only — Clang reports its own version, not the toolkit version)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND NOT CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL
                                                CUDAToolkit_VERSION
)
  message(
    FATAL_ERROR "Found two conflicting CUDA versions:\n"
                "Compiler V${CMAKE_CUDA_COMPILER_VERSION} and\n" "Toolkit V${CUDAToolkit_VERSION}"
  )
endif()

message(STATUS "XSigma: CUDA detected: ${CUDAToolkit_VERSION}")
message(STATUS "XSigma: CUDA compiler is: ${CMAKE_CUDA_COMPILER}")
message(STATUS "XSigma: CUDA toolkit directory: ${CUDAToolkit_ROOT}")

# Set C++ standard based on CUDA compiler: Clang:    inherits the host C++ standard (C++17 minimum)
# nvcc 12.0+: supports C++20 nvcc 11.0+: supports C++17 nvcc older: falls back to C++14
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
    message(STATUS "CUDA supports C++20 standard")
    set(CMAKE_CUDA_STANDARD 20)
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
    message(STATUS "CUDA supports C++17 standard")
    set(CMAKE_CUDA_STANDARD 17)
  else()
    set(CMAKE_CUDA_STANDARD 14)
  endif()
else()
  # Clang as CUDA compiler: match Memory's C++ standard.
  # MEMORY_CXX_STANDARD is set before include("${MEMORY_SOURCE_DIR}/Cmake/cuda.cmake"); CMAKE_CXX_STANDARD may still
  # be a CLI/cache value (e.g. CI's -DCMAKE_CXX_STANDARD=17) because Memory only
  # applies MEMORY_CXX_STANDARD to CMAKE_CXX_STANDARD after this include. Preferring
  # MEMORY_CXX_STANDARD keeps CUDA TUs on C++20 (needed for gtest u16string_view /
  # std::span) even when the top-level cache still says 17.
  if(DEFINED MEMORY_CXX_STANDARD AND MEMORY_CXX_STANDARD)
    set(CMAKE_CUDA_STANDARD ${MEMORY_CXX_STANDARD})
  elseif(CMAKE_CXX_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  else()
    set(CMAKE_CUDA_STANDARD 20)
  endif()
  message(STATUS "CUDA: Clang will use C++${CMAKE_CUDA_STANDARD} standard")
endif()

set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Use modern CMake CUDA architecture handling
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Default SM list shared with Vectorization (Turing+). Keep Memory and
# Vectorization in sync via XSIGMA_CUDA_ARCHITECTURES_DEFAULT (defined above).

# GPU Architecture options — only gens still supported by CUDA 12.8+ / 13.x.
set(PROJECT_CUDA_ARCH_OPTIONS "native" CACHE STRING "Which GPU Architecture(s) to compile for")
set_property(
  CACHE PROJECT_CUDA_ARCH_OPTIONS
  PROPERTY STRINGS
           native
           turing
           ampere
           ada
           hopper
           all
           none
)

# Set architectures based on user selection
if(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "native")
  set(CMAKE_CUDA_ARCHITECTURES "native")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "turing")
  set(CMAKE_CUDA_ARCHITECTURES "75")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "ampere")
  set(CMAKE_CUDA_ARCHITECTURES "80;86")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "ada")
  set(CMAKE_CUDA_ARCHITECTURES "89;90")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "hopper")
  set(CMAKE_CUDA_ARCHITECTURES "90")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "all")
  set(CMAKE_CUDA_ARCHITECTURES "${XSIGMA_CUDA_ARCHITECTURES_DEFAULT}")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "none")
  # Don't set any architectures, let parent project handle it
endif()

# Set up CUDA libraries using modern imported targets
set(PROJECT_CUDA_LIBRARIES CUDA::cudart CUDA::cuda_driver CUDA::cusparse CUDA::curand CUDA::cublas)

# Add nvperf_host if available (optional NVIDIA performance libraries)
if(TARGET CUDA::nvperf_host)
  list(APPEND PROJECT_CUDA_LIBRARIES CUDA::nvperf_host)
endif()

# Add CUDA libraries to the dependency list
list(APPEND PROJECT_DEPENDENCY_LIBS ${PROJECT_CUDA_LIBRARIES})

# Add include directories (using modern CUDAToolkit variables)
include_directories(SYSTEM "${CUDAToolkit_INCLUDE_DIRS}")
include_directories(SYSTEM "${CUDAToolkit_INCLUDE_DIRS}/thrust/system/cuda/detail")

# Add CUDA compiler flags
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  # nvcc-specific flags
  string(APPEND CMAKE_CUDA_FLAGS " -Xnvlink=--suppress-stack-size-warning")
  string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets --expt-extended-lambda")
  if(NOT MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      string(APPEND CMAKE_CUDA_FLAGS " -g -G")
    else()
      string(APPEND CMAKE_CUDA_FLAGS " -O3")
    endif()
  endif()
else()
  # Clang CUDA flags
  # Clang identifies only the CUDA releases it knows and warns about newer ones.
  string(APPEND CMAKE_CUDA_FLAGS " -Wno-unknown-cuda-version")
endif()

# For backward compatibility, set legacy variables (if needed elsewhere)
set(PROJECT_CUDA_FOUND TRUE)
