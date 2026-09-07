# =============================================================================
# XSigma HIP
# (Heterogeneous-compute Interface for Portability) Configuration Module

# This module configures HIP for AMD GPU acceleration and ROCm support. It manages HIP toolkit
# detection, architecture configuration, and GPU compilation.

# Include guard to prevent multiple inclusions
include_guard(GLOBAL)

# HIP/ROCm is a Unix (Linux) technology: the ROCm toolkit and hipcc are not distributed or
# validated for Windows in this project. Fail fast with a clear message instead of silently
# attempting find_package(hip)/enable_language(HIP), which — even if a Windows HIP SDK happens
# to be on PATH — has never been built or tested here and uses a different compiler front end
# than the flags below assume. Use MEMORY_GPU_BACKEND=cuda on Windows instead.
if(WIN32)
  message(
    FATAL_ERROR
      "MEMORY_GPU_BACKEND=hip is not supported on Windows in this project (HIP/ROCm is Unix-only "
      "here). Use MEMORY_GPU_BACKEND=cuda on Windows, or build on Linux for HIP."
  )
endif()

# HIP requires CMake 3.21 or later for proper support
if(CMAKE_VERSION VERSION_LESS "3.21")
  message(FATAL_ERROR "HIP support requires CMake 3.21 or later. Found: ${CMAKE_VERSION}")
endif()

# /opt/rocm is not a CMake default prefix; HIP_PATH=/usr (dirname of the
# alternatives hipcc wrapper) is not an SDK root. Resolve a real prefix first.
include(rocm_prefix)
xsigma_setup_rocm_prefix()

# Find HIP package
find_package(hip REQUIRED)

if(NOT hip_FOUND)
  message(FATAL_ERROR "HIP not found. Please install ROCm/HIP and ensure it's in your PATH.")
endif()

# Enable HIP language support
enable_language(HIP)

# Version checks
if(hip_VERSION VERSION_LESS "5.0")
  message(FATAL_ERROR "XSigma requires HIP 5.0 or above. Found: ${hip_VERSION}")
endif()

message(STATUS "XSigma: HIP detected: ${hip_VERSION}")
message(STATUS "XSigma: HIP compiler is: ${CMAKE_HIP_COMPILER}")
message(STATUS "XSigma: HIP toolkit directory: ${HIP_ROOT_DIR}")

# Set C++ standard for HIP
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Use modern CMake HIP architecture handling
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES "native")
endif()

# GPU Architecture options for AMD GPUs
set(PROJECT_HIP_ARCH_OPTIONS "native" CACHE STRING "Which AMD GPU Architecture(s) to compile for")
set_property(
  CACHE PROJECT_HIP_ARCH_OPTIONS
  PROPERTY STRINGS
           native
           gfx900 # Vega 10 (RX Vega 56/64)
           gfx906 # Vega 20 (Radeon VII, MI50/60)
           gfx908 # CDNA (MI100)
           gfx90a # CDNA2 (MI200 series)
           gfx1030 # RDNA2 (RX 6000 series)
           gfx1100 # RDNA3 (RX 7000 series)
           all
           none
)

# Set architectures based on user selection
if(PROJECT_HIP_ARCH_OPTIONS STREQUAL "native")
  # Let CMake handle native detection
  set(CMAKE_HIP_ARCHITECTURES "native")
elseif(PROJECT_HIP_ARCH_OPTIONS STREQUAL "gfx900")
  set(CMAKE_HIP_ARCHITECTURES "gfx900")
elseif(PROJECT_HIP_ARCH_OPTIONS STREQUAL "gfx906")
  set(CMAKE_HIP_ARCHITECTURES "gfx906")
elseif(PROJECT_HIP_ARCH_OPTIONS STREQUAL "gfx908")
  set(CMAKE_HIP_ARCHITECTURES "gfx908")
elseif(PROJECT_HIP_ARCH_OPTIONS STREQUAL "gfx90a")
  set(CMAKE_HIP_ARCHITECTURES "gfx90a")
elseif(PROJECT_HIP_ARCH_OPTIONS STREQUAL "gfx1030")
  set(CMAKE_HIP_ARCHITECTURES "gfx1030")
elseif(PROJECT_HIP_ARCH_OPTIONS STREQUAL "gfx1100")
  set(CMAKE_HIP_ARCHITECTURES "gfx1100")
elseif(PROJECT_HIP_ARCH_OPTIONS STREQUAL "all")
  set(CMAKE_HIP_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100")
elseif(PROJECT_HIP_ARCH_OPTIONS STREQUAL "none")
  # Don't set any architectures, let parent project handle it
endif()

# Set up HIP libraries using modern imported targets
# Host runtime only on the PUBLIC interface. hip::device injects `-x hip
# --offload-arch=...` into every consumer TU; leaking it PUBLIC compiles host
# SIMD tests (TestSimdUtility.cpp) as device code.
set(PROJECT_HIP_LIBRARIES hip::host)

# Add HIP libraries to the dependency list
list(APPEND PROJECT_DEPENDENCY_LIBS ${PROJECT_HIP_LIBRARIES})

# Add include directories
include_directories(SYSTEM "${HIP_INCLUDE_DIRS}")

# Add common HIP flags.
# NOTE: --expt-extended-lambda is an nvcc-only flag and was previously copy-pasted here from
# cuda.cmake; hipcc (Clang-based) does not recognize it and errors out — do not re-add it (see
# Docs/CUDA_HIP_REMEDIATION_PLAN.md, D4). HIP is Unix-only in this project (see the WIN32 guard
# above), so no MSVC branch is needed here.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  string(APPEND CMAKE_HIP_FLAGS " -g")
else()
  string(APPEND CMAKE_HIP_FLAGS " -O3")
endif()

# For backward compatibility, set legacy variables (if needed elsewhere)
set(PROJECT_HIP_FOUND TRUE)

add_compile_definitions(MEMORY_ENABLE_GPU)
