# ============================================================================= Quarisma HIP
# (Heterogeneous-compute Interface for Portability) Configuration Module

# This module configures HIP for AMD GPU acceleration and ROCm support. It manages HIP toolkit
# detection, architecture configuration, and GPU compilation.

# Include guard to prevent multiple inclusions
include_guard(GLOBAL)

# HIP GPU Support Flag Controls whether HIP GPU acceleration is enabled for AMD GPUs. When enabled,
# requires CMake 3.21+ and ROCm/HIP toolkit. Mutually exclusive with MEMORY_ENABLE_CUDA.
cmake_dependent_option(
  MEMORY_ENABLE_HIP "Support HIP backend accelerator" OFF
  "CMAKE_VERSION VERSION_GREATER_EQUAL 3.21;NOT MEMORY_ENABLE_CUDA" OFF
)
mark_as_advanced(MEMORY_ENABLE_HIP)

if(NOT MEMORY_ENABLE_HIP)
  return()
endif()

# HIP requires CMake 3.21 or later for proper support
if(CMAKE_VERSION VERSION_LESS "3.21")
  message(FATAL_ERROR "HIP support requires CMake 3.21 or later. Found: ${CMAKE_VERSION}")
endif()

# Ensure CUDA is not enabled when HIP is enabled
if(MEMORY_ENABLE_CUDA)
  message(FATAL_ERROR "Cannot enable both CUDA and HIP simultaneously. Please choose one.")
endif()

# Find HIP package
find_package(hip REQUIRED)

if(NOT hip_FOUND)
  message(FATAL_ERROR "HIP not found. Please install ROCm/HIP and ensure it's in your PATH.")
endif()

# Enable HIP language support
enable_language(HIP)

# Version checks
if(hip_VERSION VERSION_LESS "5.0")
  message(FATAL_ERROR "Quarisma requires HIP 5.0 or above. Found: ${hip_VERSION}")
endif()

message(STATUS "Quarisma: HIP detected: ${hip_VERSION}")
message(STATUS "Quarisma: HIP compiler is: ${CMAKE_HIP_COMPILER}")
message(STATUS "Quarisma: HIP toolkit directory: ${HIP_ROOT_DIR}")

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
           gfx803 # Fiji (R9 Fury, R9 Nano)
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
elseif(PROJECT_HIP_ARCH_OPTIONS STREQUAL "gfx803")
  set(CMAKE_HIP_ARCHITECTURES "gfx803")
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
  set(CMAKE_HIP_ARCHITECTURES "gfx803;gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100")
elseif(PROJECT_HIP_ARCH_OPTIONS STREQUAL "none")
  # Don't set any architectures, let parent project handle it
endif()

# HIP Allocation Strategy Configuration (uses same flag as CUDA)
message(STATUS "HIP allocation strategy: ${PROJECT_GPU_ALLOC}")

# Set preprocessor definitions based on allocation strategy
if(PROJECT_GPU_ALLOC STREQUAL "SYNC")
  add_compile_definitions(PROJECT_HIP_ALLOC_SYNC)
  message(STATUS "Using synchronous HIP allocation (hipMalloc/hipFree)")
elseif(PROJECT_GPU_ALLOC STREQUAL "ASYNC")
  add_compile_definitions(PROJECT_HIP_ALLOC_ASYNC)
  message(STATUS "Using asynchronous HIP allocation (hipMallocAsync/hipFreeAsync)")
elseif(PROJECT_GPU_ALLOC STREQUAL "POOL_ASYNC")
  add_compile_definitions(PROJECT_HIP_ALLOC_POOL_ASYNC)
  message(STATUS "Using pool-based asynchronous HIP allocation (hipMallocFromPoolAsync)")
endif()

# Set up HIP libraries using modern imported targets
set(PROJECT_HIP_LIBRARIES hip::host hip::device)

# Add HIP libraries to the dependency list
list(APPEND PROJECT_DEPENDENCY_LIBS ${PROJECT_HIP_LIBRARIES})

# Add include directories
include_directories(SYSTEM "${HIP_INCLUDE_DIRS}")

# Add common HIP flags
string(APPEND CMAKE_HIP_FLAGS " --expt-extended-lambda")

if(NOT MSVC)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_HIP_FLAGS " -g")
  else()
    string(APPEND CMAKE_HIP_FLAGS " -O3")
  endif()
endif()

# For backward compatibility, set legacy variables (if needed elsewhere)
set(PROJECT_HIP_FOUND TRUE)
set(MEMORY_ENABLE_HIP ON)

# Enable GPU compilation for HIP
add_compile_definitions(MEMORY_ENABLE_GPU)
add_compile_definitions(MEMORY_ENABLE_HIP)
