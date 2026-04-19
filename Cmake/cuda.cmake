# ============================================================================= Quarisma CUDA
# Configuration Module

# This module handles CUDA compilation support for GPU acceleration. It manages CUDA toolkit
# detection, architecture configuration, and allocation strategy selection for GPU memory
# management.

# Include guard to prevent multiple inclusions
include_guard(GLOBAL)

# CUDA Support Flag Controls whether CUDA GPU acceleration is enabled for the build. When enabled,
# requires CUDA 12.0+ and configures GPU compilation.
option(MEMORY_ENABLE_CUDA "Enable CUDA compilation" OFF)
mark_as_advanced(MEMORY_ENABLE_CUDA)

# CUDA is not supported with the MinGW/MSYS2 toolchain in this project. Keep configuration simple
# and deterministic: force-disable and skip all CUDA setup.
if(WIN32 AND (MINGW OR CMAKE_CXX_COMPILER MATCHES "msys64"))
  if(MEMORY_ENABLE_CUDA)
    message(
      STATUS "CUDA: disabled on Windows+MinGW/MSYS2 toolchains (forcing MEMORY_ENABLE_CUDA=OFF)"
    )
  endif()
  set(MEMORY_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA compilation" FORCE)
  return()
endif()

if(NOT MEMORY_ENABLE_CUDA)
  return()
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CUDA_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "CUDA compiler (Clang)" FORCE)
  message(STATUS "CUDA: using Clang ${CMAKE_CXX_COMPILER_VERSION} as CUDA compiler")
endif()

find_package(CUDAToolkit REQUIRED)

# Workaround: When Clang is used as the CUDA compiler, CMake's internal variables
# _CMAKE_CUDA_WHOLE_FLAG and CMAKE_CUDA_COMPILE_OBJECT are not reliably propagated to the
# generator's directory scope — this manifests as a hard error during the generation phase on CMake
# 3.28 and CMake 4.2+. Setting CMAKE_CUDA_COMPILER_FORCED skips the compiler test subprocess that
# triggers the broken variable lookup; Clang + CUDA 12 is known-good.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CUDA_COMPILER_FORCED TRUE)
endif()

# Enable CUDA language support
enable_language(CUDA)

# Persist CUDA language rules to the CMake cache so every directory scope can access them during
# generation.
#
# Root cause: enable_language(CUDA) is called from Library/Memory (a subdirectory), so
# CMakeCUDAInformation.cmake and Compiler/Clang-CUDA.cmake are loaded only in Memory's
# configure-phase directory scope.  The Ninja generator later processes Library/Core/Testing/Cxx
# (which contains CudaEnzymeADTest.cu) in a separate directory scope that never ran
# CMakeCUDAInformation.cmake, so it cannot find these variables. Storing them as CACHE INTERNAL
# makes them globally visible across all directory scopes without overriding any per-directory
# regular variable.
# Persist CUDA language rules to the CMake cache (FORCE) so every directory scope can read them
# during the generation phase.  enable_language(CUDA) sets these as regular variables that are only
# visible in the Memory subdirectory scope; without CACHE INTERNAL they are invisible to sibling
# directories (e.g. Core/Testing/Cxx) that compile .cu files but never called enable_language.
# Strategy: use whatever value enable_language just set, falling back to the Clang default when the
# variable was not populated (CMAKE_CUDA_COMPILER_FORCED skips parts of the compiler test).
if(NOT _CMAKE_CUDA_WHOLE_FLAG)
  # Clang uses plain -c; no whole-archive wrapper is needed.
  set(_CMAKE_CUDA_WHOLE_FLAG "-c")
endif()
set(_CMAKE_CUDA_WHOLE_FLAG "${_CMAKE_CUDA_WHOLE_FLAG}" CACHE INTERNAL "CUDA whole-object flag" FORCE)

if(NOT CMAKE_CUDA_COMPILE_OBJECT)
  # Mirrors the template in CMakeCUDAInformation.cmake for Clang. _CMAKE_COMPILE_AS_CUDA_FLAG = "-x
  # cuda" (from Clang-CUDA.cmake)
  set(CMAKE_CUDA_COMPILE_OBJECT
      "<CMAKE_CUDA_COMPILER>  <DEFINES> <INCLUDES> <FLAGS> -x cuda <CUDA_COMPILE_MODE> <SOURCE> -o <OBJECT>"
  )
endif()
set(CMAKE_CUDA_COMPILE_OBJECT "${CMAKE_CUDA_COMPILE_OBJECT}" CACHE INTERNAL "CUDA compile-object rule" FORCE)

# Version checks using consistent CUDAToolkit variables
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
  message(FATAL_ERROR "Quarisma requires CUDA 12.0 or above. Found: ${CUDAToolkit_VERSION}")
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.2")
  message(FATAL_ERROR "QUARISMA CUDA support requires compiler version 9.2+")
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

message(STATUS "Quarisma: CUDA detected: ${CUDAToolkit_VERSION}")
message(STATUS "Quarisma: CUDA compiler is: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Quarisma: CUDA toolkit directory: ${CUDAToolkit_ROOT}")

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
  # Clang as CUDA compiler: match the host C++ standard
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  if(NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
  endif()
  message(STATUS "CUDA: Clang will use C++${CMAKE_CUDA_STANDARD} standard")
endif()

set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Use modern CMake CUDA architecture handling
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# GPU Architecture options
set(PROJECT_CUDA_ARCH_OPTIONS "native" CACHE STRING "Which GPU Architecture(s) to compile for")
set_property(
  CACHE PROJECT_CUDA_ARCH_OPTIONS
  PROPERTY STRINGS
           native
           fermi
           kepler
           maxwell
           pascal
           volta
           turing
           ampere
           ada
           hopper
           all
           none
)

# Set architectures based on user selection
if(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "native")
  # "native" requires CMake to probe the GPU at configure time.  When CMAKE_CUDA_COMPILER_FORCED is
  # set (Clang + CMake 4.2 workaround) that probe is skipped, so CMake cannot resolve "native"
  # during generation. Fall back to "all-major" which compiles for every major CUDA SM without
  # needing a GPU present at configure time.
  if(CMAKE_CUDA_COMPILER_FORCED)
    # CUDA 13.0 removed pre-Pascal (sm_52-) libdevice; "all-major" still expands to sm_52 in CMake
    # 4.x, so use an explicit Turing-and-above list.
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    message(
      STATUS "CUDA: native arch detection unavailable (COMPILER_FORCED), using 75;80;86;89;90"
    )
  else()
    set(CMAKE_CUDA_ARCHITECTURES "native")
  endif()
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "fermi")
  set(CMAKE_CUDA_ARCHITECTURES "20")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "kepler")
  set(CMAKE_CUDA_ARCHITECTURES "30;35")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "maxwell")
  set(CMAKE_CUDA_ARCHITECTURES "50")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "pascal")
  set(CMAKE_CUDA_ARCHITECTURES "60;61")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "volta")
  set(CMAKE_CUDA_ARCHITECTURES "70")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "turing")
  set(CMAKE_CUDA_ARCHITECTURES "75")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "ampere")
  set(CMAKE_CUDA_ARCHITECTURES "80;86")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "ada")
  set(CMAKE_CUDA_ARCHITECTURES "89;90")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "hopper")
  set(CMAKE_CUDA_ARCHITECTURES "90")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "all")
  set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;86;89;90")
elseif(PROJECT_CUDA_ARCH_OPTIONS STREQUAL "none")
  # Don't set any architectures, let parent project handle it
endif()

# CUDA Allocation Strategy Configuration (defined in main CMakeLists.txt)
message(STATUS "CUDA allocation strategy: ${PROJECT_GPU_ALLOC}")

# Set preprocessor definitions based on allocation strategy
if(PROJECT_GPU_ALLOC STREQUAL "SYNC")
  add_compile_definitions(PROJECT_CUDA_ALLOC_SYNC)
  message(STATUS "Using synchronous CUDA allocation (cuMemAlloc/cuMemFree)")
elseif(PROJECT_GPU_ALLOC STREQUAL "ASYNC")
  add_compile_definitions(PROJECT_CUDA_ALLOC_ASYNC)
  message(STATUS "Using asynchronous CUDA allocation (cuMemAllocAsync/cuMemFreeAsync)")
elseif(PROJECT_GPU_ALLOC STREQUAL "POOL_ASYNC")
  add_compile_definitions(PROJECT_CUDA_ALLOC_POOL_ASYNC)
  message(STATUS "Using pool-based asynchronous CUDA allocation (cuMemAllocFromPoolAsync)")
endif()

# Set up CUDA libraries using modern imported targets
set(PROJECT_CUDA_LIBRARIES CUDA::cudart CUDA::cuda_driver CUDA::cusparse CUDA::curand CUDA::cublas)

# Add nvperf_host if available (required by Kineto for CUPTI range profiler)
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
  string(APPEND CMAKE_CUDA_FLAGS " -Wno-unknown-cuda-version")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CUDA_FLAGS " -g")
  else()
    string(APPEND CMAKE_CUDA_FLAGS " -O3")
  endif()
endif()

# For backward compatibility, set legacy variables (if needed elsewhere)
set(PROJECT_CUDA_FOUND TRUE)
