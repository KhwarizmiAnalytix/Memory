# =============================================================================
# XSigma Metal
# Configuration Module

# This module configures Apple's Metal for macOS GPU acceleration. Metal has no CMake
# "language" the way CUDA/HIP do (no enable_language(Metal), no device-code compiler
# pass over ordinary C++ headers) — kernel source is embedded as a string and compiled
# at runtime via -[MTLDevice newLibraryWithSource:options:error:], so this module only
# needs to enable Objective-C++ (for the .mm allocator/dispatch implementation files)
# and link the Metal/Foundation frameworks.

# Include guard to prevent multiple inclusions
include_guard(GLOBAL)

if(NOT APPLE)
  message(FATAL_ERROR "MEMORY_GPU_BACKEND=metal requires an Apple platform (macOS).")
endif()

# Enable Objective-C++ so .mm sources (Metal/Foundation API calls) can be compiled.
enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD ${MEMORY_CXX_STANDARD})
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

set(PROJECT_METAL_LIBRARIES ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
# Appended to MEMORY_DEPENDENCY_LIBS (not PROJECT_DEPENDENCY_LIBS, which hip.cmake uses but
# which is never actually consumed by Memory/CMakeLists.txt's target_link_libraries call).
list(APPEND MEMORY_DEPENDENCY_LIBS ${PROJECT_METAL_LIBRARIES})

message(STATUS "XSigma: Metal framework: ${METAL_FRAMEWORK}")
message(STATUS "XSigma: Foundation framework: ${FOUNDATION_FRAMEWORK}")
