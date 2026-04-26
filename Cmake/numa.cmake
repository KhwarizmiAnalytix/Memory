# =============================================================================
# Quarisma NUMA
# (Non-Uniform Memory Access) Configuration Module

# This module configures NUMA support for multi-socket systems. It enables memory-aware thread
# scheduling and allocation on NUMA architectures. It is activated when MEMORY_ENABLE_NUMA=ON and is
# owned by the Memory module.

cmake_minimum_required(VERSION 3.16)

include_guard(GLOBAL)

# NUMA Support Flag Controls whether NUMA (Non-Uniform Memory Access) support is enabled. When
# enabled on Unix systems, provides memory-aware scheduling for multi-socket systems. Automatically
# disabled on non-Unix platforms (Windows, macOS).
option(MEMORY_ENABLE_NUMA "Enable numa node" OFF)
mark_as_advanced(MEMORY_ENABLE_NUMA)

if(NOT MEMORY_ENABLE_NUMA)
  message(STATUS "NUMA support is disabled (MEMORY_ENABLE_NUMA=OFF)")
  return()
endif()

if(NOT UNIX)
  message(STATUS "NUMA support is not available on this platform - disabling MEMORY_ENABLE_NUMA")
  set(MEMORY_ENABLE_NUMA OFF CACHE BOOL "Enable numa node" FORCE)
  return()
endif()

find_package(Numa)
if(NOT NUMA_FOUND)
  message(WARNING "NUMA library not found. Suppress this warning with -DMEMORY_ENABLE_NUMA=OFF")
  set(MEMORY_ENABLE_NUMA OFF CACHE BOOL "Enable numa node" FORCE)
  return()
endif()

if(NOT TARGET Numa::numa)
  add_library(Numa::numa INTERFACE IMPORTED)
  target_include_directories(Numa::numa INTERFACE ${Numa_INCLUDE_DIR})
  target_link_libraries(Numa::numa INTERFACE ${Numa_LIBRARIES})
endif()

message(STATUS "NUMA support enabled")
