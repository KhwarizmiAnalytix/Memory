# =============================================================================
# Quarisma Intel TBB -
# Memory Allocator Backend (Threading Building Blocks) Scalable Memory Allocation

# This module configures Intel TBB's scalable memory allocator (tbbmalloc). It is activated when
# MEMORY_ENABLE_TBB=ON.
#
# If tbb_multithreading.cmake has already acquired TBB (PARALLEL_ENABLE_TBB=ON), the TBB::tbbmalloc
# target is available and this module just wraps it. Otherwise it acquires TBB from source itself
# before creating Tbb::tbbmalloc.

cmake_minimum_required(VERSION 3.16)

include_guard(GLOBAL)

option(MEMORY_ENABLE_TBB "Enable Intel TBB scalable memory allocator (tbbmalloc) support." OFF)
mark_as_advanced(MEMORY_ENABLE_TBB)

# Gate: only proceed if TBB memory allocator is requested
if(NOT MEMORY_ENABLE_TBB)
  message(STATUS "Intel TBB memory allocator is disabled (MEMORY_ENABLE_TBB=OFF)")
  return()
endif()

message(STATUS "Configuring Intel TBB memory allocator support...")

# =============================================================================
# Step 1: Acquire TBB
# if not already done by tbb_multithreading.cmake

if(NOT TARGET TBB::tbb)
  # TBB has not been acquired yet — acquire it now. (tbb_multithreading.cmake was skipped because
  # PARALLEL_ENABLE_TBB=OFF)

  # Reuse the same configuration options if already defined, otherwise declare them
  if(NOT DEFINED PROJECT_TBB_FORCE_BUILD_FROM_SOURCE)
    option(PROJECT_TBB_FORCE_BUILD_FROM_SOURCE
           "Force building TBB from source instead of using system TBB" OFF
    )
    mark_as_advanced(PROJECT_TBB_FORCE_BUILD_FROM_SOURCE)
  endif()

  if(NOT DEFINED PROJECT_TBB_VERSION)
    set(PROJECT_TBB_VERSION "v2021.13.0" CACHE STRING "TBB version to build from source")
    mark_as_advanced(PROJECT_TBB_VERSION)
  endif()

  if(NOT DEFINED PROJECT_TBB_REPOSITORY)
    set(PROJECT_TBB_REPOSITORY "https://github.com/oneapi-src/oneTBB.git"
        CACHE STRING "TBB repository URL"
    )
    mark_as_advanced(PROJECT_TBB_REPOSITORY)
  endif()

  set(_tbb_memory_acquired_tbb FALSE)

  if(NOT PROJECT_TBB_FORCE_BUILD_FROM_SOURCE)
    message(STATUS "Searching for system-installed Intel TBB (for memory allocator)...")

    find_package(TBB QUIET)

    if(TBB_FOUND AND TARGET TBB::tbb)
      message(STATUS "✅ Found system-installed Intel TBB")
      if(TARGET TBB::tbbmalloc)
        message(STATUS "   TBB::tbbmalloc target available")
      endif()
      set(_tbb_memory_acquired_tbb TRUE)
    else()
      message(STATUS "❌ System-installed Intel TBB not found")

      if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(
          FATAL_ERROR
            "\n"
            "================================================================================\n"
            "ERROR: Intel TBB not found - System installation required on Windows with Clang\n"
            "================================================================================\n"
            "\n"
            "Building TBB from source is not supported on Windows with Clang due to\n"
            "compiler compatibility issues. You must install TBB using a package manager.\n"
            "\n"
            "RECOMMENDED INSTALLATION METHODS:\n"
            "\n"
            "1. Using vcpkg (recommended):\n"
            "   vcpkg install tbb:x64-windows\n"
            "   \n"
            "   Then configure with:\n"
            "   -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake\n"
            "\n"
            "2. Using Chocolatey:\n"
            "   choco install tbb\n"
            "\n"
            "3. Manual installation:\n"
            "   - Download from: https://github.com/oneapi-src/oneTBB/releases\n"
            "   - Extract and set TBB_ROOT environment variable to installation path\n"
            "\n"
            "For more information, see: https://github.com/oneapi-src/oneTBB\n"
            "================================================================================\n"
        )
      endif()

      message(STATUS "   Will build TBB from source as fallback")
    endif()
  endif()

  if(NOT _tbb_memory_acquired_tbb)
    message(STATUS "Building Intel TBB from source (for memory allocator)...")
    message(STATUS "   Repository: ${PROJECT_TBB_REPOSITORY}")
    message(STATUS "   Version: ${PROJECT_TBB_VERSION}")

    include(FetchContent)

    FetchContent_Declare(
      oneTBB
      GIT_REPOSITORY ${PROJECT_TBB_REPOSITORY}
      GIT_TAG ${PROJECT_TBB_VERSION}
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE
    )

    set(TBB_TEST OFF CACHE BOOL "Disable TBB tests" FORCE)
    set(TBB_EXAMPLES OFF CACHE BOOL "Disable TBB examples" FORCE)
    set(TBB_STRICT OFF CACHE BOOL "Disable strict mode for compatibility" FORCE)

    if(POLICY CMP0126)
      cmake_policy(SET CMP0126 NEW)
    endif()

    if(WIN32)
      set(TBB_WINDOWS_DRIVER OFF CACHE BOOL "Build TBB for Windows kernel mode" FORCE)
      set(CMAKE_POSITION_INDEPENDENT_CODE OFF CACHE BOOL "Disable PIC on Windows" FORCE)
      set(TBB_ENABLE_PIC OFF CACHE BOOL "Disable TBB PIC on Windows" FORCE)
      set(CMAKE_CXX_COMPILE_OPTIONS_PIC "" CACHE STRING "Clear PIC options" FORCE)
      set(CMAKE_C_COMPILE_OPTIONS_PIC "" CACHE STRING "Clear PIC options" FORCE)
    endif()

    message(STATUS "Downloading Intel TBB source code...")

    if(WIN32)
      set(_SAVED_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
      set(_SAVED_CMAKE_CXX_COMPILE_OPTIONS_PIC "${CMAKE_CXX_COMPILE_OPTIONS_PIC}")
      set(_SAVED_CMAKE_C_COMPILE_OPTIONS_PIC "${CMAKE_C_COMPILE_OPTIONS_PIC}")

      set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
      set(CMAKE_CXX_COMPILE_OPTIONS_PIC "")
      set(CMAKE_C_COMPILE_OPTIONS_PIC "")
    endif()

    FetchContent_MakeAvailable(oneTBB)

    if(WIN32)
      set(CMAKE_POSITION_INDEPENDENT_CODE ${_SAVED_CMAKE_POSITION_INDEPENDENT_CODE})
      set(CMAKE_CXX_COMPILE_OPTIONS_PIC "${_SAVED_CMAKE_CXX_COMPILE_OPTIONS_PIC}")
      set(CMAKE_C_COMPILE_OPTIONS_PIC "${_SAVED_CMAKE_C_COMPILE_OPTIONS_PIC}")
    endif()

    FetchContent_GetProperties(oneTBB)
    if(NOT onetbb_POPULATED)
      message(FATAL_ERROR "Failed to download Intel TBB from ${PROJECT_TBB_REPOSITORY}")
    endif()

    message(STATUS "✅ Successfully downloaded Intel TBB source")
    message(STATUS "   Source directory: ${onetbb_SOURCE_DIR}")
    message(STATUS "   Binary directory: ${onetbb_BINARY_DIR}")

    if(NOT TARGET TBB::tbb)
      message(FATAL_ERROR "TBB::tbb target was not created after building from source")
    endif()

    set(TBB_FROM_SOURCE TRUE)
    set(TBB_FOUND TRUE CACHE BOOL "TBB was found or built successfully" FORCE)
    set(TBB_FROM_SOURCE TRUE CACHE BOOL "TBB was built from source" FORCE)
    set(TBB_SOURCE_DIR ${onetbb_SOURCE_DIR} CACHE PATH "TBB source directory" FORCE)
    set(TBB_BINARY_DIR ${onetbb_BINARY_DIR} CACHE PATH "TBB binary directory" FORCE)

    message(STATUS "✅ Successfully built Intel TBB from source")
  endif()
endif()

# =============================================================================
# Step 2: Verify
# tbbmalloc target is available

if(NOT TARGET TBB::tbbmalloc)
  message(
    WARNING "TBB::tbbmalloc target is not available - memory allocator support may be limited"
  )
  return()
endif()

# =============================================================================
# Step 3: Create
# Tbb::tbbmalloc interface target

if(NOT TARGET Tbb::tbbmalloc)
  add_library(Tbb::tbbmalloc INTERFACE IMPORTED)
  target_link_libraries(Tbb::tbbmalloc INTERFACE TBB::tbbmalloc)
endif()

message(STATUS "   TBB::tbbmalloc target available")

# =============================================================================
# Step 4: Configure
# output directories for the tbbmalloc target

if(TBB_FROM_SOURCE AND TARGET tbbmalloc)
  foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${config} config_upper)
    set_target_properties(
      tbbmalloc
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config_upper}
                 "${PROJECT_BINARY_DIR}/bin/${config_upper}"
                 ARCHIVE_OUTPUT_DIRECTORY_${config_upper}
                 "${PROJECT_BINARY_DIR}/lib/${config_upper}"
                 LIBRARY_OUTPUT_DIRECTORY_${config_upper}
                 "${PROJECT_BINARY_DIR}/lib/${config_upper}"
    )
  endforeach()

  set_target_properties(
    tbbmalloc
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
               ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
               LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
               FOLDER "ThirdParty/tbb"
  )
  message(STATUS "Configured output directories for TBB target 'tbbmalloc'")
endif()

message(STATUS "Intel TBB memory allocator configuration complete")
