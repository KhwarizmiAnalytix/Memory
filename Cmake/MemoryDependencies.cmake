# Configure vendored dependencies without changing the host's build options.
function(memory_add_dependency_subdirectory source_dir binary_dir)
  foreach(_option IN ITEMS BUILD_TESTING BUILD_EXAMPLES BUILD_BENCHMARKS)
    get_property(
      _had_${_option}
      CACHE ${_option}
      PROPERTY TYPE
      SET
    )
    if(_had_${_option})
      get_property(_value_${_option} CACHE ${_option} PROPERTY VALUE)
      get_property(_type_${_option} CACHE ${_option} PROPERTY TYPE)
      get_property(_help_${_option} CACHE ${_option} PROPERTY HELPSTRING)
    endif()
    set(${_option} OFF)
  endforeach()
  add_subdirectory("${source_dir}" "${binary_dir}" ${ARGN})
  foreach(_option IN ITEMS BUILD_TESTING BUILD_EXAMPLES BUILD_BENCHMARKS)
    if(_had_${_option})
      set(${_option} "${_value_${_option}}" CACHE ${_type_${_option}} "${_help_${_option}}" FORCE)
    else()
      unset(${_option} CACHE)
    endif()
  endforeach()
endfunction()

# =============================================================================
# Memory — test dependency resolution (standalone + embedded)
# =============================================================================
# Resolves Google Test and Google Benchmark without requiring an XSigma tree.
# fmt / mimalloc / Logging / Profiler are resolved by this repo's top-level
# CMakeLists.txt instead (mirrors LoggingDependencies.cmake's split).
# =============================================================================

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/third_party_helpers.cmake")

get_filename_component(_memory_repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(MEMORY_THIRD_PARTY_DIR "${_memory_repo_root}/ThirdParty"
    CACHE PATH "Root of Memory's bundled third-party sources"
)

# -----------------------------------------------------------------------------
# memory_setup_gtest: Google Test for the test suite
# -----------------------------------------------------------------------------
function(memory_setup_gtest)
  if(TARGET gtest_main OR TARGET GTest::gtest_main)
    # Fall through to alias normalization below.
  elseif(COMMAND xsigma_add_googletest)
    # Embedded in XSigma: reuse the host's googletest wiring.
    xsigma_add_googletest()
  elseif(EXISTS "${MEMORY_THIRD_PARTY_DIR}/googletest/CMakeLists.txt")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    memory_add_dependency_subdirectory(
      "${MEMORY_THIRD_PARTY_DIR}/googletest" "${CMAKE_BINARY_DIR}/ThirdParty/googletest_build"
      EXCLUDE_FROM_ALL
    )
  else()
    include(FetchContent)
    FetchContent_Declare(
      googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.18.0
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  if(TARGET gtest AND NOT TARGET GTest::gtest)
    add_library(GTest::gtest ALIAS gtest)
  endif()
  if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main ALIAS gtest_main)
  endif()
  memory_create_third_party_interface_targets(
    "Gtest::gtest=GTest::gtest|gtest" "Gtest::gtest_main=GTest::gtest_main|gtest_main"
  )
endfunction()

# -----------------------------------------------------------------------------
# memory_setup_benchmark: Google Benchmark for micro-benchmarks
# -----------------------------------------------------------------------------
function(memory_setup_benchmark)
  if(TARGET benchmark OR TARGET benchmark::benchmark)
    return()
  endif()
  set(_bench_src "")
  if(EXISTS "${MEMORY_THIRD_PARTY_DIR}/benchmark/CMakeLists.txt")
    set(_bench_src "${MEMORY_THIRD_PARTY_DIR}/benchmark")
  elseif(MEMORY_ENABLE_EXTERNAL)
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
      message(STATUS "Found external Google Benchmark")
      return()
    endif()
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-c2y-extensions")
  endif()
  if(NOT DEFINED HAVE_STD_REGEX)
    set(HAVE_STD_REGEX 1 CACHE INTERNAL "")
  endif()
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
  set(BENCHMARK_ENABLE_EXCEPTIONS ON CACHE BOOL "Enable benchmark exceptions" FORCE)
  set(BENCHMARK_ENABLE_LTO OFF CACHE BOOL "Disable benchmark LTO" FORCE)
  set(BENCHMARK_USE_LIBCXX OFF CACHE BOOL "Disable benchmark libcxx" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "Disable benchmark werror" FORCE)
  set(BENCHMARK_FORCE_WERROR OFF CACHE BOOL "Disable benchmark force werror" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install" FORCE)
  set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "Disable benchmark docs install" FORCE)
  set(BENCHMARK_ENABLE_DOXYGEN OFF CACHE BOOL "Disable benchmark doxygen" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable benchmark gtest tests" FORCE)
  set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "Don't use bundled gtest for benchmark" FORCE)
  set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "Don't download dependencies" FORCE)
  if(_bench_src)
    memory_add_dependency_subdirectory(
      "${_bench_src}" "${CMAKE_BINARY_DIR}/ThirdParty/benchmark_build" EXCLUDE_FROM_ALL
    )
  else()
    include(FetchContent)
    FetchContent_Declare(
      benchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG v1.9.5
    )
    FetchContent_MakeAvailable(benchmark)
  endif()
  memory_set_third_party_folder_properties(
    "benchmark" "${CMAKE_BINARY_DIR}/ThirdParty/benchmark_build"
  )
endfunction()

# -----------------------------------------------------------------------------
# memory_setup_logging: required sibling dependency
# (https://github.com/KhwarizmiAnalytix/Logging), submodule-or-FetchContent —
# same pattern ThirdParty/Profiler's own standalone CMakeLists.txt uses for
# its fmt/kineto/ittapi deps (profiler_fetch_git in ProfilerDependencies.cmake).
# -----------------------------------------------------------------------------
function(memory_setup_logging)
  if(TARGET Logging::Logging)
    return()
  endif()
  if(COMMAND xsigma_add_logging)
    # Embedded in XSigma: reuse the host's Logging host-overlay wiring.
    xsigma_add_logging()
    return()
  endif()
  set(LOGGING_ENABLE_TESTING OFF CACHE BOOL "Logging tests build only standalone" FORCE)
  set(LOGGING_ENABLE_EXAMPLES OFF CACHE BOOL "Logging examples build only standalone" FORCE)
  # Logging consumes fmt headers; its optional module target is unnecessary and
  # clang-cl cannot scan its imports with CMake's Ninja generator.
  set(FMT_MODULE OFF)
  # Forward tool flags so Logging inherits Memory's build settings.
  foreach(_mem_log_feat IN ITEMS CACHE ICECC CLANGTIDY FIX IWYU SANITIZER COVERAGE SPELL VALGRIND)
    set(LOGGING_ENABLE_${_mem_log_feat} "${MEMORY_ENABLE_${_mem_log_feat}}"
        CACHE BOOL "Configured by Memory" FORCE)
  endforeach()
  foreach(_mem_log_setting IN ITEMS CACHE_BACKEND SANITIZER_TYPE LINKER_CHOICE LTO_MODE)
    set(LOGGING_${_mem_log_setting} "${MEMORY_${_mem_log_setting}}"
        CACHE STRING "Configured by Memory" FORCE)
  endforeach()
  if(EXISTS "${MEMORY_THIRD_PARTY_DIR}/Logging/CMakeLists.txt")
    memory_add_dependency_subdirectory(
      "${MEMORY_THIRD_PARTY_DIR}/Logging" "${CMAKE_BINARY_DIR}/ThirdParty/Logging_build"
      EXCLUDE_FROM_ALL
    )
  else()
    include(FetchContent)
    FetchContent_Declare(
      Logging GIT_REPOSITORY https://github.com/KhwarizmiAnalytix/Logging.git GIT_TAG main
    )
    FetchContent_MakeAvailable(Logging)
  endif()
  if(NOT TARGET Logging::Logging)
    message(
      FATAL_ERROR
        "Memory requires Logging::Logging but it could not be resolved "
        "(git submodule update --init ThirdParty/Logging, or network " "access for FetchContent)"
    )
  endif()
endfunction()

# -----------------------------------------------------------------------------
# memory_setup_profiler: optional sibling XSigma repo
# (https://github.com/KhwarizmiAnalytix/Profiler). Not a vendored third-party
# library — resolved exclusively via find_package(Profiler), independent of
# MEMORY_ENABLE_EXTERNAL (that flag governs vendored ThirdParty/ copies).
# Install it first (cmake --install) and point CMAKE_PREFIX_PATH at it.
# Optional: callers gate on TARGET Profiler::Profiler, so a missing install
# just disables MEMORY_HAS_PROFILER rather than failing the configure.
# -----------------------------------------------------------------------------
function(memory_setup_profiler)
  if(TARGET Profiler::Profiler)
    return()
  endif()
  if(COMMAND xsigma_add_profiler)
    xsigma_add_profiler()
    return()
  endif()
  if(NOT MEMORY_ENABLE_PROFILER)
    return()
  endif()
  find_package(Profiler CONFIG QUIET)
  if(TARGET Profiler::Profiler)
    message(STATUS "Memory: using Profiler::Profiler (find_package)")
  else()
    message(STATUS "Memory: Profiler not found via find_package; MEMORY_HAS_PROFILER "
                   "will be 0. Install it and set CMAKE_PREFIX_PATH to enable it."
    )
  endif()
endfunction()
