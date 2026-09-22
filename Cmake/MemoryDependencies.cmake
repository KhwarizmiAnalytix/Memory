# =============================================================================
# Memory — test dependency resolution (standalone + embedded)
# =============================================================================
# Resolves Google Test and Google Benchmark without requiring an XSigma tree.
# fmt / mimalloc / Logging / Profiler are resolved by this repo's top-level
# CMakeLists.txt instead (mirrors LoggingDependencies.cmake's split).
# =============================================================================

include_guard(GLOBAL)

include(third_party_helpers)

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
    add_subdirectory(
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
  _create_third_party_interface_targets(
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
  elseif(XSIGMA_ENABLE_EXTERNAL)
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
    add_subdirectory(
      "${_bench_src}" "${CMAKE_BINARY_DIR}/ThirdParty/benchmark_build" EXCLUDE_FROM_ALL
    )
  else()
    include(FetchContent)
    FetchContent_Declare(
      benchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG v1.9.4
    )
    FetchContent_MakeAvailable(benchmark)
  endif()
  _set_third_party_folder_properties("benchmark" "${CMAKE_BINARY_DIR}/ThirdParty/benchmark_build")
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
  if(EXISTS "${MEMORY_THIRD_PARTY_DIR}/Logging/CMakeLists.txt")
    add_subdirectory(
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
# memory_setup_profiler: optional sibling dependency
# (https://github.com/KhwarizmiAnalytix/Profiler), submodule-or-FetchContent.
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
  set(PROFILER_ENABLE_TESTING OFF CACHE BOOL "Profiler tests build only standalone" FORCE)
  set(PROFILER_ENABLE_EXAMPLES OFF CACHE BOOL "Profiler examples build only standalone" FORCE)
  if(EXISTS "${MEMORY_THIRD_PARTY_DIR}/Profiler/CMakeLists.txt")
    add_subdirectory(
      "${MEMORY_THIRD_PARTY_DIR}/Profiler" "${CMAKE_BINARY_DIR}/ThirdParty/Profiler_build"
      EXCLUDE_FROM_ALL
    )
  else()
    include(FetchContent)
    FetchContent_Declare(
      Profiler GIT_REPOSITORY https://github.com/KhwarizmiAnalytix/Profiler.git GIT_TAG main
    )
    FetchContent_MakeAvailable(Profiler)
  endif()
  # Profiler's xplane_utils.cpp / bespoke/common/util.cpp / memory_tracker.cpp use
  # std::back_inserter without including <iterator>, relying on a transitive include
  # that newer libc++ (Homebrew LLVM, used by the macOS coverage job) no longer
  # provides. Force-include it until the submodule includes it itself.
  if(TARGET Profiler AND NOT MSVC)
    target_compile_options(Profiler PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include iterator>")
  endif()
  # Optional: do not FATAL_ERROR if unavailable, matching MEMORY_ENABLE_PROFILER's
  # "link when available" semantics in the host overlay (ThirdParty/memory.cmake).
endfunction()
