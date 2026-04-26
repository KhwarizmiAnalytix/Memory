# ============================================================================= Memory Library C++
# Tests
# =============================================================================

# --- GoogleTest setup ---
if(NOT TARGET gtest)
  set(_gtest_external_found FALSE)
  if(QUARISMA_ENABLE_EXTERNAL)
    find_package(GTest QUIET)
    if(GTest_FOUND)
      message(STATUS "Found external Google Test")
      _create_third_party_interface_targets("Gtest::gtest=GTest::gtest|gtest"
                                            "Gtest::gtest_main=GTest::gtest_main|gtest_main")
      set(_gtest_external_found TRUE)
    endif()
  endif()
  if(NOT _gtest_external_found)
    set(BUILD_GMOCK OFF CACHE BOOL "Disable Google Mock" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Disable Google Test install" FORCE)
    set(gtest_build_samples OFF CACHE BOOL "Disable Google Test samples" FORCE)
    set(gtest_build_tests OFF CACHE BOOL "Disable Google Test tests" FORCE)
    if(MSVC)
      set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(
      "${QUARISMA_THIRDPARTY_DIR}/googletest"
      "${CMAKE_BINARY_DIR}/ThirdParty/googletest_build"
      EXCLUDE_FROM_ALL
    )
    _set_third_party_folder_properties("googletest"
                                       "${CMAKE_BINARY_DIR}/ThirdParty/googletest_build")
  endif()
endif()
_create_third_party_interface_targets("Gtest::gtest=GTest::gtest|gtest"
                                      "Gtest::gtest_main=GTest::gtest_main|gtest_main")

file(GLOB_RECURSE test_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cpp")

set(TestFiles)
foreach(_test_source IN ITEMS ${test_sources})
  get_filename_component(file_name "${_test_source}" NAME)

  if(NOT MEMORY_ENABLE_CUDA AND NOT MEMORY_ENABLE_HIP)
    if(file_name MATCHES "TestGpu" OR file_name MATCHES "TestGPU")
      continue()
    endif()
  endif()

  if(NOT MEMORY_ENABLE_CUDA)
    if(file_name MATCHES "TestCuda")
      continue()
    endif()
  endif()

  if(NOT MEMORY_ENABLE_HIP)
    if(file_name MATCHES "TestHip")
      continue()
    endif()
  endif()

  list(APPEND TestFiles "${_test_source}")
endforeach()

add_executable(MemoryCxxTests ${TestFiles})

# Testing/ (sources + generated *Test.h) and third-party includes match the Memory::Memory interface.
target_include_directories(
  MemoryCxxTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/.."
)
if(MEMORY_DEPENDENCY_INCLUDE_DIRS)
  target_include_directories(MemoryCxxTests PRIVATE ${MEMORY_DEPENDENCY_INCLUDE_DIRS})
endif()

target_compile_definitions(
  MemoryCxxTests PRIVATE MEMORY_HAS_GTEST=1 MEMORY_GOOGLE_TEST QUARISMA_HAS_GTEST=1
                         QUARISMA_GOOGLE_TEST USE_GTEST _VARIADIC_MAX=10
)

target_link_libraries(
  MemoryCxxTests PRIVATE Gtest::gtest Gtest::gtest_main Memory::Memory
)
if(MEMORY_DEPENDENCY_LIBS)
  target_link_libraries(MemoryCxxTests PRIVATE ${MEMORY_DEPENDENCY_LIBS})
endif()
# Logging::Logging is in MEMORY_DEPENDENCY_LIBS when MEMORY_HAS_LOGGING=1; Memory::Memory also
# links it PUBLIC, so Logging headers and MEMORY_HAS_LOGGING propagate to this target.

set_target_properties(
  MemoryCxxTests PROPERTIES FOLDER "Tests" VS_DEBUGGER_WORKING_DIRECTORY
                                           "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

add_test(NAME MemoryCxxTests COMMAND MemoryCxxTests
         WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

set_tests_properties(MemoryCxxTests PROPERTIES TIMEOUT 120 LABELS "Memory;Unit")

if(COMMAND memory_set_target_output_directories)
  memory_set_target_output_directories(MemoryCxxTests)
endif()

if(MEMORY_ENABLE_BENCHMARK)
  # --- Google Benchmark setup ---
  if(NOT TARGET benchmark)
    set(_bench_external_found FALSE)
    if(QUARISMA_ENABLE_EXTERNAL)
      find_package(benchmark QUIET)
      if(benchmark_FOUND)
        message(STATUS "Found external Google Benchmark")
        _create_third_party_interface_targets(
          "Benchmark::benchmark=benchmark::benchmark|benchmark"
          "Benchmark::benchmark_main=benchmark::benchmark_main|benchmark_main"
        )
        set(_bench_external_found TRUE)
      endif()
    endif()
    if(NOT _bench_external_found)
      add_subdirectory(
        "${QUARISMA_THIRDPARTY_DIR}/benchmark"
        "${CMAKE_BINARY_DIR}/ThirdParty/benchmark_build"
        EXCLUDE_FROM_ALL
      )
      _set_third_party_folder_properties("benchmark"
                                         "${CMAKE_BINARY_DIR}/ThirdParty/benchmark_build")
    endif()
  endif()
  _create_third_party_interface_targets(
    "Benchmark::benchmark=benchmark::benchmark|benchmark"
    "Benchmark::benchmark_main=benchmark::benchmark_main|benchmark_main"
  )
  if(TARGET Benchmark::benchmark AND NOT TARGET Memory::benchmark)
    add_library(Memory::benchmark ALIAS benchmark)
    add_library(Memory::benchmark_main ALIAS benchmark_main)
  endif()

  file(GLOB_RECURSE bench_sources LIST_DIRECTORIES false
       "${CMAKE_CURRENT_SOURCE_DIR}/Benchmark*.cpp"
  )

  foreach(_bench_source IN ITEMS ${bench_sources})
    get_filename_component(bench_name "${_bench_source}" NAME_WE)
    string(REPLACE "Benchmark" "" bench_suffix "${bench_name}")
    string(TOLOWER "${bench_suffix}" bench_suffix_lower)
    set(target_name "benchmark_memory_${bench_suffix_lower}")

    add_executable(${target_name} "${_bench_source}")
    target_include_directories(
      ${target_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/.."
    )
    if(MEMORY_DEPENDENCY_INCLUDE_DIRS)
      target_include_directories(${target_name} PRIVATE ${MEMORY_DEPENDENCY_INCLUDE_DIRS})
    endif()
    target_link_libraries(
      ${target_name} PRIVATE Memory::benchmark Memory::benchmark_main Memory::Memory
    )
    if(MEMORY_DEPENDENCY_LIBS)
      target_link_libraries(${target_name} PRIVATE ${MEMORY_DEPENDENCY_LIBS})
    endif()
    # Same Logging visibility as MemoryCxxTests (via Memory::Memory + MEMORY_DEPENDENCY_LIBS).
    if(COMMAND memory_set_target_output_directories)
      memory_set_target_output_directories(${target_name})
    endif()
  endforeach()
endif()
