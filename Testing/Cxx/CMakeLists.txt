# =============================================================================
# Memory Library C++ Tests
# =============================================================================

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

target_compile_definitions(
  MemoryCxxTests PRIVATE MEMORY_HAS_GTEST=1 MEMORY_GOOGLE_TEST QUARISMA_HAS_GTEST=1
                         QUARISMA_GOOGLE_TEST USE_GTEST _VARIADIC_MAX=10
)

target_include_directories(
  MemoryCxxTests PRIVATE "${CMAKE_SOURCE_DIR}/Library/Core/Testing" "${CMAKE_SOURCE_DIR}/Library/Core"
)

target_link_libraries(
  MemoryCxxTests PRIVATE Memory::gtest Memory::gtest_main Memory::Memory Memory::build
)

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
  file(GLOB_RECURSE bench_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/Benchmark*.cpp")

  foreach(_bench_source IN ITEMS ${bench_sources})
    get_filename_component(bench_name "${_bench_source}" NAME_WE)
    string(REPLACE "Benchmark" "" bench_suffix "${bench_name}")
    string(TOLOWER "${bench_suffix}" bench_suffix_lower)
    set(target_name "benchmark_memory_${bench_suffix_lower}")

    add_executable(${target_name} "${_bench_source}")
    target_include_directories(
      ${target_name} PRIVATE "${CMAKE_SOURCE_DIR}/Library/Core"
    )
    target_link_libraries(
      ${target_name} PRIVATE Memory::benchmark Memory::benchmark_main Memory::Memory Memory::build
    )
    if(COMMAND memory_set_target_output_directories)
      memory_set_target_output_directories(${target_name})
    endif()
  endforeach()
endif()
