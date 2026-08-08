# =============================================================================
# Memory Library C++
# Tests
# =============================================================================
quarisma_add_googletest()

file(GLOB_RECURSE test_sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cpp")

set(TestFiles)
foreach(_test_source IN ITEMS ${test_sources})
  get_filename_component(file_name "${_test_source}" NAME)

  if(MEMORY_GPU_BACKEND STREQUAL "none")
    if(file_name MATCHES "TestGpu" OR file_name MATCHES "TestGPU")
      continue()
    endif()
  endif()

  if(NOT MEMORY_GPU_BACKEND STREQUAL "cuda")
    if(file_name MATCHES "TestCuda")
      continue()
    endif()
  endif()

  if(NOT MEMORY_GPU_BACKEND STREQUAL "hip")
    if(file_name MATCHES "TestHip")
      continue()
    endif()
  endif()

  if(NOT MEMORY_GPU_BACKEND STREQUAL "metal")
    if(file_name MATCHES "TestMetal")
      continue()
    endif()
  endif()

  list(APPEND TestFiles "${_test_source}")
endforeach()

add_executable(MemoryCxxTests ${TestFiles})

# Testing/ (sources + generated *Test.h) and third-party includes match the Memory::Memory
# interface.
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

target_link_libraries(MemoryCxxTests PRIVATE Gtest::gtest Gtest::gtest_main Memory::Memory)
if(MEMORY_DEPENDENCY_LIBS)
  target_link_libraries(MemoryCxxTests PRIVATE ${MEMORY_DEPENDENCY_LIBS})
endif()
# Logging::Logging is in MEMORY_DEPENDENCY_LIBS when MEMORY_HAS_LOGGING=1; Memory::Memory also links
# it PUBLIC, so Logging headers and MEMORY_HAS_LOGGING propagate to this target.

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
      add_subdirectory(
        "${QUARISMA_THIRDPARTY_DIR}/benchmark" "${CMAKE_BINARY_DIR}/ThirdParty/benchmark_build"
        EXCLUDE_FROM_ALL
      )
      _set_third_party_folder_properties(
        "benchmark" "${CMAKE_BINARY_DIR}/ThirdParty/benchmark_build"
      )
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

  # LibTorch (PyTorch C++ frontend) CPU-allocation comparison benchmark.
  # Opt-in via MEMORY_ENABLE_LIBTORCH; the default follows
  # VECTORIZATION_ENABLE_LIBTORCH so setup.py's 'torch' token drives both.
  # Requires a LibTorch install in CMAKE_PREFIX_PATH (a pip-installed torch
  # package works). The torch-dependent source is dropped from the glob when
  # LibTorch is unavailable instead of building an empty translation unit
  # guarded by #if MEMORY_HAS_LIBTORCH.
  if(DEFINED VECTORIZATION_ENABLE_LIBTORCH)
    set(_memory_libtorch_default ${VECTORIZATION_ENABLE_LIBTORCH})
  else()
    set(_memory_libtorch_default OFF)
  endif()
  option(MEMORY_ENABLE_LIBTORCH
         "Enable LibTorch (PyTorch C++) CPU allocation comparison benchmark"
         ${_memory_libtorch_default})
  unset(_memory_libtorch_default)

  set(_memory_torch_found OFF)
  if(MEMORY_ENABLE_LIBTORCH)
    set(_saved_module_path "${CMAKE_MODULE_PATH}")
    set(CMAKE_MODULE_PATH "")
    # Force MKL single-dynamic-library mode so Torch's MKL discovery emits only
    # -lmkl_rt instead of the full component set (ScaLAPACK/BLACS/CDFT), which
    # would pull in MPI symbols that may not be installed.
    set(MKL_LINK "sdl")
    find_package(Torch QUIET)
    unset(MKL_LINK)
    set(CMAKE_MODULE_PATH "${_saved_module_path}")
    unset(_saved_module_path)
    if(Torch_FOUND)
      set(_memory_torch_found ON)
      message(STATUS "LibTorch found: ${TORCH_INSTALL_PREFIX} (${Torch_VERSION})")
      # pip-installed MKL ships versioned .so.N files without the unversioned
      # .so development symlinks the linker needs for -lmkl_NAME resolution.
      if(TARGET caffe2::mkl)
        get_target_property(_mkl_link_dirs caffe2::mkl INTERFACE_LINK_DIRECTORIES)
      else()
        set(_mkl_link_dirs "")
      endif()
      foreach(_mkl_dir IN LISTS _mkl_link_dirs)
        file(GLOB _mkl_ver_libs "${_mkl_dir}/libmkl_*.so.[0-9]*")
        foreach(_vlib IN LISTS _mkl_ver_libs)
          string(REGEX REPLACE "\\.so\\.[0-9].*$" ".so" _unversioned "${_vlib}")
          if(NOT EXISTS "${_unversioned}")
            get_filename_component(_vlib_base "${_vlib}" NAME)
            get_filename_component(_unver_base "${_unversioned}" NAME)
            execute_process(
              COMMAND ${CMAKE_COMMAND} -E create_symlink "${_vlib_base}" "${_unversioned}"
              RESULT_VARIABLE _sym_rc
            )
            if(_sym_rc EQUAL 0)
              message(STATUS "MKL: created linker symlink ${_unver_base} -> ${_vlib_base}")
            endif()
          endif()
        endforeach()
      endforeach()
      unset(_mkl_link_dirs)
      unset(_mkl_ver_libs)
    else()
      message(STATUS "LibTorch not found — BenchmarkPyTorchComparison skipped "
                     "(add install dir to CMAKE_PREFIX_PATH to enable)")
    endif()
  endif()

  file(GLOB_RECURSE bench_sources LIST_DIRECTORIES false
       "${CMAKE_CURRENT_SOURCE_DIR}/Benchmark*.cpp"
  )
  if(NOT _memory_torch_found)
    list(FILTER bench_sources EXCLUDE REGEX "BenchmarkPyTorchComparison")
  endif()

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
    if(bench_name STREQUAL "BenchmarkPyTorchComparison")
      target_link_libraries(${target_name} PRIVATE torch)
      target_compile_definitions(${target_name} PRIVATE MEMORY_HAS_LIBTORCH=1)
      target_compile_options(
        ${target_name}
        PRIVATE $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-deprecated-declarations
                -Wno-unused-parameter>
                $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations -Wno-unused-parameter>
      )
      if(WIN32)
        file(GLOB _torch_dlls "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(
          TARGET ${target_name}
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_torch_dlls}
                  $<TARGET_FILE_DIR:${target_name}>
        )
      endif()
      message(STATUS "${target_name}: LibTorch comparison benchmark ENABLED")
    endif()
  endforeach()
endif()
