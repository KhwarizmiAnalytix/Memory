# =============================================================================
# Memory Library C++
# Tests
# =============================================================================
include("${MEMORY_SOURCE_DIR}/Cmake/MemoryDependencies.cmake")

memory_setup_gtest()

add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../PinnedRuntime" "${CMAKE_CURRENT_BINARY_DIR}/PinnedRuntime"
)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../CopyRuntime" "${CMAKE_CURRENT_BINARY_DIR}/CopyRuntime"
)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../CudaCachingAllocator"
  "${CMAKE_CURRENT_BINARY_DIR}/CudaCachingAllocator"
)

file(GLOB_RECURSE test_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
     "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cpp"
)

set(TestFiles)
foreach(_test_source IN ITEMS ${test_sources})
  get_filename_component(file_name "${_test_source}" NAME)

  if(MEMORY_GPU_BACKEND STREQUAL "none")
    if(file_name MATCHES "TestGpu" OR file_name MATCHES "TestGPU")
      continue()
    endif()
  endif()

  # CUDA caching-allocator tests exercise the shared CUDA/HIP TU (gpu_runtime.h
  # aliases). Run them on HIP builds too; TestHip* remains the HIP-only slot.
  if(NOT MEMORY_GPU_BACKEND STREQUAL "cuda" AND NOT MEMORY_GPU_BACKEND STREQUAL "hip")
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

# Task 2.6 exit: public L2/L3 headers must compile with a GPU backend selected and no
# vendor runtime header reachable. Deliberately inherits no CUDA/HIP include directories.
add_library(MemoryConsumerNoVendorHeaders OBJECT ConsumerNoVendorHeaders.cpp)
target_include_directories(
  MemoryConsumerNoVendorHeaders
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
          "${CMAKE_CURRENT_SOURCE_DIR}/../../ThirdParty/Logging/include/logger"
          "${CMAKE_CURRENT_SOURCE_DIR}/../../ThirdParty/Logging"
)
target_compile_features(MemoryConsumerNoVendorHeaders PRIVATE cxx_std_20)
target_compile_definitions(
  MemoryConsumerNoVendorHeaders PRIVATE MEMORY_STATIC_DEFINE MEMORY_HAS_CUDA=1 MEMORY_HAS_HIP=0
                                        MEMORY_HAS_METAL=0 MEMORY_HAS_PROFILER=0
)

add_executable(MemoryCxxTests ${TestFiles})
if(MEMORY_ENABLE_COVERAGE)
  memory_configure_coverage(MemoryCxxTests)
endif()
if(MEMORY_ENABLE_SANITIZER)
  memory_configure_sanitizer(MemoryCxxTests MEMORY)
endif()

# Testing/ (sources + generated *Test.h) and third-party includes match the Memory::Memory
# interface. Memory's own include/ root must come first: some third-party deps (e.g. Profiler)
# vendor an identically-named relative header (common/device.h) and would otherwise shadow
# Memory's own copy for quote-include resolution.
set(_memory_include_root "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
set(_memory_test_include_dirs "${_memory_include_root}" "${CMAKE_CURRENT_SOURCE_DIR}/.."
                              "${CMAKE_CURRENT_BINARY_DIR}/.."
)

# Homebrew (and other package managers) install headers straight under a shared prefix/include
# that other found packages (Fmt::fmt, Mimalloc::mimalloc) pull onto the include path too. If
# Homebrew also has `googletest` installed there, its <gtest/gtest.h> silently shadows the
# vendored ThirdParty/googletest submodule that libgtest.dylib is actually built from —
# same-named header, different ABI (e.g. MakeAndRegisterTestInfo's std::string vs const char*
# overload), causing link-time "symbol not found" instead of a compile error. Pin the vendored
# gtest's own include dir ahead of everything so it always wins.
if(TARGET gtest)
  get_target_property(_memory_gtest_src_dir gtest SOURCE_DIR)
  if(_memory_gtest_src_dir AND EXISTS "${_memory_gtest_src_dir}/include/gtest/gtest.h")
    list(PREPEND _memory_test_include_dirs "${_memory_gtest_src_dir}/include")
  endif()
endif()

target_include_directories(MemoryCxxTests PRIVATE ${_memory_test_include_dirs})
# Memory's headers must come before Profiler's to resolve device_enum correctly
if(MEMORY_DEPENDENCY_INCLUDE_DIRS)
  list(REMOVE_ITEM MEMORY_DEPENDENCY_INCLUDE_DIRS "${_memory_include_root}")
  target_include_directories(MemoryCxxTests PRIVATE ${MEMORY_DEPENDENCY_INCLUDE_DIRS})
endif()

target_compile_definitions(
  MemoryCxxTests PRIVATE MEMORY_HAS_GTEST=1 MEMORY_GOOGLE_TEST XSIGMA_HAS_GTEST=1
                         XSIGMA_GOOGLE_TEST USE_GTEST _VARIADIC_MAX=10
)

target_link_libraries(MemoryCxxTests PRIVATE Gtest::gtest Gtest::gtest_main Memory::Memory)
if(MEMORY_DEPENDENCY_LIBS)
  target_link_libraries(MemoryCxxTests PRIVATE ${MEMORY_DEPENDENCY_LIBS})
endif()
if(MEMORY_DEPENDENCY_LIBS_PRIVATE)
  target_link_libraries(MemoryCxxTests PRIVATE ${MEMORY_DEPENDENCY_LIBS_PRIVATE})
endif()
# Logging::Logging is a required PUBLIC dep of Memory::Memory.

set_target_properties(
  MemoryCxxTests PROPERTIES FOLDER "Tests" VS_DEBUGGER_WORKING_DIRECTORY
                                           "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

add_test(NAME MemoryCxxTests COMMAND MemoryCxxTests
         WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

set_tests_properties(MemoryCxxTests PROPERTIES TIMEOUT 120 LABELS "Memory;Unit")
if(MEMORY_ENABLE_SANITIZER AND MEMORY_SANITIZER_TYPE MATCHES "(^|,)address(,|$)")
  # Oversized-allocation tests expect bad_alloc, so let the allocator report failure.
  # Match CI while retaining any other ASan options supplied by the caller.
  set_property(
    TEST MemoryCxxTests APPEND PROPERTY ENVIRONMENT_MODIFICATION
                                        "ASAN_OPTIONS=string_append::allocator_may_return_null=1"
  )
endif()

if(COMMAND memory_set_target_output_directories)
  memory_set_target_output_directories(MemoryCxxTests)
endif()

if(MEMORY_ENABLE_BENCHMARK)
  # --- Google Benchmark setup ---
  memory_setup_benchmark()
  memory_create_third_party_interface_targets(
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
  option(MEMORY_ENABLE_LIBTORCH "Enable LibTorch (PyTorch C++) CPU allocation comparison benchmark"
         ${_memory_libtorch_default}
  )
  unset(_memory_libtorch_default)

  set(_memory_torch_found OFF)
  if(MEMORY_ENABLE_LIBTORCH)
    set(_saved_module_path "${CMAKE_MODULE_PATH}")
    set(CMAKE_MODULE_PATH "")
    # Force MKL single-dynamic-library mode so Torch's MKL discovery emits only
    # -lmkl_rt instead of the full component set (ScaLAPACK/BLACS/CDFT), which
    # would pull in MPI symbols that may not be installed.
    set(MKL_LINK "sdl")
    # LibTorch's CUDA detection (Caffe2/public/cuda.cmake) asserts
    # CMAKE_CUDA_COMPILER_VERSION == CUDAToolkit_VERSION unconditionally; that
    # assumption holds for nvcc but not for this project's Clang-as-CUDA-compiler
    # setup (Cmake/cuda.cmake), where CMAKE_CUDA_COMPILER_VERSION is Clang's own
    # version, not the toolkit's (Memory's own cuda.cmake narrows the equivalent
    # check to CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" for the same reason).
    # Shadow it with a directory-local regular variable (not touching the cache)
    # so Torch's check reads the toolkit version here, matching what Memory
    # already decided is the correct comparison for a Clang CUDA compiler.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND MEMORY_GPU_BACKEND STREQUAL "cuda")
      set(CMAKE_CUDA_COMPILER_VERSION "${CUDAToolkit_VERSION}")
    endif()
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
      # LibTorch's CUDA/CPU targets (Caffe2Targets.cmake) export
      # INTERFACE_COMPILE_OPTIONS of literal MSVC flags (/permissive-, /EHsc,
      # /bigobj, /Z7), baked in at LibTorch's own MSVC build time with no
      # compiler-ID guard. clang++'s GNU-style driver does not parse slash
      # switches at all (it tries to open them as files), so any target
      # linking these imported targets fails to compile. Blank the property
      # on the imported targets themselves post-find_package: Clang already
      # defaults to strict conformance and exception support, so nothing
      # here is load-bearing for a Clang consumer.
      if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        foreach(_torch_target IN ITEMS c10 c10_cuda torch_cpu torch_cuda)
          if(TARGET ${_torch_target})
            set_property(TARGET ${_torch_target} PROPERTY INTERFACE_COMPILE_OPTIONS "")
          endif()
        endforeach()
        unset(_torch_target)
      endif()
    else()
      message(STATUS "LibTorch not found — BenchmarkPyTorchComparison skipped "
                     "(add install dir to CMAKE_PREFIX_PATH to enable)"
      )
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
    if(MEMORY_ENABLE_COVERAGE)
      memory_configure_coverage(${target_name})
    endif()
    if(MEMORY_ENABLE_SANITIZER)
      memory_configure_sanitizer(${target_name} MEMORY)
    endif()
    target_include_directories(
      ${target_name} PRIVATE "${_memory_include_root}" "${CMAKE_CURRENT_SOURCE_DIR}/.."
                             "${CMAKE_CURRENT_BINARY_DIR}/.."
    )
    if(MEMORY_DEPENDENCY_INCLUDE_DIRS)
      set(_bench_dep_includes ${MEMORY_DEPENDENCY_INCLUDE_DIRS})
      list(REMOVE_ITEM _bench_dep_includes "${_memory_include_root}")
      target_include_directories(${target_name} PRIVATE ${_bench_dep_includes})
      unset(_bench_dep_includes)
    endif()
    target_link_libraries(
      ${target_name} PRIVATE Memory::benchmark Memory::benchmark_main Memory::Memory
    )
    if(MEMORY_DEPENDENCY_LIBS)
      target_link_libraries(${target_name} PRIVATE ${MEMORY_DEPENDENCY_LIBS})
    endif()
    if(MEMORY_DEPENDENCY_LIBS_PRIVATE)
      target_link_libraries(${target_name} PRIVATE ${MEMORY_DEPENDENCY_LIBS_PRIVATE})
    endif()
    # Same Logging visibility as MemoryCxxTests (via Memory::Memory + MEMORY_DEPENDENCY_LIBS).
    if(COMMAND memory_set_target_output_directories)
      memory_set_target_output_directories(${target_name})
    endif()

    # Register with CTest (same pattern as Core/Parallel/Vectorization) so
    # `setup.py ...test...` runs and reports these benchmarks instead of only
    # building them. --benchmark_min_time=0.01s keeps ctest runs fast; run the
    # binary directly (no filter/min-time) for real performance numbers.
    add_test(NAME ${target_name} COMMAND ${target_name} --benchmark_min_time=0.01s
             WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )
    set_tests_properties(
      ${target_name} PROPERTIES TIMEOUT 600 LABELS "Benchmark;Performance;Memory"
    )

    if(bench_name MATCHES "^BenchmarkPyTorchComparison")
      target_link_libraries(${target_name} PRIVATE torch)
      target_compile_definitions(${target_name} PRIVATE MEMORY_HAS_LIBTORCH=1)
      target_compile_options(
        ${target_name}
        PRIVATE $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-deprecated-declarations
                -Wno-unused-parameter> $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations
                -Wno-unused-parameter>
      )
      if(WIN32)
        file(GLOB _torch_dlls "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(
          TARGET ${target_name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                                   ${_torch_dlls} $<TARGET_FILE_DIR:${target_name}>
        )
        if(TARGET torch_cuda)
          # torch_cuda's own INTERFACE_LINK_LIBRARIES carries this exact
          # -INCLUDE: flag (Caffe2Targets.cmake) to force-reference a CUDA
          # symbol and pull in the static initializer that registers the real
          # CUDAHooks implementation; without it the linker drops the
          # otherwise-unreferenced registration object, and
          # torch::cuda::is_available() silently returns false (falls back to
          # the CPU-only stub hooks) even though the CUDA driver and cudart
          # both work fine. It is not reaching this target through the normal
          # target_link_libraries(torch) graph, so add it directly; confirmed
          # by a minimal out-of-tree repro (device_count 0 without it, 1 with
          # it, same binary otherwise).
          target_link_options(${target_name} PRIVATE
                               "-Wl,-INCLUDE:?warp_size@cuda@at@@YAHXZ"
          )
        endif()
      endif()
      message(STATUS "${target_name}: LibTorch comparison benchmark ENABLED")
    endif()
  endforeach()

  # Phase 0 baseline harnesses (plan §7, tasks 0.1/0.4): custom main(), raw samples,
  # JSON + manifest. Not Google Benchmark; not registered with CTest (run manually).
  foreach(_p0 IN ITEMS Phase0CpuBaseline Phase0CudaColdWarm)
    string(TOLOWER "${_p0}" _p0_lower)
    set(_p0_target "benchmark_memory_${_p0_lower}")
    add_executable(${_p0_target} "${CMAKE_CURRENT_SOURCE_DIR}/${_p0}.cpp")
    if(MEMORY_ENABLE_COVERAGE)
      memory_configure_coverage(${_p0_target})
    endif()
    if(MEMORY_ENABLE_SANITIZER)
      memory_configure_sanitizer(${_p0_target} MEMORY)
    endif()
    target_include_directories(
      ${_p0_target} PRIVATE "${_memory_include_root}" "${CMAKE_CURRENT_SOURCE_DIR}/.."
                            "${CMAKE_CURRENT_BINARY_DIR}/.."
    )
    if(MEMORY_DEPENDENCY_INCLUDE_DIRS)
      set(_p0_dep_includes ${MEMORY_DEPENDENCY_INCLUDE_DIRS})
      list(REMOVE_ITEM _p0_dep_includes "${_memory_include_root}")
      target_include_directories(${_p0_target} PRIVATE ${_p0_dep_includes})
      unset(_p0_dep_includes)
    endif()
    target_link_libraries(${_p0_target} PRIVATE Memory::Memory)
    if(MEMORY_DEPENDENCY_LIBS)
      target_link_libraries(${_p0_target} PRIVATE ${MEMORY_DEPENDENCY_LIBS})
    endif()
    if(MEMORY_DEPENDENCY_LIBS_PRIVATE)
      target_link_libraries(${_p0_target} PRIVATE ${MEMORY_DEPENDENCY_LIBS_PRIVATE})
    endif()
    if(COMMAND memory_set_target_output_directories)
      memory_set_target_output_directories(${_p0_target})
    endif()
  endforeach()
endif()
