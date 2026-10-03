# Compile the actual gpu/cuda_caching_allocator.cpp implementation against a
# deterministic test-only runtime (fake_runtime.h) to exercise driver-failure
# and budget-accounting paths that real hardware cannot reliably trigger on
# demand. Mirrors Testing/PinnedRuntime/'s shadow-header technique. Never
# links Memory or vendor GPU libraries, so this shim cannot replace a real
# backend in the library or its integration tests.
#
# Builds only the HIP-labeled variant (MEMORY_HAS_CUDA=0, MEMORY_HAS_HIP=1):
# the CUDA driver-API expandable-VM path (<cuda.h>, try_cu_vm_alloc) is out of
# scope to fake here, and the HIP VM path never compiles in against this fake
# (it requires HIP_VERSION >= 50600000, which is never defined). The event
# insertion / budget-check / retry-chain / block-pooling logic under test is
# shared C++ code between CUDA and HIP via gpu/gpu_runtime.h's macro aliasing,
# so exercising it under HIP labels is equally valid coverage.
get_filename_component(_cca_include "${CMAKE_CURRENT_SOURCE_DIR}/../../include" ABSOLUTE)
get_filename_component(_cca_src "${CMAKE_CURRENT_SOURCE_DIR}/../../src" ABSOLUTE)

add_executable(
  MemoryCudaCachingAllocatorRuntimeTests TestCudaCachingAllocatorRuntime.cpp
  "${_cca_src}/gpu/cuda_caching_allocator.cpp"
  "${_cca_src}/profiler/unified_memory_stats.cpp"
  "${_cca_src}/common/allocation_id.cpp"
)
target_include_directories(
  MemoryCudaCachingAllocatorRuntimeTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${_cca_include}"
)
target_compile_features(MemoryCudaCachingAllocatorRuntimeTests PRIVATE cxx_std_20)
target_compile_definitions(
  MemoryCudaCachingAllocatorRuntimeTests
  PRIVATE MEMORY_STATIC_DEFINE
          MEMORY_HAS_CUDA=0
          MEMORY_HAS_HIP=1
          MEMORY_HAS_METAL=0
          MEMORY_HAS_PROFILER=0
)
target_link_libraries(
  MemoryCudaCachingAllocatorRuntimeTests PRIVATE Logging::Logging Gtest::gtest_main
)
if(MSVC)
  target_compile_options(MemoryCudaCachingAllocatorRuntimeTests PRIVATE /Zc:__cplusplus)
endif()

add_test(
  NAME MemoryCudaCachingAllocatorRuntimeTests COMMAND MemoryCudaCachingAllocatorRuntimeTests
)
set_tests_properties(
  MemoryCudaCachingAllocatorRuntimeTests PROPERTIES TIMEOUT 60 LABELS "Memory;Unit;RuntimeShim"
)

# Phase 0 (tasks 0.2/0.3): host-overhead baseline under the fake runtime. Manual run only.
add_executable(
  MemoryGpuShimBenchPhase0 Phase0GpuShimBench.cpp
  "${_cca_src}/gpu/cuda_caching_allocator.cpp"
  "${_cca_src}/profiler/unified_memory_stats.cpp"
  "${_cca_src}/common/allocation_id.cpp"
)
target_include_directories(MemoryGpuShimBenchPhase0 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${_cca_include}")
target_compile_features(MemoryGpuShimBenchPhase0 PRIVATE cxx_std_20)
target_compile_definitions(
  MemoryGpuShimBenchPhase0 PRIVATE MEMORY_STATIC_DEFINE MEMORY_HAS_CUDA=0 MEMORY_HAS_HIP=1
                                   MEMORY_HAS_METAL=0 MEMORY_HAS_PROFILER=0
)
target_link_libraries(MemoryGpuShimBenchPhase0 PRIVATE Logging::Logging)
if(MSVC)
  target_compile_options(MemoryGpuShimBenchPhase0 PRIVATE /Zc:__cplusplus)
endif()

# Task 1.10: host-side churn replay for sanitizer runs (manual, not a CTest case).
add_executable(
  MemoryGpuShimChurn Phase1ChurnShim.cpp
  "${_cca_src}/gpu/cuda_caching_allocator.cpp"
  "${_cca_src}/profiler/unified_memory_stats.cpp"
  "${_cca_src}/common/allocation_id.cpp"
)
target_include_directories(MemoryGpuShimChurn PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${_cca_include}")
target_compile_features(MemoryGpuShimChurn PRIVATE cxx_std_20)
target_compile_definitions(
  MemoryGpuShimChurn PRIVATE MEMORY_STATIC_DEFINE MEMORY_HAS_CUDA=0 MEMORY_HAS_HIP=1
                             MEMORY_HAS_METAL=0 MEMORY_HAS_PROFILER=0
)
target_link_libraries(MemoryGpuShimChurn PRIVATE Logging::Logging)
if(MSVC)
  target_compile_options(MemoryGpuShimChurn PRIVATE /Zc:__cplusplus)
endif()
