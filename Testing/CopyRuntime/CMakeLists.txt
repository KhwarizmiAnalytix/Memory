# Compile copy_token, the byte copy router (src/transfer.cpp) and allocator paths against deterministic test-only runtimes.
# These targets never link Memory or vendor GPU libraries, exercising GPU paths
# (device guards, stream queries, peer copies) without hardware.

get_filename_component(_copy_include "${CMAKE_CURRENT_SOURCE_DIR}/../../include" ABSOLUTE)
get_filename_component(_copy_src "${CMAKE_CURRENT_SOURCE_DIR}/../../src" ABSOLUTE)

foreach(_backend IN ITEMS Cuda Hip)
  set(_target "MemoryCopy${_backend}RuntimeTests")
  add_executable(
    ${_target}
    TestCopyRuntime.cpp
    gpu_dispatch_stub.cpp
    "${_copy_src}/retained_operation_service.cpp"
    "${_copy_src}/transfer.cpp"
    "${_copy_src}/common/allocation_id.cpp"
  )
  target_include_directories(${_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_include_directories(${_target} PRIVATE "${_copy_include}")
  target_include_directories(${_target} SYSTEM PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/include")
  target_compile_features(${_target} PRIVATE cxx_std_17)
  target_compile_definitions(${_target} PRIVATE MEMORY_STATIC_DEFINE MEMORY_HAS_METAL=0)

  if(_backend STREQUAL "Cuda")
    target_compile_definitions(${_target} PRIVATE MEMORY_HAS_CUDA=1 MEMORY_HAS_HIP=0)
  else()
    target_compile_definitions(${_target} PRIVATE MEMORY_HAS_CUDA=0 MEMORY_HAS_HIP=1)
  endif()

  target_link_libraries(${_target} PRIVATE Gtest::gtest_main)
  add_test(NAME ${_target} COMMAND ${_target})
  set_tests_properties(${_target} PROPERTIES TIMEOUT 30 LABELS "Memory;Unit;RuntimeShim")
endforeach()

# Phase 0 (tasks 0.2/0.3) host-overhead baseline under the deterministic shim.
# Manual run only; not a CTest case. CUDA labels: the HIP variant shares the code.
add_executable(
  MemoryCopyShimBenchPhase0
  Phase0CopyShimBench.cpp
  gpu_dispatch_stub.cpp
  "${_copy_src}/retained_operation_service.cpp"
  "${_copy_src}/transfer.cpp"
  "${_copy_src}/common/allocation_id.cpp"
)
target_include_directories(MemoryCopyShimBenchPhase0 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${_copy_include}")
target_include_directories(MemoryCopyShimBenchPhase0 SYSTEM PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/include")
target_compile_features(MemoryCopyShimBenchPhase0 PRIVATE cxx_std_17)
target_compile_definitions(
  MemoryCopyShimBenchPhase0 PRIVATE MEMORY_STATIC_DEFINE MEMORY_HAS_METAL=0 MEMORY_HAS_CUDA=1 MEMORY_HAS_HIP=0
)
