# Compile the actual implementation against deterministic test-only runtimes.
# These targets never link Memory or vendor GPU libraries, so the shims cannot
# replace a real backend in the library or its integration tests.
get_filename_component(_pinned_include "${CMAKE_CURRENT_SOURCE_DIR}/../../include" ABSOLUTE)
foreach(_backend IN ITEMS Cuda Hip)
  set(_target "MemoryPinned${_backend}RuntimeTests")
  add_executable(
    ${_target} TestPinnedRuntime.cpp "${_pinned_include}/helper/pinned_memory_allocator.cpp"
  )
  target_include_directories(${_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${_pinned_include}")
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
