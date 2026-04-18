include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

# ---[ Check for NUMA header availability
if(MEMORY_ENABLE_NUMA)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_FLAGS "-std=c++17")
  check_cxx_source_compiles(
    "#include <numa.h>
    #include <numaif.h>

    int main(int argc, char** argv) {
    }"
    TMP_IS_NUMA_AVAILABLE
  )
  if(TMP_IS_NUMA_AVAILABLE)
    message("--NUMA is available")
  else()
    message("--NUMA is not available")
    set(MEMORY_ENABLE_NUMA OFF)
  endif()
  cmake_pop_check_state()
else()
  message("--NUMA is disabled")
  set(MEMORY_ENABLE_NUMA OFF)
endif()
