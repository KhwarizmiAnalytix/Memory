# Memory module (standalone; same layout pattern as Library/Parallel).
cmake_minimum_required(VERSION 3.16)

# =============================================================================
# Memory-owned CMake configuration
# All GPU and TBB allocator modules live in Library/Memory/Cmake/ and are
# included exclusively here.
# =============================================================================
include(cuda)        # Detects/configures CUDA when MEMORY_ENABLE_CUDA=ON
include(hip)         # Detects/configures HIP/ROCm when MEMORY_ENABLE_HIPA=ON
include(tbb_memory)  # Finds/builds TBB scalable allocator when MEMORY_ENABLE_TBB=ON

# Link TBB allocator directly: ThirdParty/CMakeLists.txt can no longer do this
# because tbb_memory.cmake now runs after ThirdParty.
if(MEMORY_ENABLE_TBB AND TARGET Tbb::tbbmalloc)
  list(APPEND MEMORY_DEPENDENCY_LIBS Tbb::tbbmalloc)
endif()

# Map feature flags to compile definitions scoped to this module.
if(MEMORY_ENABLE_TBB)
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_TBB=1)
else()
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_TBB=0)
endif()

if(MEMORY_ENABLE_CUDA)
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_CUDA=1)
else()
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_CUDA=0)
endif()

if(MEMORY_ENABLE_HIPA)
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_HIP=1)
else()
  list(APPEND MEMORY_COMPILE_DEFINITIONS MEMORY_HAS_HIP=0)
endif()

list(APPEND MEMORY_DEPENDENCY_COMPILE_DEFINITIONS ${MEMORY_COMPILE_DEFINITIONS})

set(memory_headers)
set(memory_sources)

if(NOT DEFINED MEMORY_CXX_STANDARD OR MEMORY_CXX_STANDARD STREQUAL "")
  set(MEMORY_CXX_STANDARD ${CMAKE_CXX_STANDARD})
endif()

file(
  GLOB_RECURSE memory_headers
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx"
)

file(
  GLOB_RECURSE memory_sources
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Keep test sources out of the Memory library target.
list(FILTER memory_sources EXCLUDE REGEX ".*/Testing/.*")
list(FILTER memory_headers EXCLUDE REGEX ".*/Testing/.*")

if(NOT MEMORY_ENABLE_CUDA OR MEMORY_ENABLE_HIP)
  file(GLOB_RECURSE GPU_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cpp")
  file(GLOB_RECURSE GPU_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.h"
       "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.hxx")
  list(REMOVE_ITEM memory_headers ${GPU_HEADER_FILES})
  list(REMOVE_ITEM memory_sources ${GPU_SOURCE_FILES})
  list(REMOVE_ITEM memory_sources "${CMAKE_CURRENT_SOURCE_DIR}/cpu/allocator_device.cpp")
  list(REMOVE_ITEM memory_headers "${CMAKE_CURRENT_SOURCE_DIR}/cpu/allocator_device.h")
endif()

add_library(Memory ${memory_sources} ${memory_headers})

if(BUILD_SHARED_LIBS)
  target_compile_definitions(Memory PUBLIC MEMORY_SHARED_DEFINE PRIVATE MEMORY_BUILDING_DLL)
  if(COMMAND memory_set_target_output_directories)
    memory_set_target_output_directories(Memory)
  endif()
else()
  target_compile_definitions(Memory PUBLIC MEMORY_STATIC_DEFINE)
endif()

add_library(Memory::Memory ALIAS Memory)

if(MEMORY_ENABLE_LTO)
  set_target_properties(Memory PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(COMMAND memory_apply_iwyu)
  memory_apply_iwyu(Memory)
endif()

if(COMMAND memory_module_create_filters)
  memory_module_create_filters("${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set_target_properties(
  Memory PROPERTIES CXX_STANDARD ${MEMORY_CXX_STANDARD} CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
)

if(MSVC)
  target_compile_options(Memory PRIVATE /WX)
else()
  target_compile_options(Memory PRIVATE -include cstdlib)
endif()

# Include roots: Library/Memory → "allocator.h", "backend/...", "common/memory_export.h"
target_include_directories(
  Memory
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

if(PROFILER_HAS_NATIVE_PROFILER OR PROFILER_ENABLE_NATIVE_PROFILER)
  target_include_directories(Memory PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Library/Profiler>)
endif()

if(MEMORY_DEPENDENCY_INCLUDE_DIRS)
  target_include_directories(Memory PUBLIC ${MEMORY_DEPENDENCY_INCLUDE_DIRS})
endif()

target_link_libraries(Memory PUBLIC Memory::build Memory::Logging)

if(COMMAND memory_target_clang_tidy)
  memory_target_clang_tidy(Memory)
endif()

if(MEMORY_DEPENDENCY_LIBS)
  target_link_libraries(Memory PUBLIC ${MEMORY_DEPENDENCY_LIBS})
endif()

if(MEMORY_DEPENDENCY_COMPILE_DEFINITIONS)
  target_compile_definitions(Memory PUBLIC ${MEMORY_DEPENDENCY_COMPILE_DEFINITIONS})
endif()

if(WIN32 AND NOT MSVC)
  target_link_libraries(Memory PRIVATE -ldbghelp)
endif()

if(MEMORY_ENABLE_MIMALLOC AND TARGET Memory::mimalloc)
  add_dependencies(Memory Memory::mimalloc)
endif()

install(
  TARGETS Memory
  EXPORT MemoryTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
)

install(FILES ${memory_headers} DESTINATION include/Memory/Memory COMPONENT Development)

if(BUILD_TESTING)
  add_subdirectory(Testing/Cxx)
endif()
