include_guard(DIRECTORY)
find_program(MEMORYCHECK_COMMAND NAMES valgrind)
if(NOT MEMORYCHECK_COMMAND)
  message(
    FATAL_ERROR
      "Valgrind requested but not found; set MEMORYCHECK_COMMAND or disable MEMORY_ENABLE_VALGRIND"
  )
endif()
set(MEMORYCHECK_COMMAND_OPTIONS
    "--tool=memcheck --leak-check=full --show-leak-kinds=definite --errors-for-leak-kinds=definite --track-origins=yes --track-fds=yes --num-callers=50 --trace-children=yes --error-exitcode=1"
)
set(MEMORYCHECK_SUPPRESSIONS_FILE
    "${CMAKE_CURRENT_LIST_DIR}/../Scripts/suppressions/valgrind_suppression.txt"
)
set(MEMORY_VALGRIND_TIMEOUT_MULTIPLIER 20 CACHE STRING "Valgrind test timeout multiplier")
set(DART_TESTING_TIMEOUT 1800 CACHE STRING "CTest dashboard timeout")

function(memory_apply_valgrind_timeouts)
  if(ARGC GREATER 0)
    set(_dir "${ARGV0}")
  else()
    set(_dir "${MEMORY_SOURCE_DIR}")
  endif()
  get_property(_tests DIRECTORY "${_dir}" PROPERTY TESTS)
  foreach(_test IN LISTS _tests)
    get_property(
      _timeout
      TEST "${_test}"
      DIRECTORY "${_dir}"
      PROPERTY TIMEOUT
    )
    if(_timeout)
      math(EXPR _timeout "${_timeout} * ${MEMORY_VALGRIND_TIMEOUT_MULTIPLIER}")
    else()
      set(_timeout ${DART_TESTING_TIMEOUT})
    endif()
    set_property(TEST "${_test}" DIRECTORY "${_dir}" PROPERTY TIMEOUT ${_timeout})
  endforeach()
  get_property(_children DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
  foreach(_child IN LISTS _children)
    memory_apply_valgrind_timeouts("${_child}")
  endforeach()
endfunction()
