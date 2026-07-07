cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(MoonPoly LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

include(CTest)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  if (EXISTS "${PROJECT_SOURCE_DIR}/3rdparty/cutlass")
    message(STATUS "cutlass submodule already exists, skipping update")
  else()
    message(STATUS "Running submodule update to fetch cutlass")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/cutlass
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init 3rdparty/cutlass failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass submodule")
    endif()
  endif()
endif()

set(HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/moonpoly/include
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
)
include_directories(
  ${HEADER_DIRS}
)

set(LIB_DIRS
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)
link_directories(
  ${LIB_DIRS}
)

add_subdirectory(moonpoly)
add_subdirectory(tests/cpp)
add_subdirectory(benchmarks)
