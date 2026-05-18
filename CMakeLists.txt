cmake_minimum_required(VERSION 3.15)
project(lob_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")

# pybind11: find via pip-installed pybind11 or system install
if(DEFINED PYBIND11_CMAKE_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  find_package(pybind11 CONFIG REQUIRED)
endif()

set(LOB_SOURCES
    cpp/lob/order_book.cpp
    cpp/lob/matching_engine.cpp
)

pybind11_add_module(lob_engine
    cpp/bindings/py_lob.cpp
    ${LOB_SOURCES}
)

target_include_directories(lob_engine PRIVATE cpp/)
target_compile_features(lob_engine PRIVATE cxx_std_17)

# Enable warnings
target_compile_options(lob_engine PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
