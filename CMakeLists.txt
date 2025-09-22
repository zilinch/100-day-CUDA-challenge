cmake_minimum_required(VERSION 3.20)
project(cuda_100day LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
# 1080 Ti = compute capability 6.1
set(CMAKE_CUDA_ARCHITECTURES 61)

# Auto-add every subdirectory named "day*"
file(GLOB DAY_DIRS RELATIVE ${CMAKE_SOURCE_DIR} "day*")

# Put executables in build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

foreach(D ${DAY_DIRS})
  if (IS_DIRECTORY "${CMAKE_SOURCE_DIR}/${D}")
    add_subdirectory(${D})
  endif()
endforeach()