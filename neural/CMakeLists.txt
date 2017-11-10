project(libwill)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -pthread -fopenmp")

include_directories(./lib/include /usr/local/cuda/include ./src ./src/others ./cuda)
link_directories(. ../lib /usr/local/cuda/lib64 /usr/local/cuda/lib64/stubs)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_subdirectory(cuda)
add_subdirectory(src)
add_subdirectory(windows)




