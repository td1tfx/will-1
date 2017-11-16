project(libwill-cuda)
cmake_minimum_required(VERSION 2.6)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS "-O3")
if (CMAKE_COMPILER_IS_GNUCC)
  execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                  OUTPUT_VARIABLE GCC_VERSION)
  string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
  list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
  list(GET GCC_VERSION_COMPONENTS 1 GCC_MINOR)
endif()
if(GCC_VERSION VERSION_GREATER 5.9)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} " -ccbin gcc-5")
endif()

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../lib)
cuda_add_library(will-cuda SHARED will_cuda.cu)
