cmake_minimum_required(VERSION 3.17)
project(hh3-matmul)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")

# Need to defined variable Hedgehog_INCLUDE_DIR to the hedgehog path
# hedgehog
find_package(Hedgehog REQUIRED)

# OpenBLAS
find_package(OpenBLAS QUIET)

# Cuda
find_package(CUDA QUIET)
find_package(NVTX QUIET)

# MPI
find_package(MPI QUIET)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${Hedgehog_CXX_FLAGS})

if (NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Werror -pedantic")
endif (NOT MSVC)

if(CUDA_FOUND AND MPI_FOUND)
	message(STATUS "CUDA found")
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	set(CUDA_NVCC_FLAGS_RELEASE -O3; -DNDEBUG)
	add_subdirectory(dist_gpu_demo)
endif(CUDA_FOUND AND MPI_FOUND)

if(OpenBLAS_FOUND AND MPI_FOUND)
    message(STATUS "OpenBLAS found")
	add_subdirectory(dist_blas_demo)
endif(OpenBLAS_FOUND AND MPI_FOUND)
