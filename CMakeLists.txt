cmake_minimum_required(VERSION 3.17)
project(hh3-matmul CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")

include(FetchContent)

if(NOT Hedgehog_INCLUDE_DIR)
	# Hedgehog 3.0
	FetchContent_Declare(
		Hedgehog
		GIT_REPOSITORY https://github.com/usnistgov/hedgehog.git
		GIT_TAG        master
	)
	FetchContent_MakeAvailable(Hedgehog)

	# Need to set variable Hedgehog_INCLUDE_DIR to the downloaded hedgehog src code
	set(Hedgehog_INCLUDE_DIR "${hedgehog_SOURCE_DIR}" CACHE STRING "${hedgehog_SOURCE_DIR}" FORCE)
endif(NOT Hedgehog_INCLUDE_DIR)

# Test NewHedgehog library
find_package(Hedgehog REQUIRED)

# TCLAP - Templatized Command Line Argument Parser
FetchContent_Declare(
	TCLAP
	GIT_REPOSITORY https://github.com/DAarno/tclap.git
	GIT_TAG        1.4.0-rc2
)
FetchContent_MakeAvailable(TCLAP)

# OpenBLAS
find_package(OpenBLAS CONFIG QUIET)

# Cuda
find_package(CUDAToolkit QUIET)
find_package(NVTX QUIET)

# MPI
find_package(MPI QUIET)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Hedgehog_CXX_FLAGS}")

if (NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Werror -pedantic")
endif (NOT MSVC)

if(MPI_FOUND AND CUDA_FOUND)
	# message(STATUS "CUDA found")
	# set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	# set(CUDA_NVCC_FLAGS_RELEASE -O3; -DNDEBUG)
	add_subdirectory(dist_gpu_demo)
	add_subdirectory(dist_gpu_demo_v2)
	add_subdirectory(dist_gpu_demo_v3)
endif(MPI_FOUND AND CUDA_FOUND)

if(OpenBLAS_FOUND AND MPI_FOUND)
    message(STATUS "OpenBLAS found")
	add_subdirectory(dist_blas_demo)
endif(OpenBLAS_FOUND AND MPI_FOUND)
