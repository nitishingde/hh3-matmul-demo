cmake_minimum_required(VERSION 3.17)
project(hh3-matmul)

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
	FetchContent_GetProperties(Hedgehog)
	if(NOT hedgehog_POPULATED)
		FetchContent_Populate(Hedgehog)
	endif(NOT hedgehog_POPULATED)

	# Need to set variable Hedgehog_INCLUDE_DIR to the downloaded hedgehog src code
	set(Hedgehog_INCLUDE_DIR ${hedgehog_SOURCE_DIR})
endif(NOT Hedgehog_INCLUDE_DIR)

# Test NewHedgehog library
find_package(Hedgehog REQUIRED)

# TCLAP - Templatized Command Line Argument Parser
FetchContent_Declare(
	TCLAP
	GIT_REPOSITORY https://github.com/mirror/tclap.git
	GIT_TAG        v1.2.5
)
FetchContent_GetProperties(TCLAP)
if(NOT tclap_POPULATED)
	FetchContent_Populate(TCLAP)
endif(NOT tclap_POPULATED)
set(TCLAP_SOURCE_DIR ${tclap_SOURCE_DIR})

# OpenBLAS
find_package(OpenBLAS QUIET)

# Cuda
find_package(CUDA QUIET)
find_package(NVTX QUIET)

# MPI
find_package(MPI QUIET)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Hedgehog_CXX_FLAGS}")

if (NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Werror -pedantic")
endif (NOT MSVC)

if(CUDA_FOUND AND MPI_FOUND)
	message(STATUS "CUDA found")
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	set(CUDA_NVCC_FLAGS_RELEASE -O3; -DNDEBUG)
	add_subdirectory(dist_gpu_demo)
	add_subdirectory(dist_gpu_demo_v2)
	add_subdirectory(dist_gpu_demo_v3)
endif(CUDA_FOUND AND MPI_FOUND)

if(OpenBLAS_FOUND AND MPI_FOUND)
    message(STATUS "OpenBLAS found")
	add_subdirectory(dist_blas_demo)
endif(OpenBLAS_FOUND AND MPI_FOUND)
