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
find_package(CUDA REQUIRED)
find_package(NVTX REQUIRED)

if (CUDA_FOUND)
	message(STATUS "CUDA found")
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	set(CUDA_NVCC_FLAGS_RELEASE -O3; -DNDEBUG)
	add_subdirectory(comm_gpu_demo)
	add_subdirectory(dist_gpu_demo)
endif (CUDA_FOUND)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${Hedgehog_CXX_FLAGS})

if (NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Werror -pedantic")
endif (NOT MSVC)

#Gather tclap headers
file(GLOB tclap_headers "../../../utils/tclap/*.h")

# Add tutorials target
#add_subdirectory(tutorial1)
#add_subdirectory(tutorial2)
if (OpenBLAS_FOUND)
	add_subdirectory(tutorial3)
	add_subdirectory(comm_cpu_demo)
endif (OpenBLAS_FOUND)
if (CUDA_FOUND)
#	add_subdirectory(tutorial4)
#	add_subdirectory(tutorial5)

#	add_subdirectory(advanced/tutorial1)
endif (CUDA_FOUND)
