cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

find_package(MPI REQUIRED)

include(FetchContent)

# TCLAP - Templatized Command Line Argument Parser
FetchContent_Declare(
        TCLAP
        GIT_REPOSITORY https://github.com/mirror/tclap.git
        GIT_TAG		   master
)
FetchContent_MakeAvailable(TCLAP)

add_library(hh3_matmul_dist
    utility.h

    data/contiguous_sub_matrix_container.h
    data/cuda_matrix_tile.h
    data/cuda_memory.h
    data/cyclic2d_matrix_container.h
    data/matrix_container_interface.h
    data/matrix_meta_data.h
    data/matrix_order.h
    data/matrix_tile.h
    data/redundant_matrix_container.h
)

set_target_properties(hh3_matmul_dist PROPERTIES LINKER_LANGUAGE CXX)

target_compile_definitions(hh3_matmul_dist
    PUBLIC
        HH_USE_CUDA
)

target_include_directories(hh3_matmul_dist
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${Hedgehog_INCLUDE_DIRS}
        ${TCLAP_SOURCE_DIR}/include/
)

target_link_libraries(hh3_matmul_dist
    PUBLIC
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${Hedgehog_LIBRARIES}
        MPI::MPI_C
)

add_executable(test_data test_data.cpp)
target_link_libraries(test_data
    PUBLIC
        hh3_matmul_dist
)
