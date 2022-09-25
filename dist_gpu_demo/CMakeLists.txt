cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

find_package(MPI REQUIRED)
find_package(OpenBLAS REQUIRED PATHS /home/sci/nitish/.local)#FIXME

include(FetchContent)

# TCLAP - Templatized Command Line Argument Parser
FetchContent_Declare(
        TCLAP
        GIT_REPOSITORY https://github.com/mirror/tclap.git
        GIT_TAG		   master
)
FetchContent_MakeAvailable(TCLAP)

add_library(dist_matmul
    matrix_utility.h
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

    execution_pipeline/outer_product_exec_pipeline.h

    graph/outer_product_cuda_graph.h

    state/outer_product_computation_state.h
    state/outer_product_computation_state_manager.h
    state/outer_product_cuda_state.h

    task/accumulate_task.h
    task/comm_tasks.h
    task/cuda_copy_in_gpu_task.h
    task/cuda_copy_out_gpu_task.h
    task/cuda_product_task.h
    task/matrix_col_traversal_task.h
    task/matrix_row_traversal_task.h
    task/ttl_managed_memory_recycler_task.h
)
set_target_properties(dist_matmul PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(dist_matmul
    PUBLIC
        HH_USE_CUDA
)
target_include_directories(dist_matmul
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${Hedgehog_INCLUDE_DIRS}
        ${OpenBLAS_INCLUDE_DIRS}
        ${TCLAP_SOURCE_DIR}/include/
)
target_link_libraries(dist_matmul
    PUBLIC
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${Hedgehog_LIBRARIES}
        MPI::MPI_C
        ${OpenBLAS_LIBRARIES}
)

add_executable(dist_test_data test_data.cpp)
target_link_libraries(dist_test_data
    PUBLIC
        dist_matmul
)

add_executable(dist_test_cublas test_cublas.cpp mmd.h)
target_link_libraries(dist_test_cublas
    PUBLIC
        dist_matmul
)

add_executable(dist_mpi_outer_prod_cyclic2d mpi_outer_prod_cyclic2d.cpp mmd.h)
target_link_libraries(dist_mpi_outer_prod_cyclic2d
    PUBLIC
        dist_matmul
)

add_executable(notchpeak_outer_prod_1 notchpeak_outer_prod_1.cpp mmd.h)
target_link_libraries(notchpeak_outer_prod_1
    PUBLIC
        dist_matmul
)
