cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts -Wno-error=sign-compare")

find_package(MPI REQUIRED)
find_package(OpenBLAS QUIET)#FIXME

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
)
set_target_properties(dist_matmul PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(dist_matmul
    PUBLIC
        HH_USE_CUDA
        HH_ENABLE_CHECK_CUDA
        MMD_ENABLE_CHECK_MPI
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
        MPI::MPI_CXX
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

add_executable(cibc cibc.cpp mmd.h)
target_link_libraries(cibc
    PUBLIC
        dist_matmul
)

add_executable(cibc_outer_prod_1 cibc_outer_prod_1.cpp mmd.h)
target_link_libraries(cibc_outer_prod_1
    PUBLIC
        dist_matmul
)

add_executable(cibc_outer_prod_2 cibc_outer_prod_2.cpp mmd.h)
target_link_libraries(cibc_outer_prod_2
    PUBLIC
        dist_matmul
)

add_executable(cibc_outer_prod_3 cibc_outer_prod_3.cpp mmd.h)
target_link_libraries(cibc_outer_prod_3
    PUBLIC
       dist_matmul
)

add_executable(notchpeak_outer_prod_1 notchpeak_outer_prod_1.cpp mmd.h)
target_link_libraries(notchpeak_outer_prod_1
    PUBLIC
        dist_matmul
)

add_executable(notchpeak_outer_prod_2 notchpeak_outer_prod_2.cpp mmd.h)
target_link_libraries(notchpeak_outer_prod_2
    PUBLIC
        dist_matmul
)

add_executable(benchmark_outer_prod_2 benchmark_outer_prod_2.cpp mmd.h)
target_link_libraries(benchmark_outer_prod_2
    PUBLIC
        dist_matmul
)

add_executable(benchmark_outer_prod_3 benchmark_outer_prod_3.cpp mmd.h)
target_link_libraries(benchmark_outer_prod_3
    PUBLIC
        dist_matmul
)
