cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

add_executable(gpu_demo
        mm.h
        gpu_demo.cpp
        utility.h

        data/cuda_matrix_block_data.h
        data/data_type.h
        data/matrix_block_data.h
        data/matrix_data.h

        execution_pipeline/multi_gpu_exec_pipeline.h

        graph/gpu_computation_graph.h

        state/cuda_input_block_state.h
        state/output_state.h
        state/partial_computation_state.h
        state/partial_computation_state_manager.h

        task/cuda_copy_in_gpu_task.h
        task/cuda_copy_out_gpu_task.h
        task/cuda_product_task.h
        task/matrix_column_traversal_task.h
        task/matrix_row_traversal_task.h
        task/addition_task.h
)

target_compile_definitions(gpu_demo
    PUBLIC
        HH_USE_CUDA
)

target_include_directories(gpu_demo
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${Hedgehog_INCLUDE_DIRS}
)

target_link_libraries(gpu_demo
    PUBLIC
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${Hedgehog_LIBRARIES}
)
