cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

include(FetchContent)

# Uintah2::Comm
FetchContent_Declare(
        Uintah2
        GIT_REPOSITORY https://github.com/nitishingde/Uintah2
        GIT_TAG		   main
)
FetchContent_MakeAvailable(Uintah2)

# Serialization: cereal
set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "Skip building performance sandbox comparison (requires boost)" ON)
FetchContent_Declare(
        cereal
        GIT_REPOSITORY https://github.com/USCiLab/cereal.git
        GIT_TAG		   master
)
FetchContent_MakeAvailable(cereal)

add_library(hh3_matmul
        mm.h
        utility.h

        data/cuda_matrix_block_data.h
        data/data_type.h
        data/matrix_block_data.h
        data/matrix_data.h
        data/serialization.h

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

target_compile_definitions(hh3_matmul
    PUBLIC
        HH_USE_CUDA
)

target_include_directories(hh3_matmul
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${Hedgehog_INCLUDE_DIRS}
)

target_link_libraries(hh3_matmul
    PUBLIC
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${Hedgehog_LIBRARIES}
        cereal::cereal
        Uintah2::comm
)

add_executable(gpu_demo gpu_demo.cpp)
target_link_libraries(gpu_demo
    PUBLIC
        hh3_matmul
)

add_executable(comm_gpu_demo comm_gpu_demo.cpp)
target_link_libraries(comm_gpu_demo
    PUBLIC
        hh3_matmul
)
