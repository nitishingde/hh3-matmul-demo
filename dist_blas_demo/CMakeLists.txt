cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

find_package(MPI REQUIRED)

add_library(blas_matmul
    STATIC
        data.h
        states.h
        tasks.h
        utility.h
)
set_target_properties(blas_matmul PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(blas_matmul
    PUBLIC
        HH_USE_CUDA
        HH_ENABLE_CHECK_CUDA
        MMD_ENABLE_CHECK_MPI
        OPENBLAS_NUM_THREADS=1
)
target_include_directories(blas_matmul
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${Hedgehog_INCLUDE_DIRS}
        ${OpenBLAS_INCLUDE_DIRS}
        ${TCLAP_SOURCE_DIR}/include/
)
target_link_libraries(blas_matmul
    PUBLIC
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${Hedgehog_LIBRARIES}
        MPI::MPI_CXX
        ${OpenBLAS_LIBRARIES}
)

add_executable(blas_demo blas_demo.cpp)
target_link_libraries(blas_demo
    PUBLIC
        blas_matmul
)
