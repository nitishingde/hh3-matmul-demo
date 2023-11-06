cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

find_package(Hedgehog REQUIRED)
find_package(CUDA     REQUIRED)
find_package(NVTX     REQUIRED)
find_package(MPI      REQUIRED)

add_library(dist_matmul_v3
    STATIC
        data.h
        graphs.h
        states.h
        tasks.h
        utility.h
        matrix_utility.h
        mmd.h
)
set_target_properties(dist_matmul_v3 PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(dist_matmul_v3
    PUBLIC
        HH_USE_CUDA
        HH_ENABLE_CHECK_CUDA
        MMD_ENABLE_CHECK_MPI
)
target_include_directories(dist_matmul_v3
    PUBLIC
        ${Hedgehog_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TCLAP_SOURCE_DIR}/include/
)
target_link_libraries(dist_matmul_v3
    PUBLIC
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${Hedgehog_LIBRARIES}
        MPI::MPI_CXX
        ${OpenBLAS_LIBRARIES}
)

add_executable(demo1_v3 demo1_v3.cpp)
target_link_libraries(demo1_v3
    PUBLIC
        dist_matmul_v3
)
