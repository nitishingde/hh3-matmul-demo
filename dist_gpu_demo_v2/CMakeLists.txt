cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

find_package(Hedgehog REQUIRED)
find_package(CUDA     REQUIRED)
find_package(NVTX     REQUIRED)
find_package(MPI      REQUIRED)

include(FetchContent)

# TCLAP - Templatized Command Line Argument Parser
FetchContent_Declare(
    TCLAP
    GIT_REPOSITORY https://github.com/mirror/tclap.git
    GIT_TAG		   master
)
FetchContent_GetProperties(TCLAP)
if(NOT tclap_POPULATED)
    FetchContent_Populate(TCLAP)
endif(NOT tclap_POPULATED)
set(TCLAP_SOURCE_DIR ${tclap_SOURCE_DIR})

add_library(dist_matmul_v2
    STATIC
        data.h
        graphs.h
        states.h
        tasks.h
        utility.h
        matrix_utility.h
        mmd.h
)
set_target_properties(dist_matmul_v2 PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(dist_matmul_v2
    PUBLIC
        HH_USE_CUDA
        HH_ENABLE_CHECK_CUDA
        MMD_ENABLE_CHECK_MPI
)
target_include_directories(dist_matmul_v2
    PUBLIC
        ${Hedgehog_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TCLAP_SOURCE_DIR}/include/
)
target_link_libraries(dist_matmul_v2
    PUBLIC
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${Hedgehog_LIBRARIES}
        MPI::MPI_CXX
        ${OpenBLAS_LIBRARIES}
)

add_executable(demo_v2 demo_v2.cpp)
target_link_libraries(demo_v2
    PUBLIC
        dist_matmul_v2
)

add_executable(demo_v2_tests test.cpp)
target_link_libraries(demo_v2_tests
    PUBLIC
        dist_matmul_v2
)

add_executable(demo_v2_notch notchpeak_benchmark.cpp)
target_link_libraries(demo_v2_notch
    PUBLIC
        dist_matmul_v2
)