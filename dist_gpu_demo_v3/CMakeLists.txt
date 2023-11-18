cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

find_package(Hedgehog REQUIRED)
find_package(CUDA     REQUIRED)
find_package(NVTX     REQUIRED)
find_package(MPI      REQUIRED)

add_library(v3_lib
    STATIC
        data.h
        graphs.h
        states.h
        tasks.h
        utility.h
        mmd.h
)
set_target_properties(v3_lib PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(v3_lib
    PUBLIC
        HH_USE_CUDA
        HH_ENABLE_CHECK_CUDA
        MMD_ENABLE_CHECK_MPI
)
target_include_directories(v3_lib
    PUBLIC
        ${Hedgehog_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TCLAP_SOURCE_DIR}/include/
        ../include/
)
target_link_libraries(v3_lib
    PUBLIC
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${Hedgehog_LIBRARIES}
        MPI::MPI_CXX
        ${OpenBLAS_LIBRARIES}
)

add_executable(v3_demo1 v3_demo1.cpp)
target_link_libraries(v3_demo1
    PUBLIC
        v3_lib
)

add_executable(v3_benchmark1 v3_benchmark1.cpp)
target_link_libraries(v3_benchmark1
    PUBLIC
        v3_lib
)
