cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

find_package(Hedgehog REQUIRED)
find_package(MPI      REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)

add_library(dist_simd_lib
    STATIC
        tasks.h
        utility.h
        mmd.h
)
set_target_properties(dist_simd_lib PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(dist_simd_lib PUBLIC MMD_ENABLE_CHECK_MPI)
target_include_directories(dist_simd_lib
    PUBLIC
        ../include/
        ${Hedgehog_INCLUDE_DIRS}
        ${OpenBLAS_INCLUDE_DIRS}
        ${TCLAP_SOURCE_DIR}/include/
)
target_link_libraries(dist_simd_lib
    PUBLIC
        ${Hedgehog_LIBRARIES}
        MPI::MPI_CXX
        ${OpenBLAS_LIBRARIES}
)

add_executable(simd_demo1 simd_demo1.cpp)
target_link_libraries(simd_demo1 PUBLIC dist_simd_lib)
target_compile_options(simd_demo1 PRIVATE
    $<$<CONFIG:Debug>:-O0>
)

add_executable(simd_benchmark1 simd_benchmark1.cpp)
target_link_libraries(simd_benchmark1 PUBLIC dist_simd_lib)
