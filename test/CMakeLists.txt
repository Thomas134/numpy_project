cmake_minimum_required(VERSION 3.27)

project(NumpyTestProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/../include/utils
    ${PROJECT_SOURCE_DIR}/../include/data_structure
)

file(GLOB_RECURSE TEST_SOURCES
    ${PROJECT_SOURCE_DIR}/test_*.hpp
    ${PROJECT_SOURCE_DIR}/run_all_tests.cpp
)

add_executable(run_all_tests ${TEST_SOURCES})

target_link_libraries(run_all_tests ${GTEST_LIBRARIES} ${BLAS_LIBRARIES})

target_link_libraries(run_all_tests numpycpp)

