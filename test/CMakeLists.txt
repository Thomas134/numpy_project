cmake_minimum_required(VERSION 3.28)

project(NumpyTestProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(numpy_tests
    test/test_logical.hpp
    test/test_apply.hpp
    test/test_basic_property.hpp
    test/test_matrix_operations.hpp
    test/test_shift.hpp
    test/test_sort.hpp
    test/test_math.hpp
)

target_link_libraries(numpy_tests ${GTEST_LIBRARIES} pthread)

enable_testing()
add_test(NAME numpy_tests COMMAND numpy_tests)