cmake_minimum_required(VERSION 3.27)

project(numpy_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
else()
    message(FATAL_ERROR "BLAS library not found.")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386|i686")
    find_package(xsimd REQUIRED)
    if(xsimd_FOUND)
        message(STATUS "xsimd directory found: ${xsimd_DIR}")
    else()
        message(FATAL_ERROR "xsimd directory not found.")
    endif()
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP CXX found: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP CXX not found.")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/utils
    ${PROJECT_SOURCE_DIR}/include/data_structure
)

file(READ "/etc/os-release" OS_RELEASE_CONTENTS)
if(OS_RELEASE_CONTENTS MATCHES "ID=ubuntu")
    add_compile_definitions(__UBUNTU__)
elseif(OS_RELEASE_CONTENTS MATCHES "ID=fedora")
    add_compile_definitions(__FEDORA__)
endif()

file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/include/*.cpp
    ${PROJECT_SOURCE_DIR}/include/utils/*.cpp
    ${PROJECT_SOURCE_DIR}/include/data_structure/*.cpp
)

add_library(numpycpp STATIC ${SOURCES})
target_link_libraries(numpycpp PRIVATE ${BLAS_LIBRARIES})

install(TARGETS numpycpp
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include/numpy
    FILES_MATCHING PATTERN "*.cpp"
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386|i686")
    add_definitions(-mavx2 -fopenmp -O3)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    add_definitions(-fopenmp -march=rv64gcv -O3)
endif()

add_subdirectory(test)
