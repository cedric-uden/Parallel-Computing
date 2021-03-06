cmake_minimum_required(VERSION 3.16)
# define the first project name (for the C project)
project(parallel_computing_in_C)


# create the target dir
file(MAKE_DIRECTORY "cmake-build-debug")
# set the directory to output final binaries to run
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


# ----------------------------------------------------------------------------------------------------------------------

# C program files settings
# project(parallel_computing_in_C)

set(CMAKE_C_STANDARD 11)
add_executable(${PROJECT_NAME} hello_world/hello_world_in_c.c)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "hello_world_in_c")

# ----------------------------------------------------------------------------------------------------------------------

# C++ program file settings
project(parallel_computing_in_CPP)  # overwrite the project for the c++ project

set(CMAKE_CXX_STANDARD 11)
add_executable(${PROJECT_NAME} hello_world/hello_world_in_cpp.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "hello_world_in_cpp")

# ----------------------------------------------------------------------------------------------------------------------

# CUDA program file settings
project(parallel_computing_in_cuda LANGUAGES CUDA)

include_directories(runtime_analysis)
set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)
add_executable(${PROJECT_NAME} hello_world/hello_world_in_cuda.cu runtime_analysis/Runtime_Analysis.cpp runtime_analysis/Runtime_Analysis.h)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "hello_world_in_cuda")

# ----------------------------------------------------------------------------------------------------------------------

# CUDA program file settings
project(matrix_multiply LANGUAGES CUDA)

include_directories(runtime_analysis)
include_directories(cuda_examples)
set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)
add_executable(${PROJECT_NAME} runtime_analysis/Runtime_Analysis.cpp runtime_analysis/Runtime_Analysis.h cuda_examples/matrix_multiply.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "matrix_multiply")

# ----------------------------------------------------------------------------------------------------------------------
