cmake_minimum_required(VERSION 3.21)
project(cuda_gpu_project CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_gpu_project main.cu)

set_target_properties(cuda_gpu_project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
