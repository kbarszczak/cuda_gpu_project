cmake_minimum_required(VERSION 3.21)
project(game_of_life CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(game_of_life main.cu)

set_target_properties(game_of_life PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
