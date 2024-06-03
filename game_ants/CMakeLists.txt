cmake_minimum_required(VERSION 3.21)
project(game_ants CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(game_ants main.cu)

set_target_properties(game_ants PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
