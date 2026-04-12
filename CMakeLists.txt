cmake_minimum_required(VERSION 3.20)

project(sparse_solver_gym VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

set(JSON_BuildTests OFF)
set(JSON_Install OFF)
add_subdirectory(third_party/json-develop)

add_subdirectory(src)
add_subdirectory(test)
