cmake_minimum_required(VERSION 3.10)
project(calc2)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(project)
add_executable(calc2 project/main.cpp project/system_solver.h project/utils.h project/matrix_utils.h)