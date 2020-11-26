cmake_minimum_required(VERSION 3.10)
project(calc4)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(project)
add_executable(calc4 project/main.cpp project/system_solver.h project/utils.h project/matrix_utils.h project/eigen_solver.h project/stat_holder.h)