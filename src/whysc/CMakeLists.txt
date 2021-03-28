cmake_minimum_required(VERSION 3.0)

project(WHYSC)

include_directories(${PROJECT_SOURCE_DIR}/include)
message(STATUS "${PROJECT_SOURCE_DIR}")
add_executable(hello hello.cpp)

add_subdirectory(test)
