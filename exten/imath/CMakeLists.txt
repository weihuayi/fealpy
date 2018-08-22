cmake_minimum_required(VERSION 2.8)

project(imath)

set(CMAKE_BUILD_TYPE Debug)
set(IMATH_VERSION_MAJOR 0)
set(IMATH_VERSION_MINOR 0)

option(IMATH_ENABLE_TESTING "Enable Testing"                             ON)


set(IMATH_LIB "imath")

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_BINARY_DIR}/lib)

add_subdirectory(include) 
add_subdirectory(example)
add_subdirectory(doc)

if(IMATH_ENABLE_TESTING)
      enable_testing()
      add_subdirectory(test)
endif()

set(DATA_DIR "${PROJECT_SOURCE_DIR}/data/")
