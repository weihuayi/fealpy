
cmake_minimum_required(VERSION 2.8.11)
project(cgal_app)

set (CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
  message (STATUS "No build type selected, default to Release")
  set (CMAKE_BUILD_TYPE "Release")
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake/modules)

# find Geogram 
find_package(Geogram REQUIRED)
message(STATUS "${GEOGRAM_INCLUDE_DIR}")

set(GEOGRAM_LIBRAIES_DIR /usr/local/lib)

include_directories(${GEOGRAM_INCLUDE_DIR})
link_directories(${GEOGRAM_LIBRAIES_DIR})

# find cgal 
find_package(CGAL QUIET)
message (STATUS "No build type selected, default to Release")
include(${CGAL_USE_FILE})

add_subdirectory(test)
