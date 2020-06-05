cmake_minimum_required (VERSION 2.6)

project (DETRI2)

# The version number.
set (DETRI2_VERSION_MAJOR 1)
set (DETRI2_VERSION_MINOR 5)
set (DETRI2_PATCH_VERSION 0)

set (DETRI2_VERSION "${DETRI2_MAJOR_VERSION}.${DETRI2_MINOR_VERSION}")

#find_package(GMP REQUIRED)
add_definitions(-DUSING_GMP)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
#include_directories(/opt/local/include)
#link_directories(/opt/local/lib)

add_executable(detri2 
       detri2.cpp
       flips.cpp
       pred3d.cpp
       io.cpp
       sort.cpp
       delaunay.cpp
       constrained.cpp
       refine.cpp
       voronoi.cpp
       #meshadapt.cpp
       adapt.cpp
       metric.cpp
       main.cpp)

target_link_libraries(detri2 gmp gmpxx)
