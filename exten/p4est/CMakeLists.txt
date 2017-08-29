

cmake_minimum_required(VERSION 2.8)

project(p4test)

set(P4EST_DIR /home/why/software/p4est/serial)
include_directories(${P4EST_DIR}/include)
link_directories(${P4EST_DIR}/lib)

set(SC_LIBS sc)
set(P4EST_LIBS p4est ${SC_LIBS} gomp pthread lapack blas z m)

add_subdirectory(test)

