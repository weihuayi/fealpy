cmake_minimum_required(VERSION 2.8)

project(p4test)

set(P4EST_DIR /home/why/local)

include_directories(${P4EST_DIR}/include)
link_directories(${P4EST_DIR}/lib)

find_package(MPI)
if(MPI_C_FOUND)
    include_directories(${MPI_C_INCLUDE_PATH})
    link_directories(${MPI_C_LINK_FLAG})
endif()

set(SC_LIBS sc)
set(P4EST_LIBS p4est sc ${MPI_CXX_LIBRARIES} gomp lapack cblas f77blas atlas z m)

add_subdirectory(test)

