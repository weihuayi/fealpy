cmake_minimum_required(VERSION 2.8)

project(PSC)


set(CMAKE_BUILD_TYPE Debug)
set(PSC_VERSION_MAJOR 3)
set(PSC_VERSION_MINOR 0)

set(GCC_COVERAGE_COMPILE_FLAGS "-frounding-math")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

option(PSC_ENABLE_TESTING "Enable Testing"                             ON)

if(NOT SIGMA_DIR)
    set(SIGMA_DIR "/usr/local")
endif()

if(NOT CGAL_DIR)
    set(CGAL_DIR "/usr/local")
endif()

if(NOT EIGEN_DIR)
    set(EIGEN_DIR "/usr/local")
endif()

if(NOT TRILINOS_DIR)
    set(TRILINOS_DIR "/usr/local")
endif()

message(STATUS "${SIGMA_DIR} ${CGAL_DIR}")

site_name(HOST_NAME)
if( HOST_NAME STREQUAL "why-XPS-15-9530")
    message(STATUS ${HOST_NAME})
    set(SIGMA_DIR "/home/why/software/sigma/1.2.1")
    set(CGAL_DIR  "/home/why/software/cgal/4.8.1")
    set(EIGEN3_DIR "/home/why/software/eigen/3.2.8")
    set(TRILINOS_DIR "/home/why/software/trilinos/12.6.3")
endif()

# Eigen3

set(EIGEN3_FOUND True)
if(EIGEN3_FOUND)
    include_directories("${EIGEN3_DIR}/include/eigen3")
    set(HAVE_EIGEN3 True)
else()
    set(HAVE_EIGEN3 False)
endif()

# MPI
find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    set(HAVE_MPI True)
else()
    set(HAVE_MPI False)
endif()

# SIGMA

## MOAB
set(MOAB_DIR ${SIGMA_DIR}/lib)
set(MOAB_LIBRARIES_DIR ${MOAB_DIR}) 
find_package(MOAB)
if(MOAB_FOUND)
    include_directories(${MOAB_INCLUDE_DIRS})
    link_directories(${MOAB_LIBRARIES_DIR})
    set(HAVE_MOAB True)
else()
    set(HAVE_MOAB False)
    message(STATUS "config package without MOAB!")
endif()


# CGAL

set(CGAL_DIR "${CGAL_DIR}/lib/CGAL")
message(STATUS ${CGAL_DIR})
find_package(CGAL)
if(CGAL_FOUND)
    include_directories(${CGAL_INCLUDE_DIRS})
    link_directories(${CGAL_LIBRARIES_DIR})
    set(CGAL_LIBRARIES "-lCGAL -lCGAL_Core -lgmp -lmpfr")
    set(HAVE_CGAL True)
else()
    set(HAVE_CGAL False)
    message(STATUS "Configure package without CGAL!")
endif()


# Trilinos
# Tpetra
set(Tpetra_DIR "${TRILINOS_DIR}/lib/cmake/Tpetra")
message(STATUS ${Tpetra_DIR})
find_package(Tpetra)
if(Tpetra_FOUND)
    include_directories(${Tpetra_INCLUDE_DIRS})
    link_directories(${Tpetra_LIBRARY_DIRS})
    set(TPETRA_LIBRARIES ${Tpetra_LIBRARIES})
    set(HAVE_TPETRA True)
else()
    set(HAVE_TPETRA False)
    message(STATUS "Configure package without Tpetra!")
endif()

set(PSC_LIB "PSC")

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_BINARY_DIR}/lib)

# Add subdirectory
add_subdirectory(include) 
add_subdirectory(example)
add_subdirectory(doc)

# install(FILES COPYRIGHT README.md DESTINATION share/doc/PMOPT)
# install(FILES ${PROJECT_BINARY_DIR}/PMOPT_Config.h DESTINATION include)

if(PSC_ENABLE_TESTING)
      enable_testing()
      add_subdirectory(test)
endif()

set(DATA_DIR "${PROJECT_SOURCE_DIR}/data/")
#configure_file (
#"${PROJECT_SOURCE_DIR}/PSCConfig.h.in"
#"${PROJECT_BINARY_DIR}/PSCConfig.h")
