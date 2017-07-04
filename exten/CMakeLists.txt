cmake_minimum_required(VERSION 2.8.3)

FIND_PACKAGE(PythonInterp 3)
FIND_PACKAGE(PythonLibs 3)
ENABLE_TESTING()
INCLUDE_DIRECTORIES("/home/why/software/boost/1_63_0/include" ${PYTHON_INCLUDE_DIRS})
link_directories("/home/why/software/boost/1_63_0/lib")
LINK_LIBRARIES(boost_python ${PYTHON_LIBRARIES}) # Deprecated but so convenient!
message(STATUS "${PYTHON_LIBRARIES}")

include_directories("/home/why/software/parmetis/4.0/include")
link_directories("/home/why/software/parmetis/4.0/lib")

include_directories("/usr/local/lib/python3.5/dist-packages/mpi4py/include")
link_directories("/usr/local/lib/python3.5/dist-packages/mpi4py")

if(NOT SIGMA_DIR)
    set(SIGMA_DIR "/usr/local")
endif()

message(STATUS "${SIGMA_DIR}")

site_name(HOST_NAME)
if( HOST_NAME STREQUAL "why-XPS-15-9530")
    message(STATUS ${HOST_NAME})
    set(SIGMA_DIR "/home/why/software/sigma/1.2.1")
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
    message(STATUS "${MOAB_LIBRARIES}")
    set(HAVE_MOAB True)
else()
    set(HAVE_MOAB False)
    message(STATUS "config package without MOAB!")
endif()

#add_subdirectory(HelloWorld)
#add_subdirectory(tdquadrule)
#add_subdirectory(pymetis)
add_subdirectory(pyparmetis)
#add_subdirectory(pymoab)
