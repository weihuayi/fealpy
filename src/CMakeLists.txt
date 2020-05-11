find_package(pybind11 REQUIRED)

find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})

file(GLOB SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
pybind11_add_module(fealpy_extent ${SRCS})

target_link_libraries(fealpy_extent PRIVATE ${CGAL_LIBRARIES})
