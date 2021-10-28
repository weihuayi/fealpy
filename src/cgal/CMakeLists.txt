find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})

pybind11_add_module(cgal cgal.cpp)
target_link_libraries(cgal PRIVATE ${CGAL_LIBRARIES})
set_target_properties(cgal PROPERTIES OUTPUT_NAME fealpy/cgal)
