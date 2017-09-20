add_executable(curve_intersections_test curve_intersections_test.cpp)
target_link_libraries(curve_intersections_test  ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})

add_executable(mesh_class mesh_class.cpp)
target_link_libraries(mesh_class  ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})

add_executable(cgal_demo cgal_demo.cpp)
target_link_libraries(cgal_demo ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES}
    Geogram::geogram_gfx Geogram::geogram)
