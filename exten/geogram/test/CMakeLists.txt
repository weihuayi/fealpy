
add_executable(hello_geogram hello_geogram.cpp)
target_link_libraries(hello_geogram Geogram::geogram)

add_executable(hello_GLUP_app hello_GLUP_app.cpp)
target_link_libraries(hello_GLUP_app Geogram::geogram_gfx Geogram::geogram)

add_executable(cvtopt_2d_glup_app cvtopt_2d_glup_app.cpp)
target_link_libraries(cvtopt_2d_glup_app Geogram::geogram_gfx Geogram::geogram)

add_executable(domain_2d_glup_demo domain_2d_glup_demo.cpp)
target_link_libraries(domain_2d_glup_demo  Geogram::geogram_gfx Geogram::geogram)

add_executable(CVTMesh CVTMesh.cpp)
target_link_libraries(CVTMesh  Geogram::geogram_gfx Geogram::geogram)

include(./opengl.cmake)
add_executable(GLUP_basic_example GLUP_basic_example.cpp)
target_link_libraries(GLUP_basic_example geogram_gfx geogram ${GLFW_LIBRARIES})
