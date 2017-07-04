
PYTHON_ADD_MODULE(pyparmetis pyparmetis.cpp)
target_link_libraries(pyparmetis parmetis metis boost_python boost_numpy ${MPI_CXX_LIBRARIES})
