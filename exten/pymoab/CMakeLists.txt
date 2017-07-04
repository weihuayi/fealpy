
if(HAVE_MOAB)
    PYTHON_ADD_MODULE(pymoab pymoab.cpp)
    target_link_libraries(pymoab  ${MOAB_LIBRARIES})
endif()
