
macro(add_example filename standard local_libs)
    add_executable(${filename} ${filename}.cpp)
    set_target_properties(${filename} PROPERTIES
        CXX_STANDARD ${standard} 
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        )
    target_link_libraries(${filename} ${local_libs})
endmacro(add_example)

if(HAVE_MOAB AND HAVE_MPI AND HAVE_CGAL)
    set(LOCAL_LIBS "${MOAB_LIBRARIES} ${CGAL_LIBRARIES} ${MPI_CXX_LIBRARIES}")
    add_example(Interface2d 14 ${LOCAL_LIBS})
endif()







