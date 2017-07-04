macro(add_example filename standard local_libs)
    add_executable(${filename} ${filename}.cpp)
    set_target_properties(${filename} PROPERTIES
        CXX_STANDARD ${standard} 
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        )
    target_link_libraries(${filename} ${local_libs})
endmacro(add_example)

if( HAVE_CGAL AND HAVE_MOAB)

endif()


if( HAVE_CGAL )

endif()

if (HAVE_MOAB AND HAVE_CGAL)
endif()
