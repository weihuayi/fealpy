macro(add_example filename standard local_libs)
    add_executable(${filename} ${filename}.cpp)
    set_target_properties(${filename} PROPERTIES
        CXX_STANDARD ${standard} 
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        )
    target_link_libraries(${filename} ${local_libs})
endmacro(add_example)


add_executable(test_csr_matrix test_csr_matrix.cpp)
