
cmake_minimum_required(VERSION 2.8.11)
project(hello_boost_python)

ENABLE_TESTING()

FIND_PACKAGE(PythonInterp 3)
FIND_PACKAGE(PythonLibs 3)
INCLUDE_DIRECTORIES("/home/why/software/boost/1_63_0/include" ${PYTHON_INCLUDE_DIRS})
link_directories("/home/why/software/boost/1_63_0/lib")
LINK_LIBRARIES(boost_python ${PYTHON_LIBRARIES}) # Deprecated but so convenient!
set(LIBS boost_python boost_numpy ${PYTHON_LIBRARIES})


message(STATUS "${PYTHON_LIBRARIES}")
PYTHON_ADD_MODULE(boostpy boostpy.cpp)
target_link_libraries(boostpy ${LIBS})
FILE(COPY hello.py DESTINATION .)
ADD_TEST(HelloWorld hello.py)
