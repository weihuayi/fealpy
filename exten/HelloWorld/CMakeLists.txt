
PYTHON_ADD_MODULE(hello hello.cpp)
FILE(COPY hello.py DESTINATION .)
ADD_TEST(HelloWorld hello.py)
