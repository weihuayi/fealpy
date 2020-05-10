#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(fealpy_extent, m){
    m.doc() = "My first pybind11 example!";
    m.def("add", &add,  "A function which adds two numbers");
}
