#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <array>
#include <tuple>

namespace py = pybind11;

std::tuple <py::array_t<double>, py::array_t<int> > generate_surface_mesh()
{

}

int add(int i, int j)
{
    return i + j;
}

PYBIND11_MODULE(fealpy_extent, m){
    m.doc() = "This is a module extent of fealpy package!";
    m.def("add", &add,  "A function which adds two numbers");
}
