
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <iostream>

namespace p = boost::python;
namespace np = boost::python::numpy;

char const* greet()
{
   return "hello, world";
}

p::tuple fun(np::ndarray & B, np::ndarray & D, np::ndarray & cell2dof, np::ndarray & cell2dofLocation)
{
    std::cout << "Original array:\n" << p::extract<char const *>(p::str(D)) << std::endl;
    p::tuple r =  p::make_tuple(3, 3);
    return r;
}

BOOST_PYTHON_MODULE(hello)
{
    np::initialize();
    p::def("fun", fun);
    p::def("greet", greet);
}
