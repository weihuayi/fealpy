
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <iostream>

namespace p = boost::python;
namespace np = boost::python::numpy;

char const* greet()
{
   return "hello, world";
}

p::tuple vem_stiff_matrix_1(np::ndarray & B, np::ndarray & D, np::ndarray & cell2dof, np::ndarray & cell2dofLocation)
{
    int N = cell2dofLocation.shape(0) - 1;
    p::tuple shape = p::make_tuple(N);
    np::dtype dt = np::dtype::get_builtin<int>();

    np::ndarray NDof = np::zeros(shape, dt); 
    for(int i = 0; i < N; i++)
        NDof[i] = cell2dofLocation[i+1] - cell2dofLocation[i];

    int l = 0;
    for(int i = 0; i < N; i++)
        l = l + NDof[i]*NDof[i];

    std::cout << l << std::endl;
    p::tuple r =  p::make_tuple(3, 3);
    return r;
}

BOOST_PYTHON_MODULE(hello)
{
    np::initialize();
    p::def("vem_stiff_matrix_1", vem_stiff_matrix_1);
    p::def("greet", greet);
}
