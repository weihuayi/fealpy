#include <boost/python.hpp> 
#include "moab/Interface.hpp"
using namespace boost::python;
using namespace moab;

BOOST_PYTHON_MODULE(classes)
{
    class_<Interface>("Moab") 
        .def("load_file", &Interface::load_file)
    ;
};
