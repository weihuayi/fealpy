#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <parmetis.h>
#include <mpi4py/mpi4py.h>
#include <iostream>

namespace p = boost::python;
namespace np = boost::python::numpy;

static void sayhello(MPI_Comm comm)
{
  if (comm == MPI_COMM_NULL) {
    std::cout << "You passed MPI_COMM_NULL !!!" << std::endl;
    return;
  }
  int size;
  MPI_Comm_size(comm, &size);
  int rank;
  MPI_Comm_rank(comm, &rank);
  int plen; char pname[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(pname, &plen);
  std::cout <<
    "Hello, World! " <<
    "I am process "  << rank  <<
    " of "           << size  <<
    " on  "          << pname <<
    "."              << std::endl;
}

static void hw_sayhello(np::ndarray a, p::object b, p::object py_comm)
{
  PyObject* py_obj = py_comm.ptr();
  MPI_Comm *comm_p = PyMPIComm_Get(py_obj);
  if (comm_p == NULL) p::throw_error_already_set();
  sayhello(*comm_p);

  std::string object_classname = boost::python::extract<std::string>(a.attr("__class__").attr("__name__"));
  std::cout<<"this is an Object: "<<object_classname<<std::endl;

  idx_t edgecut = 1;
  p::tuple shape = p::make_tuple(a.shape(0)-1);
  np::dtype dtype = np::dtype::get_builtin<idx_t>();
  np::ndarray part = np::zeros(shape, dtype);

  idx_t pb = p::extract<idx_t>(b);
  std::cout<<pb<< std::endl;
  std::cout<<sizeof(idx_t)<<std::endl;
  std::cout<<sizeof(int)<<std::endl;
  return;
}


static p::object wrap_part_mesh(
        p::object n, 
        const np::ndarray & elemdist, //This array describes how the elements of the mesh are distributed among the processors
        const np::ndarray & eptr,
        const np::ndarray & eind,// These arrays specifies the elements that are stored locally at each processor
        p::object  py_comm
        )
        
{
    idx_t nparts = p::extract<idx_t>(n);
    PyObject* py_obj = py_comm.ptr();
    MPI_Comm *comm = PyMPIComm_Get(py_obj);


    idx_t *pelmdist = reinterpret_cast<idx_t*>(elemdist.get_data());
    idx_t *peptr = reinterpret_cast<idx_t*>(eptr.get_data());
    idx_t *peind = reinterpret_cast<idx_t*>(eind.get_data());
    
    idx_t *elmwgt = NULL;
    idx_t wgtflag = 0;
    idx_t numflag = 0;
    idx_t ncon = 1;
    idx_t ncommonnodes = 2;
    idx_t options[3] = {1, 1, 0};

    p::tuple shape = p::make_tuple(eptr.shape(0) - 1);
    np::dtype dtype = np::dtype::get_builtin<idx_t>();
    np::ndarray part = np::zeros(shape, dtype);
    idx_t edgecut=0;

    idx_t *ppart = reinterpret_cast<idx_t*>(part.get_data());

    std::cout << nparts << std::endl;

    std::vector<real_t> tpwgts(ncon*nparts, 1.0/nparts);
    real_t ubvec = 1.05;

    int r = ParMETIS_V3_PartMeshKway(pelmdist, peptr, peind, NULL,
            &wgtflag, &numflag, &ncon, &ncommonnodes, &nparts, 
            &tpwgts[0], &ubvec, options, &edgecut, ppart, comm);

    return p::make_tuple(edgecut, part);
}


BOOST_PYTHON_MODULE(pyparmetis)
{
  if (import_mpi4py() < 0) return; /* Python 2.X */
  np::initialize();
  def("part_mesh", wrap_part_mesh);
  def("sayhello", hw_sayhello);
}

//int __cdecl ParMETIS_V3_PartMeshKway(
//             idx_t *elmdist, idx_t *eptr, idx_t *eind, idx_t *elmwgt, 
//	     idx_t *wgtflag, idx_t *numflag, idx_t *ncon, idx_t *ncommonnodes, idx_t *nparts, 
//	     real_t *tpwgts, real_t *ubvec, idx_t *options, idx_t *edgecut, idx_t *part, 
//	     MPI_Comm *comm);
//        const p::object & elmwgt, //This array stores the weights of the elements.
//        const int & wgtflag, //0 No weights (elmwgt is NULL);2 Weights on the vertices only.
//        const int & numflag,//0 C-style numbering that starts from 0;1 Fortran-style numbering that starts from 1.
//        const int & ncon, //This is used to specify the number of weights that each vertex has.
//        const int & ncommonnodes, //This parameter determines the degree of connectivity among the vertices in the dual graph.
//        const p::object & nparts, //This is used to specify the number of sub-domains that are desired
//        const p::object & tpwgts, //An array of size ncon × nparts that is used to specify the fraction of vertex weight 
//        const p::object & ubvec, //
//        const p::object & options,
