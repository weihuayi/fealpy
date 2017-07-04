#include <boost/python/numpy.hpp>
#include <metis.h>
#include <iostream>

namespace p = boost::python;
namespace np = boost::python::numpy;

p::object wrap_part_graph(
        int nparts,
        const p::object & adj,
        const p::object & adjLocation,
        const p::object & vertWeight,
        const p::object & adjWeight,
        bool recursive)
{
    int numVert = p::len(adjLocation) - 1;
    std::cout << numVert << std::endl;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    return p::object();
}

BOOST_PYTHON_MODULE(pymetis)
{
    def("part_graph", wrap_part_graph);

    /*! Return codes */
    p::enum_<rstatus_et>("return_status")
      .value("OK", METIS_OK)                 /*!< Returned normally */
      .value("INPUT", METIS_ERROR_INPUT)     /*!< Returned due to erroneous inputs and/or options */
      .value("MEMORY", METIS_ERROR_MEMORY)   /*!< Returned due to insufficient memory */
      .value("ERROR", METIS_ERROR)           /*!< Some other errors */
      ; 

    /*! Operation type codes */
    p::enum_<moptype_et>("operation_type")
      .value("PMETIS", METIS_OP_PMETIS)       
      .value("KMETIS", METIS_OP_KMETIS)
      .value("OMETIS", METIS_OP_OMETIS)
      ;

    /*! Options codes (i.e., options[]) */
    p::enum_<moptions_et>("options")
      .value("PTYPE", METIS_OPTION_PTYPE)
      .value("OBJTYPE", METIS_OPTION_OBJTYPE)
      .value("CTYPE", METIS_OPTION_CTYPE)
      .value("IPTYPE", METIS_OPTION_IPTYPE)
      .value("RTYPE", METIS_OPTION_RTYPE)
      .value("DBGLVL", METIS_OPTION_DBGLVL)
      .value("NITER", METIS_OPTION_NITER)
      .value("NCUTS", METIS_OPTION_NCUTS)
      .value("SEED", METIS_OPTION_SEED)
      .value("NO2HOP", METIS_OPTION_NO2HOP)
      .value("MINCONN", METIS_OPTION_MINCONN)
      .value("CONTIG", METIS_OPTION_CONTIG)
      .value("COMPRESS", METIS_OPTION_COMPRESS)
      .value("CCORDER", METIS_OPTION_CCORDER)
      .value("PFACTOR", METIS_OPTION_PFACTOR)
      .value("NSEPS", METIS_OPTION_NSEPS)
      .value("UFACTOR", METIS_OPTION_UFACTOR)
      .value("NUMBERING", METIS_OPTION_NUMBERING)
      .value("HELP", METIS_OPTION_HELP) /* Used for command-line parameter purposes */
      .value("TPWGTS", METIS_OPTION_TPWGTS)
      .value("NCOMMON", METIS_OPTION_NCOMMON)
      .value("NOOUTPUT", METIS_OPTION_NOOUTPUT)
      .value("BALANCE", METIS_OPTION_BALANCE)
      .value("GTYPE", METIS_OPTION_GTYPE)
      .value("UBVEC", METIS_OPTION_UBVEC)
      ;

    /*! Partitioning Schemes */
    p::enum_<mptype_et>("part_type")
      .value("RB", METIS_PTYPE_RB) 
      .value("KWAY", METIS_PTYPE_KWAY)
      ;

    /*! Graph types for meshes */
    p::enum_<mgtype_et>("graph_type")
      .value("DUAL", METIS_GTYPE_DUAL)
      .value("NODAL", METIS_GTYPE_NODAL)               
      ;

    /*! Coarsening Schemes */
    p::enum_<mctype_et>("coarsen_type")
      .value("RM", METIS_CTYPE_RM)
      .value("SHEM", METIS_CTYPE_SHEM)
      ;

    /*! Initial partitioning schemes */
    p::enum_<miptype_et>("inital_par_type")
      .value("GROW", METIS_IPTYPE_GROW)
      .value("RANDOM", METIS_IPTYPE_RANDOM)
      .value("EDGE", METIS_IPTYPE_EDGE)
      .value("NODE", METIS_IPTYPE_NODE)
      .value("METISRB", METIS_IPTYPE_METISRB)
      ;

    /*! Refinement schemes */
    p::enum_<mrtype_et>("refinement_type")
      .value("FM", METIS_RTYPE_FM)
      .value("GREEDY", METIS_RTYPE_GREEDY)
      .value("SEP2SIDED", METIS_RTYPE_SEP2SIDED)
      .value("SEP1SIDED", METIS_RTYPE_SEP1SIDED)
      ;

    /*! Debug Levels */
    p::enum_<mdbglvl_et>("debug")
      .value("INFO", METIS_DBG_INFO)  /*!< Shows various diagnostic messages */
      .value("TIME", METIS_DBG_TIME)       /*!< Perform timing analysis */
      .value("COARSEN", METIS_DBG_COARSEN)    /*!< Show the coarsening progress */
      .value("REFINE", METIS_DBG_REFINE)     /*!< Show the refinement progress */
      .value("MOVEINFO", METIS_DBG_MOVEINFO)   /*!< Show info on vertex moves during refinement */
      .value("CONNINFO", METIS_DBG_CONNINFO)   /*!< Show info on minimization of subdomain connectivity */
      .value("CONTIGINFO", METIS_DBG_CONTIGINFO) /*!< Show info on elimination of connected components */ 
      .value("MEMORY", METIS_DBG_MEMORY)     /*!< Show info related to wspace allocation */
      ;
    
    /* Types of objectives */
    p::enum_<mobjtype_et>("objectives_type")
      .value("CUT", METIS_OBJTYPE_CUT)
      .value("VOL", METIS_OBJTYPE_VOL)
      .value("NODE", METIS_OBJTYPE_NODE)
      ;
}
