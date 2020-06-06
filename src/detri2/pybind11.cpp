#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <iterator>
#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <sstream>

#include "detri2.h"


using namespace detri2;

int generate_mesh(std::string & options)
{
    std::istringstream in(options);
    std::vector<std::string> parse;
    std::copy(
            std::istream_iterator<std::string>(in), 
            std::istream_iterator<std::string>(), 
            std::back_inserter(parse));

    std::vector<char *> cstrs;
    cstrs.reserve(parse.size());
    for (auto &s : parse) 
        cstrs.push_back(const_cast<char *>(s.c_str()));

    int argc = cstrs.size();
    char ** argv = cstrs.data();
    if (argc < 2) {
        printf("Usage: detri2 [-options] filename[.node, .ele]\n");
        return 0;
    }

    Triangulation *Tr = new Triangulation();

    // Read options.
    if (!Tr->parse_commands(argc, argv)) {
        // No input or wrong parameters.
        printf("Usage: detri2 [-options] filename[.node, .poly, .ele, .edge]\n");
        delete Tr;
        return 0;
    }

    // Read inputs.
    if (!Tr->read_mesh()) {
        printf("Failed to read input from file %s[.poly, .node, .ele, .edge]\n",
                Tr->io_infilename);
        delete Tr;
        return 0;
    }

    // Generate (constrained) (weighted) Delaunay triangulation.
    if (Tr->tr_tris == NULL) {
        if (Tr->incremental_delaunay()) {
            if (Tr->tr_segs != NULL) {
                Tr->recover_segments();
                Tr->set_subdomains();
            }
        } else {
            printf("Failed to create Delaunay (regular) triangulation.\n");
            delete Tr;
            return 0;
        }
    } else {
        Tr->reconstruct_mesh(1);
    }

    // Mesh refinement and adaptation.
    //if (Tr->tr_segs != NULL) {
    if (Tr->op_quality || (Tr->op_metric > 0)) {
        if (Tr->io_omtfilename[0] != '\0') {
            // A background mesh is supplied.
            Tr->OMT_domain = new Triangulation();
            int myargc = 2;
            char *myargv[2];
            myargv[0] = argv[0];
            myargv[1] = Tr->io_omtfilename;
            Tr->OMT_domain->parse_commands(myargc, myargv);
            Tr->OMT_domain->read_mesh();
            Tr->OMT_domain->reconstruct_mesh(0);
            Tr->op_metric = METRIC_Euclidean;
        } else {
            assert(Tr->OMT_domain == NULL);
        }

        if (Tr->op_metric) {
            Tr->set_vertex_metrics();
            Tr->coarsen_mesh();
        }

        Tr->delaunay_refinement();
    }

    // Mesh export (to files).
    if (Tr->tr_tris != NULL) {
        if (Tr->ct_exteriors > 0) { 
            Tr->remove_exteriors();
        }
        Tr->save_triangulation();
        if (Tr->io_outedges) {
            Tr->save_edges();
        }
    }

    Tr->mesh_statistics();

    delete Tr;
    return 1;
}

PYBIND11_MODULE(detri2, m){
    m.doc() = "This is a interface module to Detri2!";
    m.def("generate_mesh", 
        &generate_mesh, 
        "A function which generate 2d mesh!"
        );
}
