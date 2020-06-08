#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <array>
#include <tuple>

#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Implicit_surface_3.h>

namespace py = pybind11;

typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;

typedef Tr::Geom_traits GT;
typedef GT::Sphere_3 Sphere_3;
typedef GT::Point_3 Point_3;
typedef GT::FT FT;

typedef FT (*Function)(Point_3);

typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;

FT sphere_function (Point_3 p) {
  const FT x2=p.x()*p.x(), y2=p.y()*p.y(), z2=p.z()*p.z();
  return x2+y2+z2-1;
}

std::tuple< py::array_t<double>, py::array_t<int> >  generate_surface_mesh()
{
    typedef typename C2t3::Triangulation Tr;
    typedef typename Tr::Vertex_handle Vertex_handle;

    Tr tr; 
    C2t3 c2t3(tr); 
    Surface_3 surface(sphere_function, Sphere_3(CGAL::ORIGIN, 2.)); 
    CGAL::Surface_mesh_default_criteria_3<Tr> criteria(30.,  // angular bound
                                                     0.1,  // radius bound
                                                     0.1); // distance bound
    CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Non_manifold_tag());
    int NN = tr.number_of_vertices();
    int NC = c2t3.number_of_facets();

    auto node = py::array_t<double>(3*NN);
    node.resize({NN, 3});
    auto cell = py::array_t<int>(3*NC);
    cell.resize({NC, 3});


    auto m_node = node.mutable_unchecked<2>();
    std::map<Vertex_handle, int> V;
    int i = 0;
    for(auto vit = tr.finite_vertices_begin();
      vit != tr.finite_vertices_end(); ++vit)
    {
        V[vit] = i;
        auto p = vit->point();
        m_node(i, 0) = p[0];
        m_node(i, 1) = p[1];
        m_node(i, 2) = p[2];
        i++;
    } 

    auto m_cell = cell.mutable_unchecked<2>();
    i = 0;
    for(auto fit = tr.finite_facets_begin(); 
            fit != tr.finite_facets_end(); ++fit)
    {
        const auto face = fit->first;
        const int& index = fit->second;
        if (face->is_facet_on_surface(index)==true)
        {
            m_cell(i, 0) = V[face->vertex(tr.vertex_triple_index(index, 0))];
            m_cell(i, 1) = V[face->vertex(tr.vertex_triple_index(index, 1))];
            m_cell(i, 2) = V[face->vertex(tr.vertex_triple_index(index, 2))];
            i++;
        }
    }

    std::cout << "Final number of points: " << tr.number_of_vertices() << "\n";
    return std::make_tuple(node, cell);
}

int add(int i, int j)
{
    return i + j;
}

PYBIND11_MODULE(cgal, m){
    m.doc() = "This is a module extent of fealpy package!";
    m.def("add", &add,  "A function which adds two numbers");
    m.def("generate_surface_mesh", 
        &generate_surface_mesh, 
        "A function which generate surface mesh!"
        );
}
