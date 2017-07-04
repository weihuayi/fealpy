#ifndef Mesh_h
#define Mesh_h

#include "moab/Core.hpp"
#include "Geometry/Geometry_kernel.h"
#include <cassert>
#include <memory>
#include <cmath>

namespace iMath {
namespace fem {

using namespace moab;

template<class GK = iMath::Geometry_kernel<> > 
class Triangle {
public:
    typedef typename GK::Point_3 Point_3;
    typedef typename GK::Vector_3 Vector_3;

public:
    static void compute_gradient_of_lambda(
            std::vector<Point_3> & pts, 
            std::vector<Vector_3> & g,
            double & area
            ) {

        Vector_3 v0 = pts[2] - pts[1];
        Vector_3 v1 = pts[0] - pts[2];
        Vector_3 v2 = pts[1] - pts[0]; 

        Vector_3 v = GK::cross_product(v0, v1);
        double len = std::sqrt(v.squared_length());
        area = len/2.0;
        v = v/len;
        g.resize(3);
        g[0] = GK::cross_product(v, v0)/len;
        g[1] = GK::cross_product(v, v1)/len;
        g[2] = GK::cross_product(v, v2)/len;
    }

    static double area(std::vector<Point_3> & pts) {
        Vector_3 v0 = pts[2] - pts[1];
        Vector_3 v1 = pts[0] - pts[2];
        Vector_3 v2 = pts[1] - pts[0]; 

        Vector_3 v = GK::cross_product(v0, v1);
        double len = std::sqrt(v.squared_length());
        return len/2.0;
    }
};




template<class GK = iMath::Geometry_kernel<> >
class Mesh {
public:
    typedef typename GK::Point_2 Point_2;
    typedef typename GK::Vector_2 Vector_2;
    typedef typename GK::Point_3 Point_3;
    typedef typename GK::Vector_3 Vector_3;

    enum class Mesh_type {
        PLANAR,
        SURFACE,
        SOLID
    };
    
public:
    Mesh(std::shared_ptr<Interface> mb, EntityHandle mesh_set, Mesh_type mesh_type):
        _mb(mb), _mesh_type(mesh_type), _mesh_set(mesh_set) {
    }

    std::shared_ptr<Interface>  get_moab_interface() {
        return _mb;
    }

    //Tags 

    Tag get_global_id_tag() {
        Tag g_id;
        auto rval = _mb->tag_get_handle("GLOBAL_ID", g_id);
        return g_id;
    }
    
    Tag get_material_tag() {
        Tag m_id;
        auto rval = _mb->tag_get_handle("MATERIAL_SET", m_id);
        return m_id;
    }

    Tag get_neumann_set_tag() {
        Tag n_id;
        auto rval = _mb->tag_get_handle("NEUMANN_SET", n_id);
        return n_id;
    }

    Tag get_dirichlet_set_tag() {
        Tag d_id;
        auto rval = _mb->tag_get_handle("DIRICHLET_SET", d_id);
        return d_id;
    }

    Tag get_geom_dimension_tag() {
        Tag g_d_id;
        auto rval = _mb->tag_get_handle("GEOM_DIMENSION", g_d_id);
        return g_d_id;
    }

    Tag create_area_tag() {
        Tag area_id;
        double def_val = 0.0;
        int def_val_len = 1;
        auto rval = _mb->tag_get_handle("AREA", def_val_len,
                MB_TYPE_DOUBLE, area_id,
                MB_TAG_CREAT | MB_TAG_DENSE,
                &def_val);
        return area_id;
    }


    // MeshSet
    EntityHandle get_mesh_set() {
        return 0;
    }

    void set_mesh_set(EntityHandle set_h) {
        _mesh_set = set_h;
    }

    // Global id
    void set_vertex_global_id() {

        auto g_id = get_global_id_tag();

        Range vertices;
        auto rval = _mb->get_entities_by_dimension(_mesh_set, 0, vertices);
        auto i = 1; // the id begin from 1
        for(auto& v:vertices)
        {
            rval = _mb->tag_set_data(g_id, &v, 1, &i);
            i++;
        }
    }

    template<class Point>
    void get_corner_vertex_coords(EntityHandle h, vector<Point> & pts) {

        //Get the coordinates of three vertices of element h
        const EntityHandle * conn;
        int num = 0;
        auto rval = _mb->get_connectivity(h, conn, num);

        assert(num >= 1);
        pts.resize(num);

        int dim = get_geom_dimension();
        assert(dim == 3);

        double coords[dim*num];
        rval = _mb->get_coords(conn, num, coords);
        if (_mesh_type == Mesh_type::PLANAR) {
            assert(pts[0].dimension() == 2);
            for(int i = 0; i < num; i++)
                pts[i] = GK::point_2(coords + dim*(i-1));
        } else {
            assert(pts[0].dimension() == 3);
            for(int i = 0; i < num; i++)
                pts[i] = GK::point_3(coords + dim*(i-1));
        }
    }

    void get_corner_vertex_global_id(EntityHandle h, vector<int> & g_id) {

        const EntityHandle * conn;
        int num = 0;
        auto rval = _mb->get_connectivity(h, conn, num);
        g_id.resize(num);
        auto global_id = get_global_id_tag();
        rval = _mb->tag_get_data(global_id, conn, num, g_id.data());
    }
    
    void get_all_elems(Range & elems)
    {
        if ( _mesh_type == Mesh_type::PLANAR ||
                _mesh_type == Mesh_type::SURFACE ) {
            auto rval = _mb->get_entities_by_dimension(_mesh_set, 2, elems);
        } else if ( _mesh_type == Mesh_type::SOLID ) {
            auto rval = _mb->get_entities_by_dimension(_mesh_set, 3, elems);
        }
    }

    int get_geom_dimension() {
        int dim;
        auto rval = _mb->get_dimension(dim);
        return dim;
    }

    int get_number_of_vertices() {
        int num = 0;
        auto rval = _mb->get_number_entities_by_dimension(_mesh_set, 0, num);
        return num;
    }

    int get_number_of_edges() {
        int num = 0;
        auto rval = _mb->get_number_entities_by_dimension(_mesh_set, 1, num);
        return num;
    }

    int get_number_of_faces() {
        int num = 0;
        auto rval = _mb->get_number_entities_by_dimension(_mesh_set, 2, num);
        return num;
    }

    int get_number_of_cells() {
        int num = 0;
        auto rval = _mb->get_number_entities_by_dimension(_mesh_set, 3, num);
        return num;
    }
    // Create edges 
    void create_edges() {

        if (_mesh_type == Mesh_type::PLANAR ||
                _mesh_type == Mesh_type::SURFACE) {
            Range edges, faces;
            auto rval = _mb->get_entities_by_dimension(_mesh_set, 2, faces);
            rval = _mb->get_adjacencies(faces, 1, true, edges,
                    Interface::UNION);
        } else if (_mesh_type == Mesh_type::SOLID) {

            Range edges, cells;
            auto rval = _mb->get_entities_by_dimension(_mesh_set, 3, cells);
            rval = _mb->get_adjacencies(cells, 1, true, edges,
                    Interface::UNION);
        }
    }

    // Create faces 
    void create_faces() {
        if (_mesh_type == Mesh_type::SOLID)
        {
            Range faces, cells;
            auto rval = _mb->get_entities_by_dimension(_mesh_set, 3, cells);
            rval = _mb->get_adjacencies(cells, 2, true, faces,
                    Interface::UNION);
        }
    }

    void print() {
        
        std::cout << "Number of vertices:" << get_number_of_vertices() << std::endl;

        if(_mesh_type == Mesh_type::PLANAR ||
                _mesh_type == Mesh_type::SURFACE) {
            create_edges();
            std::cout << "Nunber of edges:" << get_number_of_edges() << std::endl;
            std::cout << "Number of faces:" << get_number_of_faces() << std::endl;
        } else if (_mesh_type == Mesh_type::SOLID) {
            create_edges();
            create_faces();
            std::cout << "Nunber of edges:" << get_number_of_edges() << std::endl;
            std::cout << "Number of faces:" << get_number_of_faces() << std::endl;
            std::cout << "Number of cells:" << get_number_of_cells() << std::endl;
        }
    }

    // is function

    bool is_planar_mesh() {
        return _mesh_type == Mesh_type::PLANAR;
    }

    bool is_surface_mesh() {
        return _mesh_type == Mesh_type::SURFACE;
    }

    bool is_solid_mesh() {
        return _mesh_type == Mesh_type::SOLID;
    }

    bool is_2d_mesh() {
        return is_planar_mesh() || is_surface_mesh;
    }

    bool is_3d_mesh() {
        return _mesh_type == Mesh_type::SOLID;
    }

    // Output 
    void write_file(const string & file_name)
    {
        _mb->write_file(file_name.c_str());
    }

private:
    std::shared_ptr<Interface> _mb;
    EntityHandle _mesh_set;
    Mesh_type _mesh_type;
};

} // end of namespace fem

} // end of namespace iMath
#endif // end of Mesh_h
