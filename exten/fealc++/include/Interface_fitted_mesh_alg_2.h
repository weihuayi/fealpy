#ifndef Interface_fitted_mesh_alg_2_h
#define Interface_fitted_mesh_alg_2_h


#include "moab/Core.hpp"
#include "moab/ScdInterface.hpp"
#include "MBTagConventions.hpp"

#include "Geometry/Geometry_kernel.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>


namespace iMath {

namespace MeshAlg {

using namespace moab;

template<class GK = iMath::Geometry_kernel<> >
class Interface_fitted_mesh_alg_2
{

public:
    typedef typename GK::Point_2 Point_2;
    typedef typename GK::Point_3 Point_3;
    typedef typename GK::Delaunay_algorithm_2 Delaunay_algorithm_2;
    typedef typename GK::Bisection_algorithm Bisection_algorithm;
public:

        /** Constructor
         *
         * Input:
         *
         *  mb  MOAB Interface with a initial Cartesian mesh
         *
         */
        Interface_fitted_mesh_alg_2(){}

        /** Create the auxilliary tag.
         *
         * `vsign` tag contain the data about the location of a vertex. 
         *      vsign == 1, vertex is outside of the interface.
         *      vsign == 0, vertex is on the interface.
         *      vsign == -1, vertex is inside of the interface
         * 
         *
         *
         */
        void create_tag(Interface * mb)
        {
            const int default_val = 0;
            const int default_val_length = 1;
            rval = mb->tag_get_handle("vertex_sign", default_val_length,
                    MB_TYPE_INTEGER, 
                    vsign, 
                    MB_TAG_CREAT | MB_TAG_DENSE,
                    &default_val);
            rval = mb->tag_get_handle("cutted_tag", default_val_length, 
                    MB_TYPE_INTEGER,
                    cutted_elem_tag,
                    MB_TAG_CREAT | MB_TAG_DENSE,
                    &default_val);
        }

        /** Create edges. 
         *
         *  In MOAB, edge entity is not appear in default. So one should create it.
         *
         *
         *
         */

        void create_edge(Interface * mb)
        {
            Range elems, edges;
            rval = mb->get_entities_by_dimension(0, 2, elems);
            rval = mb->get_adjacencies(elems, 1, true, edges, 
                    Interface::UNION);
        }

        /** Find the cutted edge and create the new intersection points.
         *  
         *
         */

        template<class LevelSetFunction>
        void find_cutted_edge(Interface * mb, LevelSetFunction & fun)
        {

            Range edges;
            rval = mb->get_entities_by_dimension(0, 1, edges);
            Range::iterator it = edges.begin();
            for(; it != edges.end(); it++)
            {
                int val[2] = {0, 0};
                EntityHandle eh = *it;
                std::vector<EntityHandle> adj_vertices;
                rval = mb->get_adjacencies(&eh, 1, 0, false, adj_vertices);
                rval = mb->tag_get_data(vsign, adj_vertices.data(), 2, val);
                if(val[0]*val[1] < 0)
                {
                    std::vector<EntityHandle> adj_vertices;
                    rval = mb->get_adjacencies(&eh, 1, 0, false, adj_vertices);

                    //Find the intersect point
                    double coords[6];
                    rval = mb->get_coords(adj_vertices.data(), 2, coords);

                    Bisection_algorithm bisection;
                    Point_2 p1(coords[0], coords[1]);
                    Point_2 p2(coords[3], coords[4]);
                    Point_2 m = bisection(fun, p1, p2);

                    EntityHandle vh;
                    double new_p[] = {m[0], m[1], 0.0};
                    rval = mb->create_vertex(new_p, vh);

                    edge2vertex.insert(std::pair<EntityHandle,EntityHandle>(eh, vh));
                }
            }

        }

        /** Find all cutted element.
         *  
         */
        void find_cutted_elem(Interface * mb)
        {
            Range elems;
            rval = mb->get_entities_by_dimension(0, 2, elems);
            Range::iterator it = elems.begin();
            for(; it != elems.end(); it++)
            {
                EntityHandle eh = *it;
                if(is_cutted_elem(mb, eh))
                {
                    int val = 1;
                    rval = mb->tag_set_data(cutted_elem_tag, &eh, 1, &val); 
                }

            }
        }

        
        bool is_cutted_elem(Interface * mb, const EntityHandle eh)
        {
            std::vector<EntityHandle> conn;
            rval = mb->get_connectivity(&eh, 1, conn, true);
            int sign[conn.size()]; // conn.size() == 4
            rval = mb->tag_get_data(vsign, conn.data(), conn.size(), sign);

            int sum = std::abs(sign[0]) + std::abs(sign[1]) + std::abs(sign[2]) + 
                std::abs(sign[3]);
            return (sum < 3) || (sign[0]*sign[1] < 0) ||
                (sign[1]*sign[2] < 0) || (sign[2]*sign[3] < 0) ||  
                (sign[3]*sign[0] < 0);
        }

        int is_special_elem(Interface * mb, const EntityHandle eh)
        {
            std::vector<EntityHandle> conn;
            rval = mb->get_connectivity(&eh, 1, conn, true);
            int sign[conn.size()]; // conn.size() == 4
            rval = mb->tag_get_data(vsign, conn.data(), conn.size(), sign);
            bool flag1 = sign[0] == 0 && sign[2] == 0 && sign[1]*sign[3] < 0;
            bool flag2 = sign[1] == 0 && sign[3] == 0 && sign[0]*sign[2] < 0;

            if(flag1)
                return 1;
            else if(flag2)
                return 2;
            else 
                return 0;
        }


        /** Create triangles
         *
         *
         */


        void create_triangle(Interface * mb)
        {

            Range elems;
            rval = mb->get_entities_by_dimension(0, 2, elems);
            Range::iterator it = elems.begin();
            for(; it != elems.end(); it++)
            {
                EntityHandle eh = *it;
                int val;
                rval = mb->tag_get_data(cutted_elem_tag, &eh, 1, &val);
                if(val == 1)
                    create_triangle(mb, eh);
            }
        }


        void create_triangle(Interface * mb, const EntityHandle eh)
        {
            std::vector<EntityHandle> vertices;
            rval = mb->get_connectivity(&eh, 1, vertices, true);
            int special = is_special_elem(mb, eh);
            if(special > 0)
            {
                if(special == 1)
                {
                    EntityHandle tri1[3] = {vertices[1], vertices[2], vertices[0]};
                    EntityHandle tri2[3] = {vertices[3], vertices[0], vertices[2]};
                    EntityHandle th1;
                    EntityHandle th2;
                    rval = mb->create_element(MBTRI, tri1, 3, th1);
                    rval = mb->create_element(MBTRI, tri2, 3, th2);
                }
                else
                {
                    EntityHandle tri1[3] = {vertices[0], vertices[1], vertices[3]};
                    EntityHandle tri2[3] = {vertices[2], vertices[3], vertices[1]};
                    EntityHandle th1;
                    EntityHandle th2;
                    rval = mb->create_element(MBTRI, tri1, 3, th1);
                    rval = mb->create_element(MBTRI, tri2, 3, th2);
                }
            }
            else
            {
                Range adj_edges;
                rval = mb->get_adjacencies(&eh, 1, 1, false, adj_edges); 
                Range::iterator it = adj_edges.begin();
                std::map<EntityHandle, EntityHandle>::iterator mapit;
                for(; it != adj_edges.end(); it++)
                {
                    mapit = edge2vertex.find(*it);
                    if(mapit != edge2vertex.end())
                    {
                        vertices.push_back(mapit->second);
                    }
                }

                // Get the coords of the vertices

                double coords[3*vertices.size()];
                rval = mb->get_coords(vertices.data(), vertices.size(), coords);

                std::vector<Point_2> pts(vertices.size());
                for(int i = 0; i < vertices.size(); i++)
                {
                    pts[i] = Point_2(coords[3*i], coords[3*i+1]);
                }
                
                std::vector<int> triangles;
                Delaunay_algorithm_2 dt; 
                int num_tris = dt(pts, triangles);

                // Create the triangle entities in moab
                std::vector<EntityHandle> conn(3);
                for(int i = 0; i < num_tris; i++)
                {
                    for(int j = 0; j < 3; j++)
                        conn[j] = vertices[triangles[3*i + j]];
                    EntityHandle eh;
                    rval = mb->create_element(MBTRI, conn.data(), 3, eh);
                }

            }
        }

        
        template<class LevelSetFun>
        void compute_vertex_sign(Interface * mb, LevelSetFun & fun)
        {
            Range verts;
            rval = mb->get_entities_by_dimension(0, 0, verts);
            for(Range::iterator it = verts.begin(); it != verts.end(); it++)
            {
                EntityHandle vh = *it;

                double cp[3];
                rval = mb->get_coords(&vh, 1, cp);

                Point_2 p(cp[0], cp[1]);
                int val = fun.sign(p);
                rval = mb->tag_set_data(vsign, &vh, 1, &val);
            }
        }

        void set_global_id(Interface * mb)
        {
            Range verts;
            rval = mb->get_entities_by_dimension(0, 0, verts);
            int id = 0;
            Tag gid;
            rval = mb->tag_get_handle("GLOBAL_ID", 1, MB_TYPE_INTEGER, gid);
            for(Range::iterator it = verts.begin(); it!= verts.end(); it++)
            {
                EntityHandle vh = *it;
                rval = mb->tag_set_data(gid, &vh, 1, &id);
                id += 1;
            }

        }

        void post_process(Interface * mb)
        {
            Range squares;
            int val = 1;
            const void * val_ptr = &val;
            rval = mb->get_entities_by_type_and_tag(0, MBQUAD, 
                    &cutted_elem_tag, &val_ptr, 1,
                    squares, Interface::UNION);
            rval = mb->delete_entities(squares);

            Range edges;
            rval = mb->get_entities_by_dimension(0, 1, edges);
            rval = mb->delete_entities(edges);
        }

        template<class LevelSetFun>
        void execute(Interface * mb, LevelSetFun & fun)
        {
            create_tag(mb);
            create_edge(mb);
            compute_vertex_sign(mb, fun);
            find_cutted_edge(mb, fun);
            find_cutted_elem(mb);
            create_triangle(mb);
            set_global_id(mb);
        }


        
        void write_file(Interface *mb,  std::string & file_name)
        {
            std::ofstream out;
            out.open(file_name.c_str(), std::ios::out);
            out << " # vtk DataFile Version 3.0 \n MOAB 4.9.1 \n ASCII \n DATASET UNSTRUCTURED_GRID \n"; 

            Range verts;
            rval = mb->get_entities_by_dimension(0, 0, verts);
            out << "POINTS " << verts.size() << " double" << '\n';

            for(Range::iterator it = verts.begin(); it != verts.end(); it++)
            {
                EntityHandle vh = *it;
                int id;
                double cd[3];
                rval = mb->get_coords(&vh, 1, cd);

                out << cd[0] << " " << cd[1] << " " << cd[2] << '\n';
            }

            Range tris, squares;
            int val = 0;
            const void * val_ptr = &val;
            rval = mb->get_entities_by_type(0, MBTRI, tris);
            rval = mb->get_entities_by_type_and_tag(0, MBQUAD, 
                    &cutted_elem_tag, &val_ptr, 1,
                    squares, Interface::UNION);
            int num_t = tris.size();
            int num_s = squares.size();
            out << "CELLS " << num_t + num_s << " " << 4*num_t + 5*num_s << '\n';

            std::vector<int> cell_types(num_t + num_s);
            int idx = 0;
            Tag gid;
            rval = mb->tag_get_handle("GLOBAL_ID", 1, MB_TYPE_INTEGER, gid);
            for(Range::iterator it = tris.begin(); it != tris.end(); it++)
            {
                EntityHandle th = *it;
                std::vector<EntityHandle> conn;
                std::vector<int> id(3);
                rval = mb->get_connectivity(&th, 1, conn, true);
                rval = mb->tag_get_data(gid, conn.data(), 3, id.data());
                out<< 3 << " " << id[0] << " " << id[1] << " " << id[2] << '\n';
                cell_types[idx] = 5;
                idx++;
            }

            for(Range::iterator it = squares.begin(); it != squares.end(); it++)
            {
                EntityHandle th = *it;
                std::vector<EntityHandle> conn;
                std::vector<int> id(4);
                rval = mb->get_connectivity(&th, 1, conn, true);
                rval = mb->tag_get_data(gid, conn.data(), 4, id.data());
                out<< 4 << " " << id[0] << " " << id[1] << " " << id[2] <<
                    " " << id[3] <<  '\n';
                cell_types[idx] = 9;
                idx++;
            }

            out << "CELL_TYPES " << cell_types.size() << '\n';
            for(int i = 0; i < cell_types.size() ; i++)
            {
                out << cell_types[i] << '\n';
            }

        }


        void print(Interface * mb)
        {
            Range verts;
            rval = mb->get_entities_by_dimension(0, 0, verts);

            std::cout << "Number of vertices: " << verts.size() << std::endl;

            Tag gid;
            rval = mb->tag_get_handle("GLOBAL_ID", 1, MB_TYPE_INTEGER, gid);
            for(Range::iterator it = verts.begin(); it != verts.end(); it++)
            {
                EntityHandle vh = *it;
                int id;
                double cd[3];
                rval = mb->tag_get_data(gid, &vh, 1, &id);
                rval = mb->get_coords(&vh, 1, cd);

                std::cout << "(" << cd[0] << ", " << cd[1]  << ") with id " << id << std::endl;

            }

            Range tris, squares;
            int val = 0;
            const void * val_ptr = &val;
            rval = mb->get_entities_by_type(0, MBTRI, tris);
            rval = mb->get_entities_by_type_and_tag(0, MBQUAD, 
                    &cutted_elem_tag, &val_ptr, 1,
                    squares, Interface::UNION);

            std::cout << "The total number of elems: " << tris.size() + squares.size() << std::endl;
            std::cout << "Number of triangles:" << tris.size() << std::endl;
            std::cout << "Number of squares:" << squares.size() << std::endl;

            for(Range::iterator it = tris.begin(); it != tris.end(); it++)
            {
                EntityHandle th = *it;
                std::vector<EntityHandle> conn;
                std::vector<int> id(3);
                rval = mb->get_connectivity(&th, 1, conn, true);
                rval = mb->tag_get_data(gid, conn.data(), 3, id.data());
                std::cout<< id[0] << ", " << id[1] << ", " << id[2] << std::endl;
            }

        }

        
private:
        Tag vsign;
        Tag cutted_elem_tag;
        std::map<EntityHandle, EntityHandle> edge2vertex;
        ErrorCode rval;
};

} //end of namespace MeshAlg 

} // end of namespace iMath
#endif // end of Interface_fitted_mesh_alg_2_h
