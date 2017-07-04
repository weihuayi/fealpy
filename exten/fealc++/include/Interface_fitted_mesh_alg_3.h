#ifndef Interface_fitted_mesh_alg_3_h
#define Interface_fitted_mesh_alg_3_h


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

template<class G = iMath::Geometry_kernel<> >
class Interface_fitted_mesh_alg_base
{
public:

    typedef typename GK::Point_2 Point_2
    typedef typename GK::Point_3 Point_3;
    typedef typename GK::Bisection_algorithm Bisection_algorithm;

public:

    Interface_fitted_mesh_alg_base() 
    {
    }

    /** Create the auxilliary tag.
     *
     * `mVertexSignTag` tag contain the data about the location of a vertex. 
     *      sign == 1, vertex is outside of the interface.
     *      sign == 0, vertex is on the interface.
     *      sign == -1, vertex is inside of the interface
     */
    void create_tag(Interface * mb)
    {
        const int default_val = 0;
        const int default_val_length = 1;
        rval = mb->tag_get_handle("vertex_sign_tag", default_val_length,
                MB_TYPE_INTEGER, 
                mVertexSignTag, 
                MB_TAG_CREAT | MB_TAG_DENSE,
                &default_val);
        rval = mb->tag_get_handle("cross_cell_tag", default_val_length, 
                MB_TYPE_INTEGER,
                mCrossCellTag,
                MB_TAG_CREAT | MB_TAG_DENSE,
                &default_val);
        const EntityHandle default_handle = 0;
        const int default_handle_length = 1;
        rval = mb->tag_get_handle("edge_to_cross_vertex_tag", default_val_length, 
                MB_TYPE_HANDLE,
                mEdge2CrossVertexTag,
                MB_TAG_CREAT | MB_TAG_DENSE,
                &default_handle_val);
    }

    /** Create edges. 
     *
     *  In MOAB, edge entity is not appear in default. So one should create it.
     *
     *
     *
     */

    void create_edges(Interface * mb)
    {
        Range cells, edges;
        rval = mb->get_entities_by_dimension(0, 3, cells);
        rval = mb->get_adjacencies(cells, 1, true, edges, 
                Interface::UNION);
    }

    /** Create faces. 
     *
     *  In MOAB, faces entity are not appear in default. So one should create it.
     *
     *
     *
     */

    void create_faces(Interface * mb)
    {
        Range cells, faces;
        rval = mb->get_entities_by_dimension(0, 3, cells);
        rval = mb->get_adjacencies(cells, 2, true, faces, 
                Interface::UNION);
    }

    template<class LevelSetFun>
    void compute_vertex_sign(Interface * mb, LevelSetFun & fun)
    {
        Range verts;
        rval = mb->get_entities_by_dimension(0, 0, verts);
        for(auto it = verts.begin(); it != verts.end(); it++)
        {
            EntityHandle vh = *it;
            double cp[3];
            rval = mb->get_coords(&vh, 1, cp);
            int val = fun.sign(fun.point(cp));
            rval = mb->tag_set_data(mVertexSignTag, &vh, 1, &val);
        }
    }

    template<class LevelSetFunction>
    void find_cross_edge(Interface * mb, LevelSetFunction & fun)
    {

        Range edges;
        rval = mb->get_entities_by_dimension(0, 1, edges);
        for(auto it = edges.begin(); it != edges.end(); it++)
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
                auto m = bisection(fun, 
                        fun.point(coords),
                        fun.point(coords+3));

                EntityHandle vh;
                double new_p[] = {m[0], m[1], 0.0};
                if(m.dimension() == 3)
                    new_p[2] = m[2];
                rval = mb->create_vertex(new_p, vh);
                rval = mb->tag_set_data(mEdge2CrossVertexTag, &eh, 1, &val);
            }
        }

    }

    virtual bool is_cross_cell(Interface * mb, const EntityHandle eh) = 0;

    virtual int is_special_cell(Interface * mb, const EntityHandle eh) = 0;

    /** Find all cross element.
     *  
     */
    void find_cross_cell(Interface * mb)
    {
        Range cells;
        rval = mb->get_entities_by_dimension(0, 2, cells);
        for(auto = cells.begin(); it != cells.end(); it++)
        {
            EntityHandle eh = *it;
            if(is_cross_cell(mb, eh))
            {
                int val = 1;
                rval = mb->tag_set_data(mCrossCellTag, &eh, 1, &val); 
            }

        }
    }

    Tag get_vertex_sign_tag() {return mVertexSignTag;}
    Tag get_cross_cell_tag() {return mCrossCellTag;}
    Tag get_edge_to_cross_vertex_tag() {return mEdge2CrossVertexTag;}

private:
        Tag mVertexSignTag;/*The sign of the value of the level set function on vertices*/
        Tag mCrossCellTag;/*The crossing flag of cells*/
        Tag mEdge2crossVertexTag;/*the edge to cross vertex info*/
        ErrorCode rval;
}


/* 
 *
 *
 *
 *
 */

template<class GK = iMath::Geometry_kernel<> >
class Interface_fitted_mesh_alg_3:public Interface_fitted_mesh_alg_base<GK>
{
public:
    typedef typename GK::Point_3 Point_3;
    typedef typename GK::Bisection_algorithm Bisection_algorithm;
public:

    /** Constructor
     *
     * Input:
     *
     *  mb  MOAB Interface with a initial Cartesian mesh
     *
     */
    Interface_fitted_mesh_alg_3(){}

    void set_cross_cell_tag(Interface * mb)
    {
        Range cells;
        rval = mb->get_entities_by_dimension(0, 3, cells);
        for(auto it = cells.begin(); it != cells.end(); it++)
        {
            set_cross_cell_tag(mb, *it);
        }
    }

    void set_cross_cell_tag(Interface * mb, const EntityHandle ch)
    {
        Range adj;
        mb->get_adjacencies(&ch, 1, 1, false, adj_entities);
        EntityHandle vhs[adj.size()];
        Tag edge2crossVertexTag = get_edge_to_cross_vertex_tag();
        rval = mb->tag_get_data(edge2crossVertex, adj, vhs);

        Tag crossCellTag = get_cross_cell_tag();
        int i = 0;
        bool flag = false;
        for(auto it = adj.begin(); it != adj.end(); it++, i++)
        {
            if(vhs[i] == 0)
            {
                eh = *it;
                int val = 1;
                rval = mb->tag_set_data(crossCellTag, &ch, 1, val);
                flag = true;
                break;
            }
        }
    }


};

} // end of namespace MeshAlg

} // end of namespace iMath


#endif // end of Interface_fitted_mesh_alg_3_h
