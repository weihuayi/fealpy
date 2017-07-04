#ifndef Delaunay_alg_2_h
#define Delaunay_alg_2_h

#include <CGAL/Delaunay_triangulation_2.h>
#include "Geometry/CGAL/Triangulation_vertex_with_id_2.h"


#include <vector>
#include <ostream>

namespace iMath {

namespace GeoAlg {

template<class GK>
class Delaunay_alg_2
{
public:
    typedef typename GK::Point_2 Point_2;

    typedef CGAL::Triangulation_vertex_with_id_2<GK> TV;
    typedef CGAL::Triangulation_data_structure_2<TV> TDS;
    typedef CGAL::Delaunay_triangulation_2<GK, TDS> DT;
    typedef typename DT::Vertex_handle Vh;
    typedef typename DT::Finite_faces_iterator Ffi;
public:
    Delaunay_alg_2(){}

    int operator () (std::vector<Point_2> pts, std::vector<int> & triangles)
    {
        int num_pts = pts.size();
        for(int i = 0; i < num_pts; i++)
        {
            Vh vh = _dt.insert(pts[i]);
            vh->id() = i;
        }

        int num_tris = _dt.number_of_faces();
        triangles.resize(3*num_tris);

        Ffi ffi = _dt.finite_faces_begin();
        for(int i = 0 ; ffi != _dt.finite_faces_end(); ffi++, i++)
        {
            for(int j = 0; j < 3; j++)
                triangles[3*i + j] = ffi->vertex(j)->id();

        }
        return num_tris;
    }
    
private:
    DT _dt;
};

} // end of namespace GeoAlg 

} // end of namespace iMath

#endif // end of Delaunay_alg_2_h:
