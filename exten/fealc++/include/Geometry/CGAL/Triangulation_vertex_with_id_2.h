#ifndef Triangulation_vertex_with_id_2_h
#define Triangulation_vertex_with_id_2_h

#include "CGAL/Triangulation_vertex_base_2.h"
namespace CGAL {
template < typename GT,
         typename Vb = Triangulation_vertex_base_2<GT> >
class Triangulation_vertex_with_id_2 
: public Vb
{
public:
    typedef typename GT::Point_2                  Point;
    typedef typename Vb::Face_handle              Face_handle;
    typedef typename Vb::Vertex_handle            Vertex_handle;
#ifndef CGAL_NO_DEPRECATED_CODE
    typedef typename Vb::Vertex_circulator       Vertex_circulator; 
    typedef typename Vb::Edge_circulator         Edge_circulator;  
    typedef typename Vb::Face_circulator         Face_circulator; 
    typedef typename Vb::size_type               size_type;      
#endif
    template < typename TDS2 >
    struct Rebind_TDS {
          typedef typename Vb::template Rebind_TDS<TDS2>::Other  Vb2;
          typedef Triangulation_vertex_with_id_2<GT, Vb2>           Other;
    };

public:
    Triangulation_vertex_with_id_2(): Vb() {}
    Triangulation_vertex_with_id_2(const Point & p):Vb(p) {}
    Triangulation_vertex_with_id_2(const Point & p, Face_handle f):Vb(p, f) {}
    Triangulation_vertex_with_id_2(Face_handle f): Vb(f) {}

    const int & id() const {return _id;}
    int & id() {return _id;}
private:
    int _id;
};



}






#endif 
