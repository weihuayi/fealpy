#ifndef TRIANGULATION_VERTEX_BASE_WITH_TYPE_2_H
#define TRIANGULATION_VERTEX_BASE_WITH_TYPE_2_H

#include <CGAL/Triangulation_vertex_base_2.h>

namespace CGAL {

enum Vertex_dof {
    FIXED = 0,  //固定点
    CONSTRAINED1D, //
    CONSTRAINED2D, //
    FREE           //自由点,可以在区域内自由移动
};

enum Vertex_type {
    OLD,
    NEW
};

enum Vertex_bdtype {
    OUTERBOUND = -1,
    INNERBOUND = 1,
    INDOMAIN = 0
};

enum Vertex_mvtype{
    MOVE,
    UNMOVE
};

enum Vertex_frontype{
    UNFRONT,
    FRONTV
};

template < typename Gt,
           typename Vb = Triangulation_vertex_base_2<Gt> >
class Triangulation_vertex_base_with_type_2
  : public Vb
{
  Vertex_type _type;
  Vertex_bdtype _bdtype;
  int _id;
  Vertex_dof _dof;
  Vertex_mvtype _mvtype;
  Vertex_frontype _frontype;
public:
  typedef typename Vb::Face_handle                   Face_handle;
  typedef typename Vb::Point                        Point;

  template < typename TDS2 >
  struct Rebind_TDS {
    typedef typename Vb::template Rebind_TDS<TDS2>::Other          Vb2;
    typedef Triangulation_vertex_base_with_type_2<Gt, Vb2>   Other;
  };

  Triangulation_vertex_base_with_type_2()
      : Vb(), _type(NEW), _bdtype(INDOMAIN), _id(-1), _dof(FREE),_mvtype(UNMOVE),_frontype(UNFRONT) {}

  Triangulation_vertex_base_with_type_2(const Point & p)
      : Vb(p), _type(NEW), _bdtype(INDOMAIN), _id(-1), _dof(FREE),_mvtype(UNMOVE),_frontype(UNFRONT) {}

  Triangulation_vertex_base_with_type_2(const Point & p, Face_handle c)
      : Vb(p, c), _type(NEW), _bdtype(INDOMAIN), _id(-1),_dof(FREE),_mvtype(UNMOVE),_frontype(UNFRONT) {}

  Triangulation_vertex_base_with_type_2(Face_handle c)
      : Vb(c), _type(NEW), _bdtype(INDOMAIN), _id(-1), _dof(FREE),_mvtype(UNMOVE),_frontype(UNFRONT) {}

  Vertex_type &  type() { return _type; }
  int &  id() { return _id; }
  Vertex_dof & dof() { return _dof;}
  Vertex_bdtype & bdtype() {return _bdtype;}
  Vertex_mvtype & mvtype() {return _mvtype;}
  Vertex_frontype & frontype() {return _frontype;}
};

} //namespace CGAL
#endif // TRIANGULATION_VERTEX_BASE_WITH_MARK_2_H
