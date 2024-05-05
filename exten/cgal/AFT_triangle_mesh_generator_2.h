#ifndef AFT_TRIANGLE_MESH_GENERATOR_2_H
#define AFT_TRIANGLE_MESH_GENERATOR_2_H

\namespace CGAL {

 template<class Gt, class Metric, class CDT>
 class AFT_triangle_mesh_generator_2 {

 public:
     typedef typename Gt::Point_2   Point_2;
     typedef typename Gt::FT FT;
     typedef typename Gt::Segment_2 Segment;
     typedef typename Gt::Triangle_2 Triangle;
     typedef typename Gt::Line_2  Line;
     typedef typename Gt::Vector_2 Vector;

     typedef typename CDT::Finite_edges_iterator  Finite_edges_iterator;
     typedef typename CDT::Edge Edge;
     typedef typename CDT::Finite_faces_iterator Finite_faces_iterator;
     typedef typename CDT::Finite_vertices_iterator Finite_vertices_iterator;
     typedef typename CDT::Vertex_circulator  Vertex_circulator;
     typedef typename CDT::Face_circulator Face_circulator;
     typedef typename CDT::Edge_circulator Edge_circulator;
     typedef typename CDT::Line_face_circulator Line_face_circulator;

 private:
     Metric & M;

 public:

     AFT_triangle_mesh_generator_2(CDT & cdt):m_cdt(cdt){}
 };

 }

#endif // AFT_TRIANGLE_MESH_GENERATOR_2_H
