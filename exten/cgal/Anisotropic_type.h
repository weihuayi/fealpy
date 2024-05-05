#ifndef ANISOTROPIC_TYPE_H
#define ANISOTROPIC_TYPE_H

#include "Anisotropic_kernel.h"
#include "Metric_2.h"
#include "Constrained_triangulation_face_base_with_blind_mark_2.h"
#include "Triangulation_vertex_base_with_type_2.h"

#include <CGAL/triangulation_assertions.h>
#include <CGAL/Constrained_triangulation_2.h>

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/barycenter.h>

#include <CGAL/IO/Geomview_stream.h>
#include <CGAL/IO/Triangulation_geomview_ostream_2.h>

#include <cassert>
#include <vector>
#include <algorithm>
#include <map>
#include <CGAL/algorithm.h>




typedef CGAL::Exact_predicates_inexact_constructions_kernel KBase;
typedef CGAL::Metric_2<KBase> M;

template<> double M::a11 = 1;
template<> double M::a12 = 0;
template<> double M::a22 = 1;
template<> double M::h = 0.1;

typedef CGAL::Anisotropic_kernel<M,KBase> K;
typedef CGAL::Triangulation_vertex_base_with_type_2<K> Vb;
typedef CGAL::Constrained_triangulation_face_base_with_blind_mark_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb,Fb> Tds;
typedef CGAL::Exact_predicates_tag                               Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K,Tds, Itag> Delaunay;
typedef Delaunay::Vertex_handle Vh;
typedef Delaunay::Face_handle Fh;
typedef K::Point_2   Point;
typedef K::FT FT;
typedef K::Segment_2 Segment;
typedef K::Triangle_2 Triangle;
typedef K::Line_2  Line;
typedef K::Vector_2 Vector;
typedef std::list<Point > List;

typedef Delaunay::Finite_edges_iterator  Ei;
typedef Delaunay::Edge Edge;
typedef Delaunay::Finite_faces_iterator Fi;
typedef Delaunay::Finite_vertices_iterator Vi;
typedef Delaunay::Vertex_circulator  Vc;
typedef Delaunay::Face_circulator Fc;
typedef Delaunay::Edge_circulator Ec;
typedef Delaunay::Line_face_circulator Lfc;

#endif // ANISOTROPIC_TYPE_H
