#ifndef Geometry_kernel_h
#define Geometry_kernel_h

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include "Bisection_alg.h"
#include "Delaunay_alg_2.h"
#include "Level_set_function.h"

namespace iMath {

using namespace CGAL;

template<class GK = CGAL::Exact_predicates_inexact_constructions_kernel>  
class Geometry_kernel_base:public GK
{
public:
    typedef GK   Base;
    typedef typename GK::Point_2 Point_2;
    typedef typename GK::Point_3 Point_3;
    typedef typename GK::Vector_2 Vector_2;
    typedef typename GK::Vector_3 Vector_3;
    typedef typename GK::Triangle_2 Triangle_2;
    typedef typename GK::Triangle_3 Triangle_3;

public:

    static Point_2 point_2(const double * p) { return Point_2(p[0], p[1]);}
    static Point_3 point_3(const double * p) { return Point_3(p[0], p[1], p[2]);}
    static Vector_2 vector_2(const double *v) { return Vector_2(v[0], v[1]);}
    static Vector_3 vector_3(const double *v) { return Vector_3(v[0], v[1], v[2]);}
    static const double two_thirds() { return 2.0/3.0;}
    static const double one_thirds() { return 1.0/3.0;}
    static const double pi()  {return 3.1415926535897931e+0;}
    static const double eps() {return 1e-12;} 

    static Point_2 barycenter(const Point_2 & p1, const double & w1, const Point_2 & p2)
    {
        return CGAL::barycenter(p1, w1, p2);
    }

    static Point_2 barycenter(const Point_2 & p1, const double & w1, 
                       const Point_2 & p2, const double & w2)
    {
        return CGAL::barycenter(p1, w1, p2, w2);
    }

    static Point_2 barycenter(const Point_2 & p1, const double & w1, 
                       const Point_2 & p2, const double & w2, const Point_2 & p3)
    {
        return CGAL::barycenter(p1, w1, p2, w2, p3);
    }

    static Point_2 barycenter(const Point_2 & p1, const double & w1, 
                       const Point_2 & p2, const double & w2,
                       const Point_2 & p3, const double & w3)
    {
        return CGAL::barycenter(p1, w1, p2, w2, p3, w3);
    }

    static Point_3 barycenter(const Point_3 & p1, const double & w1, const Point_3 & p2)
    {
        return CGAL::barycenter(p1, w1, p2);
    }

    static Point_3 barycenter(const Point_3 & p1, const double & w1, 
                       const Point_3 & p2, const double & w2)
    {
        return CGAL::barycenter(p1, w1, p2, w2);
    }

    static Point_2 midpoint(const Point_2 & p1, const Point_2 & p2)
    {
        return CGAL::midpoint(p1, p2);
    }

    static Point_3 midpoint(const Point_3 & p1, const Point_3 & p2)
    {
        return CGAL::midpoint(p1, p2);
    }

    static void midpoint_3(const double * p1, const double * p2, double * p)
    {
        p[0] = (p1[0] + p2[0])/2.0;
        p[1] = (p1[1] + p2[1])/2.0;
        p[2] = (p1[2] + p2[2])/2.0;
    }

    static void midpoint_2(const double * p1, const double * p2, double * p)
    {
        p[0] = (p1[0] + p2[0])/2.0;
        p[1] = (p1[1] + p2[1])/2.0;
    }

    static Vector_3 cross_product(const Vector_3 & v1, const Vector_3 & v2)
    {
        return CGAL::cross_product(v1, v2);
    }

};

template<class GK = Geometry_kernel_base<> >
class Geometry_kernel:public GK
{
public:
    typedef GeoAlg::Bisection_alg<GK>  Bisection_algorithm;
    typedef GeoAlg::Delaunay_alg_2<GK> Delaunay_algorithm_2;

    typedef LevelSetFunction::Circle<GK> Level_set_circle;
    typedef LevelSetFunction::Sphere<GK> Level_set_sphere;
    typedef LevelSetFunction::Signed_distance_circle<GK> Signed_distance_circle;
    typedef LevelSetFunction::Signed_distance_sphere<GK> Signed_distance_sphere;

    typedef LevelSetFunction::Union<GK, typename GK::Point_2> Union_2;
    typedef LevelSetFunction::Union<GK, typename GK::Point_3> Union_3;

};

} // end of Geometry_kernel

#endif // end of Geometry_kernel_h
