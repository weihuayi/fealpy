#ifndef ANISOTROPIC_KERNEL_H
#define ANISOTROPIC_KERNEL_H

#include <CGAL/algorithm.h>
#include <CGAL/number_utils.h>
#include <CGAL/predicates/sign_of_determinant.h>
#include <CGAL/basic.h>
#include "Metric_field_2.h"

namespace CGAL{


//TODO: 利用度量中的函数计算计算点乘运算
namespace CartesianKernelFunctors {

template <class K, class MBase, Metric_field_2<K,MBase> & M >
class Side_of_oriented_anisotropic_circle_2
{
  typedef typename K::Point_2        Point_2;
  typedef typename K::FT FT;
public:
  typedef typename K::Oriented_side  result_type;

  result_type
  operator()( const Point_2& p, const Point_2& q,
          const Point_2& r, const Point_2& t) const
  { // t 位于 pqr 外接圆外部(-1)还是内部(+1)
      FT x, y;
      x = ( (p[0]+q[0]+r[0])/3.0 )/1.0;
      y = ( (p[1]+q[1]+r[1])/3.0 )/1.0;
      Point_2 o(x,y);
      return sign_of_determinant<FT> (p[0], p[1], M.dot_product(o,p[0],p[1]), 1,
                                      q[0], q[1], M.dot_product(o,q[0],q[1]), 1,
                                      r[0], r[1], M.dot_product(o,r[0],r[1]), 1,
                                      t[0], t[1], M.dot_product(o,t[0],t[1]), 1);
  }
};


template <typename K, class MBase, Metric_field_2<K,MBase> & M >
class Construct_anisotropic_bisector_2
{
  typedef typename K::FT      FT;
  typedef typename K::Point_2 Point_2;
  typedef typename K::Line_2  Line_2;
  typedef typename K::Vector_2 Vector_2;
public:
  typedef Line_2              result_type;

  result_type
  operator()(const Point_2& p, const Point_2& q) const
  {// 到 p,q 黎曼距离近似相等的线
      Point_2 r = midpoint(p,q);
      FT a11, a12, a22;
      a11 = M.a11(r);
      a12 = M.a12(r);
      a22 = M.a22(r);
      FT C;
      FT A = 2.0*(a11*(q[0]-p[0])+a12*(q[1]-p[1]));
      FT B = 2.0*(a12*(q[0]-p[0])+a22*(q[1]-p[1]));
      C = M.dot_product(r,p[0],p[1]);
      C -= M.dot_product(r,q[0],q[1]);
      return Line_2(A,B,C);
  }
};

template <typename K, class MBase, Metric_field_2<K,MBase> & M >
class Construct_anisotropic_circumcenter_2
{
  typedef typename K::Point_2     Point_2;
  typedef typename K::Triangle_2  Triangle_2;
  typedef typename K::Vector_2    Vector_2;
public:
  typedef Point_2                 result_type;

  Point_2
  operator()(const Point_2& p, const Point_2& q) const
  {
    //TODO: 如果度量是关于(x,y)的函数,　需要更新这个函数
    typename K::Construct_midpoint_2 construct_midpoint_2;
    return construct_midpoint_2(p, q);
  }

  result_type
  operator()(const Point_2& p, const Point_2& q, const Point_2& r, int flag = 0) const
  {
    // 在度量Ｍ下,　计算pqr 外接圆圆心
    typename K::Construct_point_2 construct_point_2;
    typedef typename K::FT        FT;
      if(flag == 0)
      {
          Point_2 o(( p.x()+q.x()+r.x() )/3, ( p.y()+q.y()+r.y() )/3);
          return this->operator ()(p,q,r,o);
      }
      else
      {
          Point_2 o = this->operator ()(p,q,r);
          for(int i = 0; i < flag; i++)
          {
              Point_2 oc = this->operator ()(p,q,r,o);
              o = oc;
          }
          return o;
      }
  }

  result_type
  operator()(const Triangle_2& t) const
  {
    return this->operator()(t.vertex(0), t.vertex(1), t.vertex(2));
  }

  result_type
  operator()(const Point_2& p, const Point_2& q, const Point_2& r, const Point_2& c) const
  {
      // 在度量Ｍ下,　计算pqr 外接圆圆心
      typename K::Construct_point_2 construct_point_2;
      typedef typename K::FT        FT;

      Point_2 o = c;
      FT dqx = q[0] - p[0];
      FT dqy = q[1] - p[1];
      FT drx = r[0] - p[0];
      FT dry = r[1] - p[1];

      FT dq2 = M.dot_product(o,dqx,dqy);
      FT dr2 = M.dot_product(o,drx,dry);

      FT a11 = M.a11(o);
      FT a12 = M.a12(o);
      FT a22 = M.a22(o);

      FT den = 2.0 * M.determinant(o,dqx,dqy,drx,dry);

      FT dcx =   determinant (dq2, a12*dqx + a22*dqy,
                              dr2, a12*drx + a22*dry) / den;
      FT dcy =   determinant (a11*dqx + a12*dqy, dq2,
                              a11*drx + a12*dry, dr2) / den;

      return construct_point_2(dcx+p.x(), dcy+p.y());
  }
};


template <typename K, class MBase, Metric_field_2<K,MBase> & M >
class Compute_anisotropic_area_2
{
  typedef typename K::FT                FT;
  typedef typename K::Iso_rectangle_2   Iso_rectangle_2;
  typedef typename K::Triangle_2        Triangle_2;
  typedef typename K::Point_2           Point_2;
public:
  typedef FT               result_type;

  result_type
  operator()( const Point_2& p, const Point_2& q, const Point_2& r ) const
  {
    Point_2 o(( p.x()+q.x()+r.x() )/3, ( p.y()+q.y()+r.y() )/3);
    FT v1x = q.x() - p.x();
    FT v1y = q.y() - p.y();
    FT v2x = r.x() - p.x();
    FT v2y = r.y() - p.y();

    return sqrt(M.determinant(o))*determinant(v1x, v1y, v2x, v2y)/2;
  }

  //　TODO: 利用度量计算面积
  result_type
  operator()( const Iso_rectangle_2& r ) const
  { return (r.xmax()-r.xmin()) * (r.ymax()-r.ymin()); }

  result_type
  operator()( const Triangle_2& t ) const
  { return this->operator()(t.vertex(0), t.vertex(1), t.vertex(2)); }
};

template <typename K, class MBase, Metric_field_2<K,MBase> & M >
class Compute_anisotropic_squared_distance_2
{
    typedef typename K::FT                FT;
    typedef typename K::Point_2           Point_2;
public:
  typedef FT               result_type;

    result_type
    operator()( const Point_2& p, const Point_2& q, int n = 1 ) const
    {
        FT dpq = 0;
        for( int i = 0; i < n; i++ )
        {
            dpq += std::sqrt(this->operator ()(p,q,n,i));
        }
        return dpq*dpq;
//      Point_2 o(( p.x()+q.x())/2, ( p.y()+q.y())/2);
//      Point_2 op((p.x()+o.x())/2,(p.y()+o.y())/2);
//      Point_2 oq((q.x()+o.x())/2,(q.y()+o.y())/2);
//      Point_2 opp( (p.x()+op.x())/2,(p.y()+op.y())/2 );
//      Point_2 oop( (o.x()+op.x())/2,(o.y()+op.y())/2 );
//      Point_2 ooq( (o.x()+oq.x())/2,(o.y()+oq.y())/2 );
//      Point_2 oqq( (q.x()+oq.x())/2,(q.y()+oq.y())/2 );
//      result_type d_pop =  std::sqrt(this->operator ()(p,op,opp));
//      result_type d_opo =  std::sqrt(this->operator ()(o,op,oop));
//      result_type d_ooq =  std::sqrt(this->operator ()(o,oq,ooq));
//      result_type d_oqq =  std::sqrt(this->operator ()(q,oq,oqq));
//      return std::pow( d_pop+d_opo+d_ooq+d_oqq,2);
    }

    result_type
    operator()( const Point_2& p, const Point_2& q, int n, int i ) const
    {
        Point_2 p0 = p+i/n*(q-p);
        Point_2 q0 = p+(i+1)/n*(q-p);
      FT a1 = ( M.a11(p0)+M.a11(q0) )/2;
      FT a2 = ( M.a12(p0)+M.a12(q0) )/2;
      FT a3 = ( M.a22(p0)+M.a22(q0) )/2;
      FT u = p0.x()-q0.x();
      FT v = p0.y()-q0.y();
      return a1*u*u+2*a2*u*v+a3*v*v;
    }

};

template <typename K, class MBase, Metric_field_2<K,MBase> & M >
class Construct_anisotropic_bisector_point_2
{
  typedef typename K::Point_2     Point_2;
  typedef typename K::Triangle_2  Triangle_2;
  typedef typename K::Vector_2    Vector_2;
  typedef typename K::Line_2  Line_2;
public:
  typedef Point_2                 result_type;

  result_type
  operator()(const Point_2& p, const Point_2& q, const double h=1.0, int flag = 0) const
  {
    // 在度量Ｍ下,　计算距离 P，Q两点为h 的点，在方向向量PQ的右侧
    typename K::Construct_bisector_2 construct_bisector_2;
    typename K::Construct_point_2 construct_point_2;
    typedef typename K::FT        FT;
      if(flag == 0)
      {
          Point_2 o(( p.x()+q.x() )/2, ( p.y()+q.y() )/2);
          return this->operator ()(p,q,o,h);
      }
      else
      {
           Point_2 o(( p.x()+q.x() )/2, ( p.y()+q.y() )/2);
          for(int i = 0; i < flag; i++)
          {
              Point_2 c( (o.x()+p.x()+q.x())/3, (o.y()+p.y()+q.y())/3 );
              o = this->operator ()(p,q,c,h);
          }
          return o;
      }
  }


  result_type
  operator()(const Point_2& p, const Point_2& q, const Point_2 & c, const double h=1.0) const
  {
      // 在度量Ｍ下,　计算距离 P，Q两点为h 的点，在方向向量PQ的右侧
      typename K::Construct_bisector_2 construct_bisector_2;
      typename K::Construct_point_2 construct_point_2;
      typedef typename K::FT        FT;
      Point_2 o(( p.x()+q.x() )/2, ( p.y()+q.y() )/2);
        Line_2 L = construct_bisector_2(p,q);
        FT A = M.dot_product(c,-L.b(),L.a());
        FT B = - M.dot_product(c,q.x()-o.x(),q.y()-o.y(),-L.b(),L.a());
        FT C = M.dot_product(c,q.x()-o.x(), q.y()-o.y()) - h*h;

        if (B*B-4*A*C >= 0)
        {
            FT t1 = (-B + CGAL::sqrt(B*B-4*A*C))/(2*A);
            FT t2 = (-B - CGAL::sqrt(B*B-4*A*C))/(2*A);

            //      Point_2 r1(o.x()-t1*L.b(),o.y()+t1*L.a());
            Point_2 r2(o.x()-t2*L.b(),o.y()+t2*L.a());

            if ( area(p,q,r2) < 0) return construct_point_2(o.x()-t2*L.b(),o.y()+t2*L.a());
            else return construct_point_2(o.x()-t1*L.b(),o.y()+t1*L.a());
        }
        else
        {
            B = - M.dot_product(c,o.x()-p.x(),o.y()-p.y(),-L.b(),L.a());
            C = M.dot_product(c,o.x()-p.x(), o.y()-p.y()) - h*h;
            if (B*B-4*A*C >= 0)
            {
                FT t1 = (-B + CGAL::sqrt(B*B-4*A*C))/(2*A);
                FT t2 = (-B - CGAL::sqrt(B*B-4*A*C))/(2*A);

                //      Point_2 r1(o.x()-t1*L.b(),o.y()+t1*L.a());
                Point_2 r2(o.x()-t2*L.b(),o.y()+t2*L.a());

                if ( area(p,q,r2) < 0) return construct_point_2(o.x()-t2*L.b(),o.y()+t2*L.a());
                else return construct_point_2(o.x()-t1*L.b(),o.y()+t1*L.a());
            }
            else
            {
                std::cout << "The bisector point is midpoint." << std::endl;
                return o;
            }
        }
  }

};

}



template<class K, class MBase, Metric_field_2<K,MBase> & M >
  class Anisotropic_kernel: public K
{
public:
   typedef  CartesianKernelFunctors::Side_of_oriented_anisotropic_circle_2<K, MBase, M>
Side_of_oriented_circle_2;
   typedef  CartesianKernelFunctors::Construct_anisotropic_bisector_2<K,MBase,M>
 Construct_bisector_2;
   typedef  CartesianKernelFunctors::Construct_anisotropic_circumcenter_2<K,MBase,M>
 Construct_circumcenter_2;
   typedef  CartesianKernelFunctors::Compute_anisotropic_area_2<K,MBase,M>
Compute_area_2;
   typedef CartesianKernelFunctors::Compute_anisotropic_squared_distance_2<K,MBase,M>
      Compute_squared_distance_2;
   typedef CartesianKernelFunctors::Construct_anisotropic_bisector_point_2<K,MBase,M>
         Construct_bisector_point_2;

public:
Side_of_oriented_circle_2 side_of_oriented_circle_2_object() const
 {return Side_of_oriented_circle_2();}
Construct_bisector_2 construct_bisector_2_object() const
 {return Construct_bisector_2();}
Construct_circumcenter_2 construct_circumcenter_2_object() const 
{return Construct_circumcenter_2();}
Compute_area_2  compute_area_2_object() const 
{return Compute_area_2();}

Compute_squared_distance_2 compute_squared_distance_2_object() const
{return Compute_squared_distance_2();}

Construct_bisector_point_2 construct_bisector_point_2_object() const
{return Construct_bisector_point_2(); }

};
}

#endif // ANISOTROPIC_KERNEL_H
