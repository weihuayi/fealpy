/**
 * @file Metric_2.h
 * @author Wei, Huayi 
 * @date 2014-03-17 17:21:55
 * 
 * @brief 提供一个各向异性的度量类 
 */

#ifndef METRIC_2_H
#define METRIC_2_H 


#include "Metric_field_base_2.h"

namespace CGAL {
	
/**
 * @brief 二维度量类
 *
 * \f[
 *   M = \left(\begin{array}{rr}
 *         a(x,y) & b(x,y) \\
 *         b(x,y) & c(x,y)
 *       \end{array}\right)
 * \f]
 * 对任意的点 \f$ (x,y) \f$, \f$ M \f$ 对称正定，
 * 即 \f$ a*c-b^2 > 0 \f$。
 * 
 */


	
template<class K, class MBase = Metric_field_base_2<K> >
class Metric_field_2: public MBase
{
public:
	typedef typename K::FT FT;
	typedef typename K::Point_2 Point_2;
	typedef typename K::Vector_2 Vector_2;
	typedef typename K::Triangle_2 Triangle_2;


public:
    Metric_field_2(FT h1, FT h2, FT theta):MBase(h1,h2,theta){}
    Metric_field_2(FT h1, FT h2):MBase(h1,h2){}
    Metric_field_2():MBase(){}

    inline FT a11(FT x,FT y)
    {
        return MBase::a11(x,y);
    }

    inline FT a12(FT x,FT y)
    {
        return MBase::a12(x,y);
    }

    inline FT a22(FT x,FT y)
    {
        return MBase::a22(x,y);
    }

    inline FT a11(Point_2 & p)
    {
        return MBase::a11(p.x(),p.y());
    }

    inline FT a12(Point_2 & p)
    {
        return MBase::a12(p.x(),p.y());
    }

    inline FT a22(Point_2 & p)
    {
        return MBase::a22(p.x(),p.y());
    }

    inline FT dot_product(Point_2 & p, FT x, FT y)
	{

        return a11(p)*x*x + 2.0*a12(p)*x*y+a22(p)*y*y;
	}

    inline FT dot_product(const Point_2 & p, FT x, FT y)
    {
        Point_2 p0 = p;
        return dot_product(p0,x,y);
    }

    inline FT dot_product(Point_2 &p, FT v1x, FT v1y, FT v2x, FT v2y)
    {
        return a11(p)*v1x*v2x + a22(p)*v1y*v2y + a12(p)*v1x*v2y + a12(p)*v1y*v2x;
    }

    inline FT dot_product(const Point_2 &p, FT v1x, FT v1y, FT v2x, FT v2y)
    {
        Point_2 p0 = p;
        return dot_product(p0,v1x,v1y,v2x,v2y);
    }

    /// 2014.10.28
    /*!
     * \brief 计算两点 q1,q2 在度量 p 下的黎曼距离平方
     * \param p 度量 M(p)
     * \param q1,q2 两点
     * \return 黎曼距离平方
     */
    inline FT dot_product(Point_2 &p, Point_2 &q1, Point_2 &q2)
    {
        FT v1x = q2.x() - q1.x();
        FT v1y = q2.y() - q1.y();
        return dot_product(p,v1x,v1y);
    }

    inline FT dot_product(const Point_2 &p, const Point_2 &q1, const Point_2 &q2)
    {
        Point_2 p0 = p;
        Point_2 p1 = q1;
        Point_2 p2 = q2;
        return dot_product(p0,p1,p2);
    }

	inline FT determinant(Point_2 & p)
	{

        return a11(p)*a22(p) - a12(p)*a12(p);

	}

	inline FT determinant(Point_2 & p, FT v1x, FT v1y, FT v2x, FT v2y)
	{

        return (a11(p)*v1x + a12(p)*v1y)*(a12(p)*v2x + a22(p)*v2y)
            - (a12(p)*v1x+a22(p)*v1y)*(a11(p)*v2x+a12(p)*v2y);
	}


};

}

#endif //METRIC_2_H
