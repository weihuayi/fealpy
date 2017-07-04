#ifndef Bisection_alg_h
#define Bisection_alg_h

#include <cmath>
namespace iMath {

namespace GeoAlg {

template<class GK> 
class Bisection_alg
{
public:
    typedef typename GK::Point_3 Point_3;
    typedef typename GK::Vector_3 Vector_3;

    typedef typename GK::Point_2 Point_2;
    typedef typename GK::Vector_2 Vector_2;
public:
    Bisection_alg(){}

    template<class LevelSetFunction >
    Point_3 operator () (LevelSetFunction & fun, const Point_3 & p1, const Point_3 & p2) 
    {
        Point_3 a = p1;
        Point_3 b = p2;
        Point_3 m = a + 0.5*(b - a); 
        double h = std::sqrt((b - a).squared_length());

        int a_sign = fun.sign(a);
        int b_sign = fun.sign(b);
        int m_sign = fun.sign(m);
        while(h > GK::eps())
        {
            if(m_sign == 0)
            {
                return m; 
            }
            else if( a_sign*m_sign < 0)
                b = m;
            else if( b_sign*m_sign < 0)
                a = m;
            m = a + 0.5*(b - a);
            h = std::sqrt((b - a).squared_length()); 
        }

        return m;
    }

    template<class LevelSetFunction >
    Point_2 operator () (LevelSetFunction & fun, const Point_2 & p1, const Point_2 & p2) 
    {
        Point_2 a = p1;
        Point_2 b = p2;
        int a_sign = fun.sign(a);
        int b_sign = fun.sign(b);

        double h = std::sqrt((b - a).squared_length());
        Point_2 m = midpoint(a,b); 
        int m_sign = fun.sign(m);
        while(h > GK::eps())
        {
            if(m_sign == 0)
            {
                return m; 
            }
            else if( a_sign*m_sign < 0)
            {
                b = m;
                b_sign = m_sign;
            }
            else if( b_sign*m_sign < 0)
            {
                a = m;
                a_sign = m_sign;
            }

            h = std::sqrt((b - a).squared_length()); 

            m = midpoint(a,b); 
            m_sign = fun.sign(m);
        }

        return m;
    }
};

} // end of namespace GeoAlg

} // end of namespace iMath
#endif // end of Bisection_alg_h
