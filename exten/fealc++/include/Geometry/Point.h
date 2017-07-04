#ifndef Point_h
#define Point_h

#include "Vector.h"

namespace  iMath {

namespace GeometryObject {

template<int DIM>
class Point
{
public:
    Point()
    {
        for(int d = 0; d < DIM; d++)
        {
            _data[d] = 0.0;
        }
    }
    
    Point(const Point & p)
    {
        for(int d = 0; d < DIM; d++)
            _data[d] = p[d];
    }

    Point(double * p)
    {
        for(int d = 0; d < DIM; d++)
            _data[d] = p[d];
    }

    template<class P>
    void copy_from(const P & p)
    {
        for(int d = 0; d < DIM; d++)
            _data[d] = p[d];
    }

    void copy_to(double * p)
    {
        for(int d = 0; d < DIM; d++)
            p[d] = _data[d];
    }

    static int dimension() {return DIM;}

    double * data() {return _data;}

    double & operator[](const int i) 
    {
        return _data[i];
    }

    const double & operator[](const int i) const
    {
        return _data[i];
    }
    
    template<class Vector>
    Point<DIM> & operator -= (const Vector & rhs)
    {
        for(int d = 0; d < DIM; d++)
            _data[d] -= rhs[d];
        return *this;
    }

    template<class Vector>
    Point<DIM> & operator += (const Vector & rhs)
    {
        for(int d = 0; d < DIM; d++)
            _data[d] += rhs[d];
        return *this;
    }

private:
    double _data[DIM];
};

template<int DIM >
inline Vector<DIM> operator - (const Point<DIM> & p,
                         const Point<DIM> & q)
{
    Vector<DIM> v;
    for(int d = 0; d < DIM; d++)
        v[d] = p[d] - q[d];
    return v;
}

template<int DIM>
inline Point<DIM> operator + (const Point<DIM> & p,
                        const Vector<DIM> & v)
{
    Point<DIM> q;
    for(int d = 0; d < DIM; d++)
        q[d] = p[d] + v[d]; 
    return q;
}

template<int DIM>
inline Point<DIM> operator - (const Point<DIM> & p,
                        const Vector<DIM> & v)
{
    Point<DIM> q;
    for(int d = 0; d < DIM; d++)
        q[d] = p[d] - v[d]; 
    return q;
}

template<int DIM>
std::ostream& operator << (std::ostream & os, const Point<DIM> & p)
{
    if( DIM == 2)
        return os << "Point_2(" << p[0] << ", " <<p[1] <<')';
    else if( DIM == 3)
        return os << "Point_3(" << p[0] << ", " << p[1] << ", " << p[2] << ')';
    else
        return os;
}


}// end of namespace GeometryObject

}// end of namesapce iMath

#endif
