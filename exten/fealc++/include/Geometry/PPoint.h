#ifndef PPoint_h
#define PPoint_h

#include "Geometry/Point.h"
#include "Geometry/Vector.h"

namespace iMath {

namespace GeometryObject {

template<int DIM>
class PPoint
{
public:

    PPoint()
    {
        m_data = NULL;
    }

    PPoint(const double * d)
    {
        m_data = d;
    }

    int dimension() {return DIM;}

    double * & data() {return m_data;}
    const double * & data() const {return m_data;}
    
    double & operator[](const int i) 
    {
        return m_data[i];
    }

    const double & operator[](const int i) const
    {
        return m_data[i];
    }

    template<class Vector>
    PPoint<DIM> & operator -= (const Vector & rhs)
    {
        for(int d = 0; d < DIM; d++)
            m_data[d] -= rhs[d];
        return *this;
    }

    template<class Vector>
    PPoint<DIM> & operator += (const Vector & rhs)
    {
        for(int d = 0; d < DIM; d++)
            m_data[d] += rhs[d];
        return *this;
    }

private:
    double * m_data;
};

template<int DIM >
inline Vector<DIM> operator - (const PPoint<DIM> & p, const PPoint<DIM> & q)
{
    Vector<DIM> v;
    for(int d = 0; d < DIM; d++)
        v[d] = p[d] - q[d];
    return v;
}

template<int DIM >
inline Vector<DIM> operator - (const PPoint<DIM> & p, const Point<DIM> & q)
{
    Vector<DIM> v;
    for(int d = 0; d < DIM; d++)
        v[d] = p[d] - q[d];
    return v;
}

template<int DIM >
inline Vector<DIM> operator - ( const Point<DIM> & p, const PPoint<DIM> & q)
{
    Vector<DIM> v;
    for(int d = 0; d < DIM; d++)
        v[d] = p[d] - q[d];
    return v;
}

template<int DIM>
inline Point<DIM> operator + (const PPoint<DIM> & p, const Vector<DIM> & v)
{
    GeometryObject::Point<DIM> q;
    for(int d = 0; d < DIM; d++)
        q[d] = p[d] + v[d]; 
    return q;
}

template<int DIM>
inline Point<DIM> operator - (const PPoint<DIM> & p, const Vector<DIM> & v)
{
    Point<DIM> q;
    for(int d = 0; d < DIM; d++)
        q[d] = p[d] - v[d]; 
    return q;
}

template<int DIM>
std::ostream& operator << (std::ostream & os, const PPoint<DIM> & p)
{
    if( DIM == 2)
        return os << "PPoint_2(" << p[0] << ", " <<p[1] <<')';
    else if( DIM == 3)
        return os << "PPoint_3(" << p[0] << ", " << p[1] << ", " << p[2] << ')';
    else
        return os;
}

} // end of namespace Mesh

} // end of namespace iMath

#endif // end of PPoint_h 
