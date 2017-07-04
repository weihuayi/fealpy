#ifndef Vector_h
#define Vector_h

#include <cassert>

namespace  iMath {

namespace GeometryObject {

template<int DIM>
class Vector
{
public:
    Vector()
    {
        for(int d = 0; d < DIM; d++)
            _data[d] = 0;
    }

    Vector(const Vector<DIM> & v)
    {
        for(int d = 0; d < DIM; d++)
            _data[d] = v[d];
    }

    double & operator [] (const int i) 
    {
        return _data[i];
    }

    const double & operator [] (const int i) const
    {
        return _data[i];
    }

    int dimension() const {return DIM;}

    double squared_length()
    {
        double sum = 0.0;
        for(int d = 0; d < DIM; d++)
            sum += _data[d]*_data[d];
        return sum;
    }

    template<class RVector>
    double operator * (const RVector & w)
    {
        double sum = 0.0;
        for(int d = 0; d < DIM; d++)
            sum += _data[d]*w[d];
        return sum;
    }

    Vector<DIM> & operator *= (const double & s)
    {
        for(int d = 0; d < DIM; d++)
            _data[d] *= s;
        return * this;
    }


    Vector<DIM> & operator /= (const double & s)
    {
        for(int d = 0; d < DIM; d++)
            _data[d] /= s;
        return * this;
    }

    template<class RVector>
    Vector<DIM> & operator += (const RVector & w)
    {
        for(int d = 0; d < DIM; d++)
            _data[d] += w[d];
        return * this;
    }


    template<class RVector>
    Vector<DIM> & operator -= (const RVector & w)
    {
        for(int d = 0; d < DIM; d++)
            _data[d] -= w[d];
        return * this;
    }

private:
    double _data[DIM];
};


template<int DIM>
std::ostream& operator << (std::ostream & os, const Vector<DIM> & v)
{
    int dim = v.dimension();
    if( dim == 2)
        return os << "Vector_2(" << v[0] << ", " <<v[1] <<')';
    else if( dim == 3)
        return os << "Vector_3(" << v[0] << ", " << v[1] << ", " << v[2] << ')';
    else
        return os;
}

template<int DIM>
inline double squared_length(const Vector<DIM> & v)
{
    double sum = 0.0;
    for(int d = 0; d < DIM; d++)
        sum += v[d]*v[d];
    return sum;
}

template<int DIM>
inline Vector<DIM> cross(const Vector<DIM> & v, const Vector<DIM> & w)
{
    assert( DIM == 3);
    return Vector<DIM>(v[1]*w[2] - v[2]*w[1], 
           v[2]*w[0] - v[0]*w[2], 
           v[0]*w[1] - v[1]*w[0]);
}

template<int DIM>
inline double dot(const Vector<DIM> & v, const Vector<DIM> & w)
{
    double sum = 0.0;
    for(int d = 0; d < DIM; d++)
        sum += v[d]*w[d];

    return sum;
}

template<int DIM>
inline double operator * (const Vector<DIM> & v, const Vector<DIM> & w)
{
    double sum = 0.0;
    for(int d = 0; d < DIM; d++)
        sum += v[d]*w[d];

    return sum;
}

template<int DIM>
inline Vector<DIM> operator * (const Vector<DIM> & v, const double & s)
{
    Vector<DIM> rv;
    for(int d = 0; d < DIM; d++)
        rv[d] = v[d]*s; 
    return rv; 
}

template<int DIM>
inline Vector<DIM> operator * (const double & s, const Vector<DIM> & v)
{
    Vector<DIM> rv;
    for(int d = 0; d < DIM; d++)
        rv[d] = v[d]*s; 
    return rv; 
}

} // end of namespace GeometryObject

} // end of namespace iMath

#endif
