#ifndef Vector_h
#define Vector_h

#include <cmath>

namespace iMath {


namespace LinearAlgebra {

template<class I, class F>
class Vector
{
public:
    F * data;
    I size;

    bool from_other;
public:
    Vector()
    {
        data = NULL;
        size = 0;
        from_other = false;
    }

    Vector(F * _data, I _size)
    {
        size = _size;
        data = _data;
        from_other = true;
    }

    template<class V> 
    Vector(V & v)
    {
        size = v.size;
        data = new F[size];
        for(int i = 0; i < size; i++)
        {
            data[i] = v[i];
        }
        from_other = false;
    }

    Vector(I _size)
    {
        size = _size;
        data = new F[size];
        from_other = false;
    }

    ~Vector()
    {
        if( (from_other == false) & (data != NULL) )
        {
            delete[] data;
        }
    }

    F & operator [] (int i)
    {
        return data[i];
    }


    template<class V>
    void copy(V & v)
    {
        for(I i = 0; i < size; i++)
        {
            data[i] = v[i];
        }
    }

    F norm()
    {
        F r = 0.0;
        for(I i = 0; i < size; i++)
        {
            r += data[i]*data[i];
        }
        return sqrt(r);
    }

    template<class V> 
    F dot(V & v)
    {
        F sum = 0.0;
        for(I i=0; i < size; i++)
            sum += data[i]*v[i];
        return sum;
    }

    template<class V>
    void wplus(F alpha, V & v)
    {
        for(I i=0; i < size; i++)
            data[i] += alpha*v[i];
    }

    template<class V>
    void wminus(F alpha, V & v)
    {
        for(I i=0; i < size; i++)
            data[i] += alpha*v[i];
    }


    template<class V>
    void operator += (V & v)
    {
        for(I i=0; i < size; i++)
            data[i] += v[i];
    }

    template<class V>
    void operator -=(V & v){
        for(I i=0; i < size; i++)
            data[i] -= v[i];
    }

    template<class V>
    void operator /=(V & v){
        for(I i=0; i < size; i++)
            data[i] /= v[i];
    }

    template<class V>
    void operator *=(F a){
        for(I i=0; i < size; i++)
            data[i] *= a;
    }

    void print()
    {
        for(I i = 0; i < size; i++)
        {
            std::cout << data[i] << std::endl;
        }
    }
};

} // end of namespace LinearAlgebra

} // end of namespace iMath
#endif // end of Vector_h
