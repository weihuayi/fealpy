#ifndef Vector_h
#define Vector_h

#include <cmath>

namespace iMath {


namespace LinearAlgebra {

template<class I, class T>
class Vector
{
public:
    T * data;
    I size;

    bool from_other;
public:
    Vector()
    {
        data = NULL;
        size = 0;
        from_other = false;
    }

    Vector(T * _data, I _size)
    {
        size = _size;
        data = _data;
        from_other = true;
    }

    template<class V> 
    Vector(V & v)
    {
        size = v.size;
        data = new T[size];
        for(int i = 0; i < size; i++)
        {
            data[i] = v[i];
        }
        from_other = false;
    }

    Vector(I _size)
    {
        size = _size;
        data = new T[size];
        from_other = false;
    }

    void reinit(I _size)
    {
        if( (from_other == false) & (data != NULL) )
        {
            delete[] data;
        }
        size = _size;
        data = new T[size];
        from_other = false;
    }


    ~Vector()
    {
        if( (from_other == false) & (data != NULL) )
        {
            delete[] data;
        }
    }

    T & operator [] (int i)
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

    T norm()
    {
        T r = 0.0;
        for(I i = 0; i < size; i++)
        {
            r += data[i]*data[i];
        }
        return sqrt(r);
    }

    template<class V> 
    T dot(V & v)
    {
        T sum = 0.0;
        for(I i=0; i < size; i++)
            sum += data[i]*v[i];
        return sum;
    }

    template<class V>
    void wplus(T alpha, V & v)
    {
        for(I i=0; i < size; i++)
            data[i] += alpha*v[i];
    }

    template<class V>
    void wminus(T alpha, V & v)
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
