#ifndef Array_h
#define Array_h

#include <array>
#include <algorithm>
#include <initializer_list>
#include <assert.h>

namespace WHYSC {

namespace AlgebraObject {

template<typename F=double, typename I=int>
struct Array
{
    /*Data member*/
    F * data;
    I size;
    bool fromother;
    std::vector<I> shape;

    /*Function member*/
    Array()
    { 
        data = NULL;
        fromother = false;
    }

    Array(const std::initializer_list<I> &l)
    { 
        shape.resize(l.size());
        std::copy_n(l.begin(), l.size(), shape.data());
        size = 1;
        for(auto it = l.begin(); it !=l.end(); it++)
            size *= &it;
        data = new F[size];
    }

    ~Array()
    {
        if(data != NULL) && (!fromother))
            delete[] data;
    }


    void fill(F a=0)
    {
        std::fill_n(data, size, a);
    }

    F & operator[](const I i, const I j) 
    {
        return _data[i*shape[0]+j];
    }

    const F & operator[](const I i, const I j) const
    {
        return _data[i*shape[0]+j];
    }

};


template<typename F, typename I>
std::ostream& operator << (std::ostream & os, const Array<F, I> & a)
{
    for(auto i = 0; i < a.shape[0]; i ++)
    {
        for(auto j = 0; j < a.shape[1]; j++)
        {
            os << a[i, j] << " "; 
        }
        os << "\n";
    }
}

} // end of namespace AlgebraObject

} // end of namespace WHYSC
#endif // end of Matrix_h
