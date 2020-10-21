#ifndef Matrix_h
#define Matrix_h

#include <array>
#include <algorithm>
#include <initializer_list>
#include <assert.h>

namespace WHYSC {
namespace AlgebraObject {

template<typename F, int ROW, int COL>
struct Matrix: public std::array<F, ROW*COL>
{
    F & operator[](const I i, const I j) 
    {
        return _data[i*ROW+j];
    }

    const F & operator[](const I i, const I j) const
    {
        return _data[i*ROW+j];
    }
};

} // end of namespace AlgebraObject

} // end of namespace WHYSC
#endif // end of Matrix_h
