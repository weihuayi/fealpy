#ifndef BSRMatrix_h
#define BSRMatrix_h

#include <new>
#include "type.h"
#include <vector>

namespace iMath {

namespace LinearAlgebra {

template<class I, class F>
class BSRData
{
public:
    F * data;
    I shape[3];
    bool from_other;
public:
    BSRData(I _M, I _N, I _R)
    {
        data = new F[_M*_N*_R];
        shape[0] = _M;
        shape[1] = _N;
        shape[2] = _R;
        from_other = false;
    }
    BSRData(F * _data, I _M, I _N, I _R)
    {
        data = _data;
        shape[0] = _M;
        shape[1] = _N;
        shape[2] = _R;
        from_other = true;
    }

    ~BSRData()
    {
        if( from_other == false & data != NULL)
        {
            delete[] data;
        }
    }
};

template<class I, class F>
class BSRMatrix
{
public:
    BSRData data;
    I * indices;
    I * indptr;
    I shape[2];
    I ndim;
    I blocksize[2];
    I nnz;
public:
};

} // end of namespace LinearAlgebra

} // end of namespace iMath
#endif // end of BSRMatrix_h
