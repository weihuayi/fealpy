#ifndef BSRMatrix_h
#define BSRMatrix_h

#include <new>
#include "type.h"
#include <vector>

namespace iMath {

namespace LinearAlgebra {

template<class I, class T>
class BSRData
{
public:
    T * data;
    I shape[3];
    bool from_other;
public:
    BSRData(I _M, I _N, I _K)
    {
        data = new T[_M*_N*_K];
        shape[0] = _M;
        shape[1] = _N;
        shape[2] = _K;
        from_other = false;
    }
    BSRData(T * _data, I _M, I _N, I _K)
    {
        data = _data;
        shape[0] = _M;
        shape[1] = _N;
        shape[2] = _K;
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

template<class I, class T>
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
    bool has_sorted_indices;
public:
    BSRMatrix(
            T * _data, 
            I * _indices, 
            I * _indptr, 
            I _nnz, 
            I _n_rows, 
            I _n_cols,
            I _M, 
            I _N, 
            I _K,
            bool _has_sorted_indices=true
            ): data(_data, _M, _N, _K)
    {
       indices = _indices;
       indptr = _indptr;
       blocksize[0] = _N;
       blocksize[1] = _K;
       ndim = 2;
       shape[0] = _n_rows * _N;
       shape[1] = _n_cols * _K;
       nnz = _M*_N*_K;
    }
};

} // end of namespace LinearAlgebra

} // end of namespace iMath
#endif // end of BSRMatrix_h
