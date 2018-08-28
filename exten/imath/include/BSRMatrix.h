#ifndef BSRMatrix_h
#define BSRMatrix_h

#include <new>
#include "type.h"

#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>

namespace iMath {

namespace LinearAlgebra {

template<class I, class T>
class BSRMatrix
{
public:
    T * data;
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
            I _R, 
            I _C,
            bool _has_sorted_indices=true
            )
    {
        data = _data;
        indices = _indices;
        indptr = _indptr;
        blocksize[0] = _R;
        blocksize[1] = _C;
        ndim = 2;
        shape[0] = _n_rows;
        shape[1] = _n_cols;
        nnz = _nnz;
    }

    /* 转换 CSR 格式为 BSR 格式
     *
     *
     */
    BSRMatrix(
            CSRMatrix<I, T> & A,
            I R,
            I C
            )
    {
        ndim = 2;
        blocksize[0] = R;
        blocksize[1] = C;
        shape[0] = A.shape[0];
        shape[1] = A.shape[1];

        // 分配内存
        I n_blocks = A.count_blocks(R, C);
        nnz = n_blocks*R*C;

        indptr = new I[shape[0]/R + 1];
        indices = new I[n_blocks];
        data = new T[nnz];
        std::fill(data, data + nnz, 0.0);

        // 转换格式
        std::vector<T*> blocks(shape[1] / C + 1, (T*)0 );

        assert( shape[0] % blocksize[0] == 0 );
        assert( shape[1] % blocksize[1] == 0 );

        I n_brow = shape[0] / R;
        I RC = R*C;


        indptr[0] = 0;
        n_blocks = 0;
        for(I bi = 0; bi < n_brow; bi++)
        {
            for(I r = 0; r < R; r++)
            {
                I i = R*bi + r;  //行指标
                for(I jj = A.indptr[i]; jj < A.indptr[i+1]; jj++)
                {
                    I j = A.indices[jj]; // 列指标

                    I bj = j / C;
                    I c  = j % C;

                    if( blocks[bj] == 0 )
                    {
                        blocks[bj] = data + RC*n_blocks;
                        indices[n_blocks] = bj;
                        n_blocks++;
                    }

                    *(blocks[bj] + C*r + c) += A.data[jj];
                }
            }

            for(I jj = A.indptr[R*bi]; jj < A.indptr[R*(bi+1)]; jj++)
            {
                blocks[A.indices[jj]/C] = 0;
            }

            indptr[bi+1] = n_blocks;
        }
    }

    void print(bool showelem=true)
    {
        std::cout<<"matrix shape (" << shape[0] <<","<< shape[1] <<")"<<std::endl;
        if(showelem)
        {
            I n_brow = shape[0] / blocksize[0];
            I RC = blocksize[0]*blocksize[1];
            I n_blocks = nnz/RC;
            for(I bi = 0; bi < n_brow; bi++)
            {
                for(I bj = indptr[bi]; bj < indptr[bi+1]; bj++)
                {
                    std::cout << data[bj*RC] << "," << data[bj*RC + 1] << "\n"
                              << data[bj*RC + 2] << "," << data[bj*RC + 3] << std::endl;
                    std::cout << "\n";
                }
            }
        }
    }

    void symmetric_strength_of_connection(CSRMatrix<I, T> & C, double theta)
    {
        I RC = blocksize[0]*blocksize[1];
        I n_blocks = nnz/RC;
        std::vector<T> c(n_blocks, 0.0);
        for(I bi = 0; bi < n_blocks; bi++)
        {
            data[bi] += data[bi*RC]*data[bi*RC];
            data[bi] += data[bi*RC + 1] * data[bi*RC + 1];
            data[bi] += data[bi*RC + 2] * data[bi*RC + 2];
            data[bi] += data[bi*RC + 3] * data[bi*RC + 3];
        }

        CSRMatrix<I, T> A(
                c.data(), 
                indices, 
                indptr,
                nnz/RC,
                shape[0]/blocksize[0],
                shape[1]/blocksize[1]);

        C.reinit(A.nnz, A.shape[0], A.shape[1]);

        A.symmetric_strength_of_connection(C, theta);
    }
};

} // end of namespace LinearAlgebra

} // end of namespace iMath
#endif // end of BSRMatrix_h
