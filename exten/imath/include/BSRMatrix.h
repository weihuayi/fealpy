#ifndef BSRMatrix_h
#define BSRMatrix_h

#include <new>
#include "type.h"

#include <vector>
#include <algorithm>
#include <functional>

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

    BSRMatrix(
            CSRMatrix<I, T> & A,
            I _R,
            I _C
            )
    {
        blocksize[0] = _R;
        blocksize[1] = _C;
        ndim = 2;
        shape[0] = A.shape[0];
        shape[1] = A.shape[1];
        std::vector<T*> blocks(n_col/C + 1, (T*)0 );

        assert( n_row % R == 0 );
        assert( n_col % C == 0 );

        I n_brow = n_row / R;
        //I n_bcol = n_col / C;

        I RC = R*C;
        I n_blks = 0;

        Bp[0] = 0;

        for(I bi = 0; bi < n_brow; bi++){
            for(I r = 0; r < R; r++){
                I i = R*bi + r;  //row index
                for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
                    I j = Aj[jj]; //column index

                    I bj = j / C;
                    I c  = j % C;

                    if( blocks[bj] == 0 ){
                        blocks[bj] = Bx + RC*n_blks;
                        Bj[n_blks] = bj;
                        n_blks++;
                    }

                    *(blocks[bj] + C*r + c) += Ax[jj];
                }
            }

            for(I jj = Ap[R*bi]; jj < Ap[R*(bi+1)]; jj++){
                blocks[Aj[jj] / C] = 0;
            }

            Bp[bi+1] = n_blks;
        }
    }
};

} // end of namespace LinearAlgebra

} // end of namespace iMath
#endif // end of BSRMatrix_h
