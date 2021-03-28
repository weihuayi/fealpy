#ifndef CSRMatrix_h
#define CSRMatrix_h

#include <iostream>
#include <cmath>
#include <initializer_list>
#include <cassert>

#endif // end of CSRMatrix_h

namespace WHYSC {
namespace AlgebraObject {


template<typename F=double, typename I=int>
struct CSRMatrix
{
    typedef F Float;
    typedef I Int;

    F *data; // 非零元数组
    I *indices; // 非零元对应列指标数组
    I *indptr; // 非零元行起始位置数组
    I shape[2]; // 矩阵的阶数
    I nnz; // 非零元的个数

    CSRMatrix()
    {
        data = NULL;
        indices = NULL;
        indptr = NULL;
        shape[0] = 0;
        shape[1] = 0;
        nnz = 0;
    }

    template<typename Vector> 
    CSRMatrix(Vector & d, Vector & idx, Vector & idxp, I nrow, I ncol)
    {
        shape[0] = nrow;
        shape[1] = ncol;
        nnz = d.size();

        data = new F[nnz];
        indices = new I[nnz];
        indptr = new I[nrow + 1];

        for(auto i=0; i < nnz; i++)
        {
            data[i] = d[i];
            indices[i] = idx[i];
        }

        for(auto i=0; i < nrow+1; i++)
        {
            indptr[i] = idxp[i];
        }
    }


    template<typename Matrix> 
    CSRMatrix(Matrix & m)
    {
        shape[0] = m.shape[0];
        shape[1] = m.shape[1];
        nnz = 0;
        for(auto i=0; i < shape[0]; i++)
        {
            for(auto j=0; j < shape[1]; j++)
            {
                if( m[i][j] != 0)
                    nnz++;
            }
        }

        data = new F[nnz];
        indices = new I[nnz];
        indptr = new I[shape[0] + 1];
        indptr[shape[0]] = nnz;
        auto c = 0;
        for(auto i=0; i < shape[0]; i++)
        {
            indptr[i] = c;
            for(auto j=0; j < shape[1]; j++)
            {
                if( m[i][j] != 0)
                {
                    data[c] = m[i][j];
                    indices[c] = j;
                    c++;
                }
            }
        }
    }

    F operator() (const I i, const I j)
    {
        for(auto k = indptr[i]; k < indptr[i+1]; k++)
        {
            if(indices[k] == j)
            {
                return data[k];
            }
        }
        return 0.0;
    }

    ~CSRMatrix()
    {
        if(data != NULL)
            delete [] data;

        if(indices != NULL)
            delete [] indices;

        if(indptr != NULL)
            delete [] indptr;
    }
};

template<typename F, typename I, typename Vector>
inline Vector operator * (const CSRMatrix<F, I> & m, 
        const Vector & v0)
{
    assert( m.shape[1] == v0.size );
    Vector v1(m.shape[0], 0.0);
    for(auto i = 0; i < m.shape[0]; i++)
    {
        for(auto k = m.indptr[i]; k < m.indptr[i+1]; k++)
        {
            auto j = m.indices[k]; // 列指标
            v1[i] += m.data[k]*v0[j];
        }
    }
    return v1;
}

template<typename F, typename I>
std::ostream& operator << (std::ostream & os, const CSRMatrix<F, I> & m)
{
    std::cout << "CSRMatrix("<< m.shape[0] << ","
        << m.shape[1] << "):" << std::endl;

    for(auto i = 0; i < m.shape[0]; i++)
    {
        for(auto j = m.indptr[i]; j < m.indptr[i+1]; j++)
        {
            os << "(" << i << ", " << m.indices[j] << 
                ") " << m.data[j] << std::endl;
        }
    }
    os << "\n";
    return os;
}

} // end of namespace AlgebraObject
} // end of namespace WHYSC
