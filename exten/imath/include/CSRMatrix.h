#ifndef CSRMatrix_h
#define CSRMatrix_h

#include <new>
#include <vector>

#include "type.h"
#include "amg/smoothed_aggregation.h"

namespace iMath {

namespace LinearAlgebra {

template<class I, class T>
class CSRMatrix
{
public:
    I nnz;
    I ndim;
    I shape[2];

    T * data;
    I * indices;
    I * indptr;

    bool has_sorted_indices;
    bool from_other;
    MatType mat_type;

    SparseType sparse_type;

public:
    CSRMatrix()
    {
        ndim = 2;
        nnz = 0;
        data = NULL;
        indices = NULL;
        indptr = NULL;
        shape[0] = 0;
        shape[1] = 0;
        mat_type = G;
        sparse_type = CSR;
        has_sorted_indices = true;
        from_other = false;
    }


    CSRMatrix(
            T * _data, 
            I * _indices, 
            I * _indptr, 
            I _nnz, 
            I _n_rows, 
            I _n_cols,
            MatType _mat_type=G,
            bool _has_sorted_indices=true
            )
    {
        data = _data;
        indices = _indices;
        indptr = _indptr;
        nnz = _nnz;
        ndim = 2;

        shape[0] = _n_rows;
        shape[1] = _n_cols;
        has_sorted_indices = _has_sorted_indices;

        from_other = true;  

        mat_type = _mat_type;
        sparse_type = CSR;
    }

    CSRMatrix(
            I _nnz, 
            I _n_rows, 
            I _n_cols, 
            MatType _mat_type=G, 
            bool _has_sorted_indices=true
            )
    {
        has_sorted_indices = _has_sorted_indices;
        from_other = false;
        nnz = _nnz;
        shape[0] = _n_rows;
        shape[1] = _n_cols;
        mat_type = _mat_type;
        sparse_type = CSR;

        data = new T[nnz];
        indices = new I[nnz];
        indptr = new I[shape[0]+1];
    }

    template<class Index>
    CSRMatrix(
            Index * Ai,
            Index * Aj,
            T * Ax,
            I _nnz,
            I _n_rows,
            I _n_cols,
            MatType _mat_type=G, 
            bool _has_sorted_indices=true
            )
    {
        from_other = false;
        nnz = _nnz;
        shape[0] = _n_rows;
        shape[1] = _n_cols;
        mat_type = _mat_type;
        sparse_type = CSR;
        has_sorted_indices = _has_sorted_indices;

        data = new T[nnz];
        indices = new I[nnz];
        indptr = new I[shape[0]+1];

        //计算三元组每一行非零元的个数 
        std::fill(indptr, indptr + shape[0], 0);

        for (I n = 0; n < nnz; n++){            
            I i = Ai[n];
            indptr[i]++;
        }

        // 累加构造行指针数组
        for(I i = 0, cumsum = 0; i < shape[0]; i++)
        {     
            I temp = indptr[i];
            indptr[i] = cumsum;
            cumsum += temp;
        }

        indptr[shape[0]] = nnz; 

        // 把列指标 Aj, Ax 复制进 indices, data
        for(I n = 0; n < nnz; n++){
            I row  = Ai[n];
            I dest = indptr[row];

            indices[dest] = Aj[n];
            data[dest] = Ax[n];

            indptr[row]++;//TODO: WHY
        }

        for(I i = 0, last = 0; i <= shape[0]; i++){
            I temp = indptr[i];
            indptr[i]  = last;
            last  = temp;
        }

        // 可能有重复
    }


    void reinit(I _nnz, I _n_rows, I _n_cols)
    {
        shape[0] = _n_rows;
        shape[1] = _n_cols;

        from_other = false;
        data = new T[nnz];
        indices = new T[nnz];
        indptr = new T[shape[0]];
    }


    I count_blocks(const I R, const I C)
    {
        std::vector<I> mask(shape[1]/C + 1, -1);
        I n_blocks = 0;

        for(I i = 0; i < shape[0]; i++)
        {
            I bi = i/R;
            for(I jj = indptr[i]; jj < indptr[i+1]; jj++){
                I bj = indices[jj]/C;
                if(mask[bj] != bi){
                    mask[bj] = bi;
                    n_blocks++;
                }
            }
        }
        return n_blocks;
    }


    ~CSRMatrix()
    {
        if( from_other == false )
        {
            if( data != NULL )
                delete[] data;
            if( indices != NULL )
                delete[] indices;

            if( indptr != NULL )
                delete[] indptr;
        }
    }

    void print(bool showelem=true)
    {
        std::cout<<"matrix shape (" << shape[0] <<","<< shape[1] <<")"<<std::endl;
        if(showelem)
        {
            for(I i=0; i < shape[0]; i++)
            {
                for(I j = indptr[i]; j < indptr[i+1]; j++)
                {
                    std::cout<< j << std::endl;
                    std::cout<<"("<< i <<","<< indices[j] <<") " << data[j] << std::endl;
                }
            }
        }
    }

    template<class V>
    void diag(V & d)
    {
        for(I i = 0; i < shape[0]; i++)
        {
            for(I j = indptr[i]; j < indptr[i+1]; j++)
            {
                if(indices[j] == i)
                    d[i] += data[j];
            }
        }
    }

    template<class V>
    void vtimes(V & v, V & out)
    {
        for(I i=0; i < shape[0]; i++)
        {
            for(I j = indptr[i]; j < indptr[i+1]; j++)
            {
                out[i] += data[j]*v[indices[j]];
            }
        }
    }

    template<class V>
    void residual(V & b, V & x, V & r)
    {
        r.copy(b);
        for(I i = 0; i < shape[0] ; i++)
        {
            for(I j = indptr[i]; j < indptr[i+1]; j++)
            {
                r[i] -= data[j]*x[indices[j]];
            }
        }
    }


    template<class V>
    void tdivide(V & b, V & x)
    {
        x.copy(b);
        switch(mat_type)
        {
            case L:
                for(I i=0; i < shape[0]; i++)
                {
                    I j = indptr[i];
                    for(; j < indptr[i+1]-1; j++)
                    {
                        x[i] -= data[j]*x[indices[j]];
                    }

                    x[i] /= data[j];
                }
                break;
            case U:
                for(I i = shape[0] - 1; i > 0; i--)
                {
                    I j = indptr[i+1]-1;
                    for(I j = indptr[i+1] -1 ; j > indptr[i]; j--)
                    {
                        x[i] -= data[j]*x[indices[j]];
                    }
                    x[i] /=data[j];
                }
                break;
            case G:
                std::cout<<"This is not a triangle matrix\n"<<std::endl;
                break;
        }
    }

    template<class V>
    void tdivide(V & b, V & x, bool lt)
    {// 该程序要求, 
     //   1. 每一行的非零元按列指标的从小到大进行排序
     //   2. 每个主对角线元素非零
     //   3. 该矩阵是一个一般的CSR 矩阵
     
        x.copy(b);
        x.print();
        if(lt == true)
        {
            for(I i = 0; i < shape[0]; i++)
            {
                I j = indptr[i];
                for(; j < indptr[i+1]-1 & indices[j] != i; j++)
                {
                    x[i] -= data[j]*x[indices[j]];
                }

                x[i] /= data[j];
            }
        }
        else
        {
            for(I i = shape[0] - 1; i >= 0; i--)
            {
                I j = indptr[i+1]-1;
                for(; j >= indptr[i] & indices[j] != i; j--)
                {
                    x[i] -= data[j]*x[indices[j]];
                }
                x[i] /=data[j];
            }
        }
    }

    void symmetric_strength_of_connection(CSRMatrix<I, T> & S, T theta)
    {
        symmetric_strength_of_connection(shape[0],
                                         theta,
                                         indptr, shape[0]+1,
                                         indices, nnz,
                                         data, nnz,
                                         S.indptr, shape[0]+1,
                                         S.indices, nnz,
                                         S.data, nnz);
    }

    template<class V>
    I standard_aggregation(CSRMatrix<I, T> & Agg, V & Cpts)
    {
        std::vector<I> Aj(shape[0]);
        std::vector<I> C(shape[0]);
        I num_agg = standard_aggregation(shape[0], 
                indptr, 
                indices, 
                Aj.data(),
                C.data(),);

        Agg.reinit(shape[0], shape[0], num_agg);
        Cpts.reinit(num_agg);

        std::fill(Agg.data, Agg.data + Agg.nnz, 1.0);
        for(I i = 0; n < Agg.nnz; i++)
        {
            Agg.indices[i] = Aj[i];
        }
        for(I i = 0; i <= shape[0]; i++)
        {
            Agg.indptr[i] = i;
        }

    }

};


} // end of namespace LinearAlgebra
} // end of namespace iMath 
#endif
