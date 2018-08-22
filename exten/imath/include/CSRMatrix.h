#ifndef CSCMatrix_h
#define CSCMatrix_h

#include <new>
#include "type.h"
#include <vector>

namespace iMath {

namespace LinearAlgebra {

template<class I, class F>
class CSRMatrix
{
public:
    typedef I Int;
    typedef F Float;
public:
    I nnz;
    I ndim;
    I shape[2];

    F * data;
    I * indices;
    I * indptr;

    bool has_sorted_indices;
    bool from_other;
    MatType mat_type;

    SparseType sparse_type;

public:
    CSRMatrix(
            F * _data, 
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

        data = new F[nnz];
        indices = new I[nnz];
        indptr = new I[shape[0]+1];
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
    void symmetric_strength_of_connection(CSRMatrix<I, F> & S, F theta)
    {
        std::vector<F> diags(shape[0]);

        //compute norm of diagonal values
        for(I i = 0; i < shape[0]; i++)
        {
            F diag = 0.0;
            for(I j = indptr[i]; j < indptr[i+1]; j++)
            {
                if(indices[j] == i)
                {
                    diag += data[j]; 
                }
            }
            diags[i] = std::abs(diag);
        }

        S.nnz = 0;
        S.indptr[0] = 0;

        for(I i = 0; i < shape[0]; i++)
        {
            F eps_Aii = theta*theta*diags[i];

            for(I jj = indptr[i]; jj < indptr[i+1]; jj++)
            {
                I j = indices[jj];
                if(i == j)
                {
                    // Always add the diagonal
                    S.indices[nnz] =   j;
                    S.data[nnz] = data[jj];
                    nnz++;
                }
                else if(data[jj]*data[jj] >= eps_Aii * diags[j]){
                    //  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|)
                    S.indices[nnz] =   j;
                    S.data[nnz] = data[jj];
                    nnz++;
                }
            }
            S.indptr[i+1] = nnz;
        }
    }

    I standard_aggregation(I x[],  I y[])
    {
        std::fill(x, x + shape[0], 0);

        I next_aggregate = 1; // number of aggregates + 1

        //Pass #1
        for(I i = 0; i < shape[0]; i++)
        {
            if(x[i]){ continue; } //already marked

            const I row_start = indptr[i];
            const I row_end   = indptr[i+1];

            //Determine whether all neighbors of this node are free (not already aggregates)
            bool has_aggregated_neighbors = false;
            bool has_neighbors            = false;
            for(I jj = row_start; jj < row_end; jj++)
            {
                const I j = indices[jj];
                if( i != j )
                {
                    has_neighbors = true;
                    if( x[j] )
                    {
                        has_aggregated_neighbors = true;
                        break;
                    }
                }
            }

            if(!has_neighbors)
            {
                //isolated node, do not aggregate
                x[i] = -shape[0];
            }
            else if (!has_aggregated_neighbors){
                //Make an aggregate out of this node and its neighbors
                x[i] = next_aggregate;
                y[next_aggregate-1] = i;              //y stores a list of the Cpts
                for(I jj = row_start; jj < row_end; jj++){
                    x[indices[jj]] = next_aggregate;
                }
                next_aggregate++;
            }
        }

        //Pass #2
        // Add unaggregated nodes to any neighboring aggregate
        for(I i = 0; i < shape[0]; i++)
        {
            if(x[i]){ continue; } //already marked

            for(I jj = indptr[i]; jj < indptr[i+1]; jj++)
            {
                const I j = indices[jj];
                const I xj = x[j];
                if(xj > 0)
                {
                    x[i] = -xj;
                    break;
                }
            }
        }

        next_aggregate--;

        //Pass #3
        for(I i = 0; i < shape[0]; i++)
        {
            const I xi = x[i];

            if(xi != 0)
            {
                // node i has been aggregated
                if(xi > 0)
                    x[i] = xi - 1;
                else if(xi == -shape[0])
                    x[i] = -1;
                else
                    x[i] = -xi - 1;
                continue;
            }

            // node i has not been aggregated
            const I row_start = indptr[i];
            const I row_end   = indptr[i+1];

            x[i] = next_aggregate;
            y[next_aggregate] = i;              //y stores a list of the Cpts

            for(I jj = row_start; jj < row_end; jj++)
            {
                const I j = indices[jj];
                if(x[j] == 0){ //unmarked neighbors
                    x[j] = next_aggregate;
                }
            }
            next_aggregate++;
        }


        return next_aggregate; //number of aggregates
    }

    template<class V>
    void gauss_seidel(V & b, V &x,
            const I row_start,
            const I row_stop,
            const I row_step
            )
    {
        for(I i = row_start; i != row_stop; i += row_step) 
        {
            I start = indptr[i];
            I end   = indptr[i+1];
            F rsum = 0;
            F diag = 0;

            for(I jj = start; jj < end; jj++)
            {
                I j = indices[jj];
                if (i == j)
                    diag  = data[jj];
                else
                    rsum += data[jj]*x[j];
            }

            if (diag != (F) 0.0){
                x[i] = (b[i] - rsum)/diag;
            }
        }
    }
};


} // end of namespace LinearAlgebra
} // end of namespace iMath 
#endif
