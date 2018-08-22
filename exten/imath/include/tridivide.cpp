#include "mex.h"
#include "matrix.h"

#include <iostream>
#include "CSRMatrix.h"
#include "Vector.h"

typedef iMath::LinearAlgebra::CSRMatrix<int, double>  CSRMatrix;
typedef iMath::LinearAlgebra::Vector<int, double>  Vector;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Declare variables */ 
    int n_rows;
    int nnz;
    double *data = NULL;
    double *b=NULL;
    double *x0=NULL;
    double *x1=NULL;

    mwIndex * m_indices = NULL;
    mwIndex * m_indptr = NULL;

    /* Get the data */
    n_rows = (int) mxGetM(prhs[0]);
    nnz = (int) mxGetNzmax(prhs[0]);
    std::cout<<"nnz:" << nnz << std::endl;
    std::cout<<"n_rows:" << n_rows << std::endl;

    data = mxGetPr(prhs[0]);
    m_indices = mxGetIr(prhs[0]);
    m_indptr = mxGetJc(prhs[0]);

    int indices[nnz];
    int indptr[n_rows+1];

    std::cout<< "test 0!" << std::endl;
    for(int i = 0; i < nnz; i++)
    {
        indices[i] = m_indices[i];
    }
    std::cout<< "test 1!" << std::endl;
    for(int i = 0; i < n_rows+1; i++)
    {
        indptr[i] = m_indptr[i];
    }


    b = mxGetPr(prhs[1]);

    plhs[0]=mxCreateDoubleMatrix(n_rows, 1, mxREAL);
    x0=mxGetPr(plhs[0]);

    plhs[1]=mxCreateDoubleMatrix(n_rows, 1, mxREAL);
    x1=mxGetPr(plhs[1]);

    CSRMatrix A(data, indices, indptr, nnz, n_rows, n_rows);
    Vector X0(x0, n_rows);
    Vector X1(x1, n_rows);
    Vector bb(b, n_rows);
    std::cout<< "test 2!" << std::endl;

    A.print();
    bb.print();
    A.tdivide(bb, X0, true);
    A.tdivide(bb, X1, false);
    return;
}
