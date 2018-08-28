#include "mex.h"
#include "matrix.h"

#include <iostream>
#include "CSRMatrix.h"
#include "BSRMatrix.h"
#include "Vector.h"

typedef iMath::LinearAlgebra::CSRMatrix<int, double>  CSRMatrix;
typedef iMath::LinearAlgebra::BSRMatrix<int, double>  BSRMatrix;
typedef iMath::LinearAlgebra::Vector<int, double>  Vector;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Declare variables */ 
    int n_rows;
    int n_cols;
    int nnz;

    double * I = NULL;
    double * J = NULL;
    double * data = NULL;
    double * b = NULL;

    nnz = mxGetM(prhs[2]);
    n_rows = mxGetM(prhs[3]);
    n_cols = n_rows;

    I = mxGetPr(prhs[0]);
    J = mxGetPr(prhs[1]);
    data = mxGetPr(prhs[2]);
    b = mxGetPr(prhs[3]);


    std::cout << "nnz: " << nnz << std::endl;
    std::cout << "n_rows: "<< n_rows << std::endl;
    std::cout << "n_cols: " << n_cols << std::endl;
    for(int i = 0; i < nnz; i++)
    {
        std::cout << " I[" << i << "]: " << I[i] << ", ";
        std::cout << " J[" << i << "]: " << J[i] << ", ";
        std::cout << " data[" << i << "]: " << data[i] << std::endl;
    }

    CSRMatrix  A(I, J, data, nnz, n_rows, n_cols);
    A.print();

    BSRMatrix B(A, 2, 2);
    B.print();
    return;
}
