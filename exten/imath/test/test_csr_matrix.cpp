#include <iostream>
#include "CSRMatrix.h"
#include "Vector.h"

typedef iMath::LinearAlgebra::CSRMatrix<int, double>  CSRMatrix;
typedef iMath::LinearAlgebra::Vector<int, double>  Vector;
int main()
{
    double data[] = {1, 1, 2, 1};
    int indices[] = {0, 2, 2, 1};
    int indptr[] = {0, 2, 3, 4};
    int nnz = 4;
    int n_rows = 3;
    int n_cols = 3;

    CSRMatrix m(data, indices, indptr, nnz, n_rows, n_cols, G);
    m.print();

    Vector v(data, 4);
    v.print();
    return 0;
}

