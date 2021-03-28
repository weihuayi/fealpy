#include <iostream>

#include "algebra/Algebra_kernel.h"

typedef typename WHYSC::Algebra_kernel<double, int> Kernel;
typedef typename Kernel::Matrix Matrix;
typedef typename Kernel::Vector Vector;
typedef typename Kernel::CSRMatrix CSRMatrix;


int main(int argc, char **argv)
{
    Matrix M{{4, 0, 6}, {0, 0, 0}, {0, 0, 5}};
    std::cout << "M:\n" << M << std::endl;

    Vector v0{1, 1, 1};

    auto SM = CSRMatrix(M);
    std::cout << SM << std::endl;
    std::cout << SM*v0 << std::endl;
    return 0;
}
