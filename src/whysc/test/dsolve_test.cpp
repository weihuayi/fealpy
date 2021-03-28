#include <iostream>

#include "algebra/Algebra_kernel.h"

typedef typename WHYSC::Algebra_kernel<double, int> Kernel;
typedef typename Kernel::Matrix Matrix;
typedef typename Kernel::Vector Vector;

int main(int argc, char **argv)
{
    //Matrix M{{1,2,3},{2,5,2},{3,1,5}};

    Matrix M{{4,5,6}, {8,17,20}, {12,43,59}};
    std::cout << "M:\n" << M << std::endl;

    auto n = M.shape[0];
    Matrix L(n, n);
    Matrix U(n, n);

    Kernel::lu(M, L, U);
    std::cout << "M:\n" << M << std::endl;
    std::cout << "L:\n" << L << std::endl;
    std::cout << "U:\n" << U << std::endl;

    Matrix C = L*U;

    std::cout << C << std::endl;

    //Matrix A{{1,2,3},{2,5,2},{3,1,5}};
    Matrix A{{4,5,6}, {8,17,20}, {12,43,59}};
    Matrix E = C - A;
    std::cout << "E.norm:" << E.norm() << std::endl;

    Vector v = {1, 1, 1};
    std::cout << v.norm() << std::endl;

    return 0;
}
