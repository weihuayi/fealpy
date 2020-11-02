
#include <iostream>

#include "algebra/Matrix.h"

typedef WHYSC::AlgebraObject::Matrix<double, int> Matrix;

int main(int argc, char **argv)
{

    double * data = new double[10];
    std::fill_n(data, 10, 2.345);

    for(auto i = 0; i < 10; i++)
        std::cout << data[i] << " ";
    std::cout << "\n";
    delete [] data;

    Matrix M(10, 5);
    std::fill_n(M.data, 50, 100);
    M.print();
    M.fill(2.0);
    std::cout << M;


    return 0;
}
