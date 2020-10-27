
#include <iostream>

#include "algebra/Matrix.h"

typedef WHYSC::AlgebraObject::Matrix<double> Matrix;

int main(int argc, char **argv)
{
    Matrix M(10, 5, 1.0);

    M.print();

    return 0;
}
