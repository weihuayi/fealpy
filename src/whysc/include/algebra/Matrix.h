#ifndef Matrix_h
#define Matrix_h

#include <iostream>

namespace WHYSC {
namespace AlgebraObject {

template<typename F=double>
struct Matrix
{
    F ** data;
    int shape[2];

    Matrix(int nr, int nc, F fill=0.0)
    {
        shape[0] = nr;
        shape[1] = nc;
        data = new F*[nr];
        for(int i=0; i < nr; i++)
        {
            data[i] = new F[nc];
            for(int j = 0; j < nc; j++)
                data[i][j] = fill;
        }

    }

    ~Matrix()
    {
        delete [] data;
    }

    F * operator[](const int i) 
    {
        return data[i];
    }

    void print()
    {
        std::cout << "Matrix("<< shape[0] << ","
            << shape[1] << ")" << std::endl;

        for(int i = 0; i < shape[0]; i++)
        {
            for(int j = 0; j < shape[1]; j++)
            {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
        
    }

};

} // end of namespace AlgebraObject

} // end of namespace WHYSC
#endif // end of Matrix_h
