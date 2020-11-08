#include <iostream>
#include <math.h>


#include "algebra/Algebra_kernel.h"

typedef typename WHYSC::Algebra_kernel<double, int> Kernel;
typedef typename Kernel::Matrix Matrix;


/*
#include "algebra/Matrix.h"

using namespace std;
typedef WHYSC::AlgebraObject::Matrix<double> Matrix;
*/

int main()
{
    int n=3;//n is dim of matrix
    double Larr2[n][n];
    double arr[n][n]={{4,5,6}, {8,17,20}, {12,43,59}};

    for(int i=0; i < n; i++)
        for(int j=0; j < n; j++)
            std::cout << arr[i][j] << " " << std::endl;
    //Matrix arr{{4,5,6}, {8,17,20}, {12,43,59}};
    Matrix Larr(n,n,1.0);
    Matrix Uarr(n,n,0.0);

    std::cout << "before:" << std::endl;
    std::cout << Larr << std::endl;
    std::cout << Uarr << std::endl;
/* ---------------------------------------------- */
for(int w1=0;w1<n;w1++)
{
    for(int w2=0;w2<n;w2++)
    {
        if(w1 !=w2)
            Larr[w1][w2]=0;
    }
}
/* ---------------------------------------------- */
    std::cout << "Iniitialize:" << std::endl;
    std::cout << Larr << std::endl;
    std::cout << Uarr << std::endl;
for (int k=0;k<n;k++)
{
        for (int i=k+1;i<n;i++)
        {
            Larr[i][k]=arr[i][k]/arr[k][k];
        }

        for(int j=k;j<n;j++)
        {
            Uarr[k][j]=arr[k][j];
        }
        for(int i1=k+1;i1<n;i1++)
        {
            for(int j1=k+1;j1<n;j1++)
            {
                arr[i1][j1]=arr[i1][j1]-Larr[i1][k]*Uarr[k][j1];
            }
        }
}
/* ---------------------------------------------- */
    std::cout << "Result:" << std::endl;
    std::cout << Larr << std::endl;
    std::cout << Uarr << std::endl;

return 0;
}
