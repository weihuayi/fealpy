#include <iostream>

#include <cmath>


int main()
{
    double a[4] = {0, 1, 2, 3};

    std::cout << "vector a:(";
    for(int i=0; i < 4; i++)
    {
        std::cout << a[i] << " ";
    }
    std::cout << ")" << std::endl;

    double sum = 0.0;
    for(int i=0; i < 4; i++)
    {
        sum += std::abs(a[i]);
    }

    std::cout << "The norm is " << 
        sum << std::endl;

    sum = 0.0;
    for(int i=0; i < 4; i++)
    {
        sum += a[i]*a[i];
    }
    sum = std::sqrt(sum);
    
    std::cout << "The norm is " << 
        sum << std::endl;

    double B[3][3] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    double c[3] = {1, 1, 1};
    double d[3] = {0, 0, 0};

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
        {
            d[i] += B[i][j]*c[j];
        }

    std::cout << "d: (" << d[0] << " "
        << d[1] << " " << d[2] << ")" << std::endl;

    return 0;

}
