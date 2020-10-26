#include <iostream>


int main(int argc, char **argv) 
{

    std::cout << "Size of char : " 
        << sizeof(char) << std::endl;
    std::cout << "Size of int : " 
        << sizeof(int) << std::endl;
    std::cout << "Size of short int : " 
        << sizeof(short int) << std::endl;
    std::cout << "Size of long int : " 
        << sizeof(long int) << std::endl;
    std::cout << "Size of float : " 
        << sizeof(float) << std::endl;
    std::cout << "Size of double : " 
        << sizeof(double) << std::endl;
    std::cout << "Size of wchar_t : " 
        << sizeof(wchar_t) << std::endl;
   
   return 0;
}
