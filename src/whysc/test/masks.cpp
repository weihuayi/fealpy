#include <iostream>
#include <bitset>

int main(int argc, char **argv)
{
    unsigned int test_masks[32];
    unsigned int clear_masks[32];
    unsigned int extract_masks[32];
    for (int i = 0; i < 32; i++) {
        test_masks[i] = (1 << i); 
        clear_masks[i] = ~(1 << i);
        extract_masks[i] = (1 << i) - 1;

        std::cout << i << ": ";
        std::cout << test_masks[i] << " " 
            << clear_masks[i] << " "
            << extract_masks[i] << std::endl;
    }

    char flags = 0;
    flags |= 1; // 00000001
    flags |= 2; // 00000011
    flags |= 4; // 00000111

    flags &= clear_masks[0]; // 11111111111111111111111111111110
    std::bitset<8> x(flags);
    std::cout <<  x << std::endl;

    bool b0 = flags & 1;
    flags |= 1; // 00000001
    b0 = flags & 1; 

    flags |= 2;
    return 0;
}

// 00000000000000000000000000000001 = 1 
// 00000000000000000000000000000010 = 2 
// 00000000000000000000000000000100 = 4 
// 10000000000000000000000000000000 = 2^31
