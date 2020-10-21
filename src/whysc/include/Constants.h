#ifndef Constants_h
#define Constants_h

#include<cmath>
#include<limits>
#include<string>

namespace WHYSC 
{
namespace Constants 
{
    constexpr double        c = 2.99792458e8; // 光速
    constexpr double        e = 2.718281828459045; // 
    constexpr double        pi = 3.14159265358979323846; // 圆周率
    constexpr double        inf = std::numeric_limits<double>::infinity();
    const double            nan = std::nan("1");
}
}

#endif // end of Constants_h
