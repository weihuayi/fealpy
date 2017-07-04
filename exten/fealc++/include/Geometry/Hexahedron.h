#ifndef Hexahedron_h
#define Hexahedron_h

#include "constant.h"

namespace  iMath {

namespace GeometryObject {

template<class Point_3>
class Hexahedron 
{

public:
    typedef Point_3 Point;

public:

    Hexahedron()
    {
        data[0] = Point(0.0, 0.0, 0.0);
        data[1] = Point(1.0, 0.0, 0.0);
        data[2] = Point(1.0, 1.0, 0.0);
        data[3] = Point(0.0, 1.0, 0.0);
        data[4] = Point(0.0, 0.0, 1.0);
        data[5] = Point(1.0, 0.0, 1.0);
        data[6] = Point(1.0, 1.0, 1.0);
        data[7] = Point(0.0, 1.0, 1.0);
    }

    Hexahedron(double h)
    {
        data[0] = Point(0.0, 0.0, 0.0);
        data[1] = Point(h, 0.0, 0.0);
        data[2] = Point(h,  h,  0.0);
        data[3] = Point(0.0, h, 0.0);
        data[4] = Point(0.0, 0.0, h);
        data[5] = Point(h, 0.0, h);
        data[6] = Point(h, h, h);
        data[7] = Point(0.0, h, h);
    }

    Point & operator [] (const int i) 
    {
        return data[i];
    }

    const Point & operator [] (const int i) const
    {
        return data[i];
    }

    Point & vertex(int i)
    {
        return data[i];
    }

    Point & point(int i)
    {
        return data[i];
    }

private:
    Point data[8];
};

} // end of namespace GeometryObject

} // end of namespace iMath

#endif
