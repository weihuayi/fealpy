#ifndef Tetrahedron_h
#define Tetrahedron_h

#include "constant.h"

namespace  iMath {

namespace GeometryObject {

template<class Point_3>
class Tetrahedron
{
public:
    typedef Point_3 Point;
public:

    Tetrahedron()
    {
        data[0] = Point(0.0, 0.0, 0.0);
        data[1] = Point(1.0, 0.0, 0.0);
        data[2] = Point(0.0, 1.0, 0.0);
        data[3] = Point(0.0, 0.0, 1.0);
    }

    Point & operator[] (const int i)
    {
        return data[i];
    }

    const Point & operator[] (const int i) const
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
    Point data[4];
};

} // end of namespace GeometryObject

} // end of namespace iMath
