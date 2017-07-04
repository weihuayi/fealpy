#ifndef Box_h
#define Box_h


#include <vector>

namespace iMath {

namespace GeometryObject {

template<class Point>
class Box {

public:
    Box()
    {

    }

    Box(std::vector<Point> & pc)
    {
        int dim = Point::dimension();
        int n = pc.size();
        maxp = pc[0];
        minp = pc[0];
        for(int i = 1; i < n; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                maxp[j] = maxp[j] < pc[i][j] ? pc[i][j] : maxp[j];
                minp[j] = minp[j] > pc[i][j] ? pc[i][j] : minp[j];
            }
            
        }
    }

    Point centroid()
    {
        return minp + 0.5*(maxp - minp);
    }


private:
    Point maxp;
    Point minp;
};

}
}

#endif // end of Box_h
