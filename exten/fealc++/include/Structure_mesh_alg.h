#ifndef Structure_mesh_alg_h
#define Structure_mesh_alg_h

#include "moab/Core.hpp"
#include "moab/ScdInterface.hpp"

#include "Geometry/Geometry_kernel.h"

namespace iMath {

namespace MeshAlg {

using namespace moab;

template<class GK = Geometry_kernel<> >
class Structure_mesh_alg
{
public:
    typedef typename GK::Point_2 Point_2;
    typedef typename GK::Point_3 Point_3;
public:

    void execute(Interface * mb, int dim,  
            double min, double max, int I) 
    {
        if(dim == 2)
            execute(mb, 
                    min, max, I, 
                    min, max, I);
        else if(dim == 3)
            execute(mb,
                    min, max, I, 
                    min, max, I,
                    min, max, I);
    }

    void execute(Interface * mb, 
            double x_0, double x_1, int I, 
            double y_0, double y_1, int J)
    {
        double hx = (x_1 - x_0)/double(I);
        double hy = (y_1 - y_0)/double(J);

        ScdInterface * scdiface;
        ErrorCode rval = mb->query_interface(scdiface); 
        int num_nodes = (I+1)*(J+1);
        double coords[3*num_nodes];
        for(int j = 0; j < J+1; j++)
            for(int i = 0; i < I+1; i++)
            {
                int idx = (I+1)*j + i;
                coords[3*idx] = x_0 + i*hx;
                coords[3*idx + 1] = y_0 + j*hy;
                coords[3*idx + 2] = 0;
            }

        ScdBox * box;
        rval = scdiface->construct_box(HomCoord(0, 0, 0), 
                HomCoord(I, J, 0), coords, num_nodes, box);
    }

    void execute(Interface * mb, 
            double x_0, double x_1, int I, 
            double y_0, double y_1, int J,
            double z_0, double z_1, int K)
    {
        double hx = (x_1 - x_0)/double(I);
        double hy = (y_1 - y_0)/double(J);
        double hz = (z_1 - z_0)/double(K);

        ScdInterface * scdiface;
        ErrorCode rval = mb->query_interface(scdiface); 
        int num_nodes = (I+1)*(J+1)*(K+1);

        double coords[3*num_nodes];
        int num0 = (I+1)*(J+1);
        int num1 = I+1;
        for(int k = 0; k < K+1; k++)
            for(int j = 0; j < J+1; j++)
                for(int i = 0; i < I+1; i++)
                {
                    int idx = num0*k + num1*j + i;
                    coords[3*idx] = x_0 + i*hx;
                    coords[3*idx + 1] = y_0 + j*hy;
                    coords[3*idx + 2] = z_0 + k*hz;
                }

        ScdBox * box;
        rval = scdiface->construct_box(HomCoord(0, 0, 0), 
                HomCoord(I, J, K), coords, num_nodes, box);

    }

    
private:
    double _x[2];
    double _y[2];
    double _I;
    double _J;
};

}// end of MeshAlg

} // end of iMath

#endif // end of Structure_mesh_alg_3_h
