
#include "Geometry/Geometry_kernel.h"
#include "Mesh_generation_alg.h"
#include <string>
#include <stdlib.h>

using namespace moab;
using namespace std;

typedef iMath::Geometry_kernel<>  GK;
typedef GK::Point_2 Point_2;
typedef GK::Level_set_circle Circle;

int main()
{
    Circle  circle(0.0, 0.0, 0.5);

    Interface * mb = new (std::nothrow) Core;// structure mesh interface
    
 
    int I = 10;
    int J = 10;
    iMath::MeshAlg::Structure_mesh_alg<GK> sm_alg;
    sm_alg.execute(mb, 2, -1.0, 1.0, 10);

    iMath::MeshAlg::Interface_fitted_mesh_alg_2<GK> ifm_alg;
    ifm_alg.execute(mb, circle);

    string file_name = "test.h5m";
    mb->write_file(file_name.c_str());

    string mbc = "mbconvert -f h5m " + file_name + " test.vtk";
    system(mbc.c_str());

    string paraview = "paraview test.vtk";
    system(paraview.c_str());

    delete mb;

    return 0;
}
