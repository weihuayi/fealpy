#include "Geometry/Geometry_kernel.h"
#include "Mesh_generation_alg.h"
#include <string>
#include <stdlib.h>

using namespace moab;
using namespace std;

typedef iMath::Geometry_kernel<>  GK;
typedef GK::Point_2 Point_2;
typedef GK::Level_set_sphere Sphere;

int main()
{
    Sphere sphere();

    Interface * mb = new (std::nothrow) Core;// structure mesh interface
    
 
    int n = 10;
    iMath::MeshAlg::Structure_mesh_alg<GK> sm_alg;
    sm_alg.execute(mb, 3, -1.2, 1.2, n);

    iMath::MeshAlg::Interface_fitted_mesh_alg_3<GK> ifm_alg;
    ifm_alg.execute(mb, sphere);

    string file_name = "test.h5m";
    mb->write_file(file_name.c_str());

    string mbc = "mbconvert -f h5m " + file_name + " test.vtk";
    system(mbc.c_str());

    string paraview = "paraview test.vtk";
    system(paraview.c_str());

    delete mb;

    return 0;
}
