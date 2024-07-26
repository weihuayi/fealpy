from fealpy.np.functionspace import LagrangeFESpace as Space
from fealpy.np.mesh.triangle_mesh import TriangleMesh
from fealpy.np.mesh.uniform_mesh_2d import UniformMesh2D

from fealpy.np.fem import (
    LinearElasticityPlaneStrainOperatorIntegrator, BilinearForm, LinearForm
    )

from fealpy.utils import timer

tmr = timer()

mesh_tri = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=2, ny=2)
NN_tri = mesh_tri.number_of_nodes()
print("NN_tri:", NN_tri)
NC_tri = mesh_tri.number_of_cells()
print("NC_tri:", NC_tri)
mesh_uni2d = UniformMesh2D(extent=(0, 2, 0, 2), h=(1, 1), origin=(0, 0))
NN_uni2d = mesh_uni2d.number_of_nodes()
print("NN_uni2d:", NN_uni2d)
NC_uni2d = mesh_uni2d.number_of_cells()
print("NC_uni2d:", NC_uni2d)
next(tmr)

space_u = Space(mesh, p=1, ctype='C')
space_rho = Space(mesh, p=1, ctype='D')
gdof_u = space_u.number_of_global_dofs()
ldof_u = space_u.number_of_local_dofs()
print("gdof_u:", gdof_u)
print("ldof_u:", ldof_u)
gdof_rho = space_rho.number_of_global_dofs()
gdof_rho = space_rho.number_of_local_dofs() 
print("gdof_rho:", gdof_rho)
print("ldof_rho:", gdof_rho)
print("space_rho:", space_rho)
tmr.send('mesh_and_vspace')