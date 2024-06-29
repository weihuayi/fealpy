from fealpy.np.functionspace import LagrangeFESpace as Space
from fealpy.np.mesh.triangle_mesh import TriangleMesh
import numpy as np

mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=2, ny=2)
space = Space(mesh, p=1, ctype='C')
uh = space.function(dim=2)
print("uh:", uh.shape, "\n", uh)

bcs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
test = uh(bcs)
print("test:", test)


gdof = vspace[0].number_of_global_dofs()
vgdof = gdof * GD
ldof = vspace[0].number_of_local_dofs()
vldof = ldof * GD
cell2dof = space.cell_to_dof()
print("gdof", gdof)
print("vldof", ldof)
print("cell2dof:", cell2dof)