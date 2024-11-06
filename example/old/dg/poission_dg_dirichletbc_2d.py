
import numpy as np

from fealpy.dg import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ScalarDirichletBoundarySourceIntegrator,
    ScalerBoundarySourceIntegrator,
    ScalerInterfaceIntegrator,
    ScalerInterfaceMassIntegrator
)

from fealpy.mesh import TriangleMesh, PolygonMesh
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.pde.poisson_2d import CosCosData
from fealpy.fem import BilinearForm, LinearForm
from scipy.sparse.linalg import spsolve

P = 1
Q = P + 2

pde = CosCosData()
mesh_tri = TriangleMesh.from_box(box=pde.domain(), nx=64, ny=64)
mesh = PolygonMesh.from_mesh(mesh_tri)
is_bd_face = mesh.ds.boundary_face_flag()
space = ScaledMonomialSpace2d(mesh, p=P)

gamma = P * (P+1)
fm = np.sqrt(mesh.entity_measure('face'))

bform = BilinearForm(space)
bform.add_domain_integrator(ScalarDiffusionIntegrator(q=Q))
bform.add_boundary_integrator(ScalerInterfaceIntegrator(q=Q))
bform.add_boundary_integrator(ScalerInterfaceMassIntegrator(q=Q, c=gamma/fm)) # penalty item
A = bform.assembly()

lform = LinearForm(space)
lform.add_domain_integrator(ScalarSourceIntegrator(source=pde.source, q=Q))
lform.add_boundary_integrator(ScalarDirichletBoundarySourceIntegrator(source=pde.dirichlet, q=Q))
lform.add_boundary_integrator(
    ScalerBoundarySourceIntegrator(source=pde.dirichlet, q=Q, c=gamma/fm[is_bd_face])
) # penalty item
F = lform.assembly()

uh = space.function()
uh[:] = spsolve(A, F)

### visualization

from matplotlib import pyplot as plt

err = mesh.error(pde.solution, lambda x, index: uh(x, index))
gerr = mesh.error(pde.gradient, lambda x, index: uh.grad_value(x, index))

print(err, gerr)

from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
cell = mesh_tri.entity('cell')
dof = mesh_tri.entity('node', index=cell).reshape(-1, 2)
dof_to_cell = np.arange(cell.shape[0]).repeat(cell.shape[1])
cax = axes.plot_trisurf(
        dof[:, 0], dof[:, 1],
        uh(dof, index=dof_to_cell), triangles=space.cell_to_dof(), cmap='rainbow', lw=0.1)

plt.show()
