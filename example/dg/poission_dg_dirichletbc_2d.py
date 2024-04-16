
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
mesh_tri = TriangleMesh.from_box(box=pde.domain(), nx=16, ny=16)
mesh = PolygonMesh.from_mesh(mesh_tri)
is_bd_face = mesh.ds.boundary_face_flag()
space = ScaledMonomialSpace2d(mesh, p=P)

gamma = P * (P+1)
fm = mesh.entity_measure('face')

bform = BilinearForm(space)
bform.add_domain_integrator(ScalarDiffusionIntegrator(q=Q))
# bform.add_domain_integrator(ScalerInterfaceIntegrator(q=Q))
# bform.add_domain_integrator(ScalerInterfaceMassIntegrator(q=Q, c=gamma/fm)) # penalty
A = bform.assembly()

# lform = LinearForm(space)
# lform.add_domain_integrator(ScalarSourceIntegrator(source=pde.source, q=Q))
# lform.add_boundary_integrator(ScalarDirichletBoundarySourceIntegrator(source=pde.dirichlet, q=Q))
# lform.add_boundary_integrator(ScalerBoundarySourceIntegrator(source=pde.dirichlet, q=Q, c=gamma/fm[is_bd_face]))  # penalty
# F = lform.assembly()

from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
axes.imshow(A.toarray())
plt.show()

# uh = space.function()
# uh[:] = spsolve(A, F)

# err = mesh.error(pde.solution, lambda x, index: uh(x, index))
# gerr = mesh.error(pde.gradient, lambda x, index: uh.grad_value(x, index))

# print(err, gerr)


# from matplotlib import pyplot as plt

# fig = plt.figure()
# axes = fig.add_subplot(1, 1, 1, projection='3d')
# mesh_tri.show_function(axes, uh)

# plt.show()
