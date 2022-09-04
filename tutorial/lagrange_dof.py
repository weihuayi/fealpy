
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace


def f(p):
    x = p[..., 0]
    y = p[..., 1]
    return 1+x**2+y**2

p = 3
box = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box, nx=1, ny=1, meshtype='tri')
space = LagrangeFiniteElementSpace(mesh, p=p)

gdof = space.number_of_global_dofs()
ldof = space.number_of_local_dofs()

cell2dof = space.cell_to_dof() # (NC, ldof) cell2dof[i, j] 

qf = mesh.integrator(p+3)
bcs, ws = qf.get_quadrature_points_and_weights()

gphi = space.grad_basis(bcs) # (NQ, NC, ldof, GD)

cm = mesh.entity_measure('cell')

S = np.einsum('q, qcid, qcjd, c-> cij', ws, gphi, gphi, cm)

I = np.broadcast_to(cell2dof[:, :, None], shape=S.shape)
J = np.broadcast_to(cell2dof[:, None, :], shape=S.shape)

S = csr_matrix((S.flat, (I.flat, J.flat)), shape=(gdof, gdof))


phi = space.basis(bcs) # (NQ, 1, ldof)

M = np.einsum('q, qci, qcj, c->cij', ws, phi, phi, cm)

M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))


ps = mesh.bc_to_point(bcs)
val = f(ps)

bb = np.einsum('q, qc, qci, c->ci', ws, val, phi, cm) # (NC, ldof)

# cell2dof.shape == (NC, ldof)
b = np.zeros(gdof, dtype=np.float64)
np.add.at(b, cell2dof, bb)








ipoints = space.interpolation_points() # (gdof, 2)

print('cell2dof:')
for i, val in enumerate(cell2dof):
    print(i, ": ", val)


edge = mesh.entity('edge')
print('edge:')
for i, val in enumerate(edge):
    print(i, ": ", val)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=24)
mesh.find_edge(axes, showindex=True, fontsize=22)
mesh.find_cell(axes, showindex=True, fontsize=20)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ipoints, showindex=True, color='r', fontsize=24)
plt.show()
