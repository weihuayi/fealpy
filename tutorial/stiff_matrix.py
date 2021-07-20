
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace

p = 1
q = 3

box = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box, nx=1, ny=1, meshtype='tri')
NN = mesh.number_of_nodes()
space = LagrangeFiniteElementSpace(mesh, p=p)

gdof = space.number_of_global_dofs()

qf = mesh.integrator(q, 'cell')
bcs, ws = qf.get_quadrature_points_and_weights() 

cellmeasure = mesh.entity_measure('cell') # 
gphi = space.grad_basis(bcs) # (NQ, NC, ldof, 2) 

# (NC, ldof, ldof)
A = np.einsum('q, qcid, qcjd, c->cij', ws, gphi, gphi, cellmeasure)

cell2dof = space.cell_to_dof() # (NC, ldof)
print('cell2dof:\n', cell2dof)

# (NC, ldof) --> (NC, ldof, 1) --> (NC, ldof, ldof)
I = np.broadcast_to(cell2dof[:, :, None], shape=A.shape)
print('I:\n', I)

# (NC, ldof) --> (NC, 1, ldof) --> (NC, ldof, ldof)
J = np.broadcast_to(cell2dof[:, None, :], shape=A.shape)
print('J:\n', J)

A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))

phi = space.basis(bcs) # (NQ, NC, ldof)
# (NC, ldof, ldof)
M = np.einsum('q, qci, qcj, c->cij', ws, phi, phi, cellmeasure)
M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))

def f(p):
    pi = np.pi
    x = p[..., 0]
    y = p[..., 1]
    return np.sin(pi*p[..., 0])*np.sin(pi*p[..., 1])

ps = mesh.bc_to_point(bcs) # (NQ, NC, GD)
val = f(ps) # (NQ, NC)
# (NC, ldof)
bb = np.einsum('q, qc, qci, c->ci', ws, val, phi, cellmeasure)
print('bb:\n', bb)

b = np.zeros(gdof, dtype=np.float64)
np.add.at(b, cell2dof, bb)
print('b:\n', b)




ipoints = space.interpolation_points()
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ipoints, showindex=True, color='r', fontsize=24)
plt.show()


print('bcs:\n', bcs)
print('ws:\n', ws)
