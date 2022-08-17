
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


from fealpy.mesh import MeshFactory as MF


def f(p):
    x = p[..., 0] # x.shape == (NQ, NC)
    y = p[..., 1] # y.shape == (NQ, NC)
    return np.exp(x**2 + y**2) # (NQ, NC)


domain = [0, 1, 0, 1]

mesh = MF.boxmesh2d(domain, nx=5, ny=5, meshtype='tri')

node = mesh.entity('node')
cell = mesh.entity('cell')
NN = mesh.number_of_nodes()


qf = mesh.integrator(3)
bcs, ws = qf.get_quadrature_points_and_weights()
ps = mesh.bc_to_point(bcs) # (NQ, NC, GD)

cm = mesh.entity_measure('cell')
# fval.shape == (NQ, NC)
fval = f(ps)
# (NQ, 3)
bb = np.einsum('q, qc, qi, c->ci', ws, fval, bcs, cm, optimize=True)

# \int_\Omega f \phi_i d x
b = np.zeros(NN, dtype=np.float64)

# cell.shape == (NC, 3)
# bb.shape == (NC, 3)
np.add.at(b, cell, bb)

print(b)

# gphi.shape == (NC, 3, 2)
gphi = mesh.grad_lambda() 

# S.shape == (NC, 3, 3)
S = np.einsum('cid, cjd, c->cij', gphi, gphi, cm)

# (NC, 3) --> (NC, 3, 1)
I = np.broadcast_to(cell[:, :, None], shape=S.shape) # (NC, 3, 3)
# (NC, 3) --> (NC, 1, 3)
J = np.broadcast_to(cell[:, None, :], shape=S.shape)# (NC, 3, 3)


A = csr_matrix((S.flat, (I.flat, J.flat)), shape=(NN, NN), dtype=np.float64)


uh = np.zeros(NN, dtype=np.float64)
uh[20] = 1.0 # \phi_20
uh[15] = 1.0 # \phi_15

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=30)
mesh.find_cell(axes, showindex=True, fontsize=35)

fig = plt.figure()
axes= fig.add_subplot(projection='3d')
axes.plot_trisurf(node[:, 0], node[:, 1], cell, uh, cmap='rainbow', lw=3, edgecolors='k')
plt.show()
