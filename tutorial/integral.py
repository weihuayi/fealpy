
import numpy as np
import matplotlib.pyplot as plt


from fealpy.mesh import MeshFactory as MF


def f(p):
    x = p[..., 0] # x.shape == (NQ, NC)
    y = p[..., 1] # y.shape == (NQ, NC)
    return np.exp(x**2 + y**2) # (NQ, NC)




domain = [0, 1, 0, 1]

mesh = MF.boxmesh2d(domain, nx=2, ny=2, meshtype='tri')

qf = mesh.integrator(3)
bcs, ws = qf.get_quadrature_points_and_weights()

node = mesh.entity('node')
cell = mesh.entity('cell')
cm = mesh.entity_measure('cell')

# bcs.shape == (NQ, 3)
# node[cell].shape == (NC, 3, 2)
# ps.shape == (NQ, NC, 2)
ps = np.einsum('qj, cjd->qcd', bcs, node[cell])

val = f(ps) # val.shape == (NQ, NC)

# \int_\Omega f d x 
I = np.einsum('q, qc, c->c', ws, val, cm)  # I.shape == (NC, )

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ps.reshape(-1, 2))
plt.show()
