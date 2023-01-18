
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace

domain = [0, 1, 0, 1]

mesh = MF.boxmesh2d(domain, nx=5, ny=5, meshtype='tri')
cellmeasure = mesh.entity_measure('cell')

space = LagrangeFiniteElementSpace(mesh, p=1)

qf = mesh.integrator(3, 'cell')
bcs, ws = qf.get_quadrature_points_and_weights()
print('bcs:\n', bcs) # (NQ, 3)
print('ws:\n', ws) # (NQ, )

phi = space.basis(bcs) # (NQ, 1, 3) --> (NQ, NC, 3)

H = np.einsum('q, qci, qcj, c -> cij', ws, phi, phi, cellmeasure)

print(H)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()

