import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace

def source_vector(m):
    x = m[..., 0]
    y = m[..., 1]
    val = np.zeros(m.shape, dtype=np.float)
    val[..., 0] = 3*x**2
    val[..., 1] = 3*y**2
    return val

node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)
    ], dtype=np.float)

cell = np.array([
    (1, 2, 0),
    (3, 0, 2)], dtype=np.int)

p = 1
print('p',p)
mesh = TriangleMesh(node, cell)
mesh.uniform_refine()
node = mesh.node
cell = mesh.ds.cell
NC = mesh.number_of_cells()
bc = mesh.entity_barycenter('cell')
print('bc',bc)
if p == 0:
    integrator = mesh.integrator(1)
else:
    integrator = mesh.integrator(p+2)
    bcs = integrator.quadpts
    ws = integrator.weights
    pxy = np.zeros((6,NC,2), dtype=np.float)
    print('node',node[cell[:, 1]])
    for i in range(6):
        pxy[i,:,:] = bcs[i,0]*node[cell[:,0]] + bcs[i,1]*node[cell[:,1]]\
                + bcs[i,2]*node[cell[:,2]]
        print('pxy', pxy[i,:,:])
print('integrator', integrator)
cellmeasure = mesh.entity_measure('cell')

V = VectorLagrangeFiniteElementSpace(mesh, p, spacetype='D')
b = V.source_vector(source_vector,integrator, cellmeasure)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
print(b)
plt.show()
