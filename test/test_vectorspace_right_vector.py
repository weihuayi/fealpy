import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace


def f1(p):
    x = p[..., 0]
    y = p[..., 1]
    val = 3*np.exp(x**2 + y**2)
    return val

def f2(m):
    x = m[..., 0]
    y = m[..., 1]
    val = np.zeros(m.shape, dtype=np.float)
    val[..., 0] = 3*np.exp(x**2 + y**2)
    val[..., 1] = 3*np.exp(x**2 + y**2)
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

p = 0
mesh = TriangleMesh(node, cell)
mesh.uniform_refine()
node = mesh.entity('node')
cell = mesh.entity('cell')
NC = mesh.number_of_cells()
bc = mesh.entity_barycenter('cell')
if p == 0:
    integrator = mesh.integrator(1)
else:
    integrator = mesh.integrator(p+2)

cellmeasure = mesh.entity_measure('cell')

V = VectorLagrangeFiniteElementSpace(mesh, p, spacetype='D')
b = V.source_vector(f2,integrator, cellmeasure)
c = V.scalarspace.source_vector(f1, integrator, cellmeasure);

print(b)
print(c)
print(V.number_of_global_dofs())

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()
