import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Tritree import Tritree

def solution(p):
    x = p[:, 0]
    y = p[:, 1]
    val = np.exp(x**2 + y**2)
    return val

node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)], dtype=np.float)

cell = np.array([
        [1, 2, 0],
        [3, 0, 2]], dtype=np.int)

mesh = TriangleMesh(node, cell)
mesh.uniform_refine(1)
node = solution(node)
print(node)
cell = mesh.entity('cell')
Dlambda = mesh.grad_lambda()
grad = np.einsum('ijm, ijm->im', node[cell], Dlambda)
eta = np.sqrt(np.sum(grad**2, axis=1)*mesh.cell_area())
print(eta)


tmesh = Tritree(node, cell)



fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
tmesh.find_node(axes, showindex=True)
#tmesh.find_edge(axes, showindex=True)
#tmesh.find_cell(axes, showindex=True)
plt.show()
