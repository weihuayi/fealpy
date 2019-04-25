import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

from fealpy.mesh import StructureIntervalMesh
from fealpy.pde.poisson_1d import CosData

I = np.array([0, 1], dtype=np.float)
h = 0.05
mesh = StructureIntervalMesh(I, h)
node = mesh.entity('node')

pde = CosData()

A = -mesh.laplace_operator()
b = pde.source(node)
print("A:", A.toarray())
print("b:", b)

isBdNode = mesh.ds.boundary_node_flag()

NN = mesh.number_of_nodes() 
x = np.zeros((NN,), dtype=np.float)
x[isBdNode] = pde.dirichlet(node[isBdNode]) 

print("x:", x)

b -= A@x

print("b:", b)

bdIdx = np.zeros(NN, dtype=np.int)
bdIdx[isBdNode] = 1
Tbd = spdiags(bdIdx, 0, NN, NN)
T = spdiags(1-bdIdx, 0, NN, NN)
A = T@A@T + Tbd

b[isBdNode] = x[isBdNode] 
print("A:", A.toarray())
print("b:", b)

x = spsolve(A, b)

print(np.max(np.abs(x - pde.solution(node))))



fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
plt.show()
