#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import StructureMeshND
from fealpy.mesh import StructureQuadMesh

def u(p):
    x = p[0]
    y = p[1]
    pi = np.pi
    return np.sin(pi*x)*np.sin(pi*y)

def f(p):
    x = p[0]
    y = p[1]
    pi = np.pi
    return 2*pi**2*np.sin(pi*x)*np.sin(pi*y)

box = [0, 2*np.pi, 0, 2*np.pi]
N = 4
mesh = StructureMeshND(box, N)
node = mesh.node
k = mesh.ps_coefficients(flat=True)
print('k:\n', k)
print('node:\n', node)

qmesh = StructureQuadMesh(box, N, N)
mi = qmesh.multi_index()

fig = plt.figure()
axes = fig.gca()
qmesh.add_plot(axes)
qmesh.find_node(axes, showindex=True, multiindex=k)
plt.show()
