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
    return (2*pi**2+1)*np.sin(pi*x)*np.sin(pi*y)

box = [0, 2*np.pi, 0, 2*np.pi]
N = 4
mesh = StructureMeshND(box, N)
node = mesh.node

F0 = f(node)
F1 = np.fft.fft2(F0)
F2 = np.fft.ifft2(F1).real

print('F0:\n', F0)
print('F2:\n', F2)

u = u(node)
k2 = mesh.ps_coefficients_square()
U = F/(k2+1)
U = np.fft.ifft2(U).real
print('u:\n', u)
print('U:\n', U)
print('k2:\n', k2+1)




qmesh = StructureQuadMesh(box, N, N)
mi = qmesh.multi_index()
fig = plt.figure()
axes = fig.gca()
qmesh.add_plot(axes)
qmesh.find_node(axes, showindex=True)
plt.show()
