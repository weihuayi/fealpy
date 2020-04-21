#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import StructureMeshND
from fealpy.mesh import StructureQuadMesh

def u(p):
    x = p[0]
    return np.sin(x) 

def f(p):
    x = p[0]
    return 2*np.sin(x) 

box = [0, 2*np.pi]
N = 10 
mesh = StructureMeshND(box, N)
node = mesh.node

xi = np.fft.fftfreq(N)*N
print('xi:\n', xi)

F0 = f(node)
print('F0:', F0)

F1 = np.fft.fft(F0)
print('F1:', F1)


U1 = F1/(1+xi**2)
print('U1:', U1)
U1 = np.fft.ifft(U1).real
print('U1:', U1)

U0 = u(node)
print('U0:', U0)

error = np.sqrt(np.sum((U0 - U1)**2)/N)
print(error)



if False:
    k = mesh.ps_coefficients()
    print('k:', k)

    k2 = mesh.ps_coefficients_square()
    print('k2:', k2)
    qmesh = StructureQuadMesh(box, N, N)
    mi = qmesh.multi_index()
    fig = plt.figure()
    axes = fig.gca()
    qmesh.add_plot(axes)
    qmesh.find_node(axes, showindex=True)
    plt.show()
