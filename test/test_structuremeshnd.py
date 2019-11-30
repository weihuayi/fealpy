import numpy as np
from numpy.fft import fftn, ifftn

from fealpy.mesh import StructureMeshND

def solution(x):
    return np.sin(x[0])*np.exp(-x[0])

def source(x):
    return 2*np.exp(-x[0])*np.cos(x[0])


box = [0, 2*np.pi]

mesh = StructureMeshND(box, 4)

x = mesh.node
f = source(x)
k = mesh.ps_coefficients_square()

F = fftn(f)
print(F)
print(ifftn(F).real)
print(f)



