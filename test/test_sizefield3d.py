import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, spdiags, eye, bmat

from fealpy.mesh import UniformMesh3d, MeshFactory, QuadrangleMesh, TriangleMesh
import time

def ff(x):
    y = np.linalg.norm(x - np.array([[-0.1, -0.1]]), axis=-1)
    #return np.sin(np.pi*x[..., 0])*np.sin(np.pi*x[..., 1])
    return np.sin(2*np.pi*x[..., 0])*np.sin(2*np.pi*x[..., 1])+1
    #return y**2

def exu(x):
    return np.sin(np.pi*x[..., 0])*np.sin(np.pi*x[..., 1])

def source(x):
    return 2*np.pi**2*exu(x)

box = [0, 1, 0, 1, 0, 1]
N = int(sys.argv[1])
h = [1/N, 1/N, 1/N]
origin = [0, 0, 0]
extend = [0, N, 0, N, 0, N]

meshb = UniformMesh3d(extend, h, origin) # 背景网格
pnode = meshb.entity('node').reshape(-1, 3)

J = meshb.grad_jump_matrix()
S = meshb.stiff_matrix()
M = meshb.mass_matrix()
G2 = meshb.grad_2_matrix()

F = meshb.source_vector(source)
x = meshb.function().reshape(-1)

isDDof = meshb.boundary_node_flag()
x[isDDof] = exu(pnode[isDDof])

F -= S@x
bdIdx = np.zeros(S.shape[0], dtype=np.int_)
bdIdx[isDDof] = 1
Tbd = spdiags(bdIdx, 0, S.shape[0], S.shape[0])
T = spdiags(1-bdIdx, 0, S.shape[0], S.shape[0])
S = T@S@T + Tbd
F[isDDof] = x[isDDof]
print(np.sum(isDDof))

x = spsolve(S, F)
print(np.max(np.abs(x - exu(pnode))))

ff = meshb.function().reshape(-1)
F = meshb.source_vector(exu)
ff[:] = spsolve(M, F)
#print(np.max(np.abs(ff - exu(pnode))))


