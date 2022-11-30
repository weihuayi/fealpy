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
N = 10
h = [1/N, 1/N, 1/N]
origin = [0, 0, 0]
extend = [0, N+1, 0, N+1, 0, N+1]

meshb = UniformMesh3d(extend, h, origin) # 背景网格
J = meshb.grad_jump_matrix()
