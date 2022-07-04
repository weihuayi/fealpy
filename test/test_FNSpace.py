#!/usr/bin/env python3
# 

import time
import sys
import argparse 
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TetrahedronMesh
from fealpy.functionspace import FirstNedelecFiniteElementSpace3d 
from fealpy.decorator import cartesian, barycentric
# solver
from scipy.sparse.linalg import spsolve, cg

import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ff(p):
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]

    val = np.zeros(p.shape, dtype=np.float_)
    val[..., 0] = 0
    val[..., 1] = 0
    val[..., 2] = 3*z
    return val

node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float_)
#cell = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int_)
node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float_)
cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
mesh = TetrahedronMesh(node, cell) 
mesh.uniform_refine(2)

space = FirstNedelecFiniteElementSpace3d(mesh)

M = space.mass_matrix()
A = space.curl_matrix()

fh = space.interpolation(ff)
#fh = space.project(ff)

#fh[:] = 0
#fh[0] = 1

print(mesh.entity("edge"))
print(mesh.grad_lambda())
print(mesh.ds.localEdge)

bcs = np.array([[0.5, 0.5, 0.0, 0.0]])
phi = space.basis(bcs)
glambda = mesh.grad_lambda()
cell2edge = mesh.ds.cell_to_edge()

et = mesh.edge_tangent()[cell2edge[0]]

val = np.einsum("ad, vd->av", phi[0, 0], et)
print("val = ", val)





print("fh = ", fh(bcs))

print(phi)
print(fh[:])

# 计算误差
err = space.integralalg.error(ff, fh)
print(err)


