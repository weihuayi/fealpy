#!/usr/bin/env python3
# 

import time
import sys
import argparse 
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TetrahedronMesh, MeshFactory
from fealpy.functionspace import FirstNedelecFiniteElementSpace3d, FirstNedelecFiniteElementSpace2d 
from fealpy.decorator import cartesian, barycentric
# solver
from scipy.sparse.linalg import spsolve, cg

import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@cartesian
def ff(p):
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]

    val = np.zeros(p.shape, dtype=np.float_)
    val[..., 0] = -y
    val[..., 1] = x
    val[..., 2] = np.sin(y)
    return val

node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float_)
#cell = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int_)
node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float_)
cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
mesh = TetrahedronMesh(node, cell) 
#mesh.uniform_refine(1)

space = FirstNedelecFiniteElementSpace3d(mesh)

M = space.mass_matrix()
A = space.curl_matrix()

fh = space.interpolation(ff)
fh = space.project(ff)

# 计算误差
err = space.integralalg.error(ff, fh)
print(err)

#fh[:] = 0
#fh[0] = 1

def test_basis():
    bcs = np.zeros([6, 4], dtype=np.float_)
    le = mesh.ds.localEdge
    bcs[np.arange(6)[:, None], le] = 0.5

    phi = space.basis(bcs)
    glambda = mesh.grad_lambda()
    cell2edge = mesh.ds.cell_to_edge()

    et = mesh.edge_tangent()[cell2edge[0]]

    val = np.einsum("qclg, qg->lq", phi, et)
    n = val.shape[0]
    print(np.sum(np.abs(val-np.identity(n))))

def test_value():
    bcs = np.array([[0.1, 0.2, 0.3, 0.4]])

    uh = space.function()
    uh[2] = 1
    val = uh(bcs)
    print("val = ", val)

def test_face_basis():
    cell2dof = space.dof.cell_to_dof()
    face2dof = space.dof.face_to_dof()
    face = mesh.entity("face")
    cell = mesh.entity("cell")

    print("cell", cell)
    print("face", face)
    print("cell2dof", cell2dof)
    print("face2dof", face2dof)

    bcs = np.array([[0, 0.2, 0.2, 0.6]])
    bcs0 = np.array([[0.2, 0.2, 0.6]])
    p0 = mesh.bc_to_point(bcs)
    p1 = mesh.bc_to_point(bcs0)
    print(p0)
    print(p1)

    phi = space.basis(bcs)
    fphi = space.face_basis(bcs0)
    fn = mesh.face_unit_normal()[-1]
    a = np.cross(np.array([fn]), fphi[0, -1])
    b = np.cross(np.array([fn]), phi[0, 0])
    print("phi", phi.shape)
    print("fphi", fphi.shape)

    print("phi", b)
    print("fphi", a)





#test_basis()
#test_value()
test_face_basis()



