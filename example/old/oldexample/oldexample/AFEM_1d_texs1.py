#!/usr/bin/env python3 
#

import sys
import numpy as np
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

from fealpy.mesh import IntervalMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

from fealpy.mesh.adaptive_tools import mark
class Pde():
    def __init__(self, c):
        self.c = c

    def init_mesh(self, n=1):
        node = np.array([0, 1], dtype=np.float)
        cell = np.array([(0, 1)], dtype=np.int)
        mesh = IntervalMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    def solution(self, p):
        return 0

    def gradient(self, p):
        return 0

    def source(self, p):
        return np.exp(-self.c*(p - 0.5)**2)

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

def post_error(pde, uh, space, recover=False):
    mesh = space.mesh
    h = space.cellmeasure
    f = lambda bcs : pde.source(mesh.bc_to_point(bcs))
    eta = h*space.integralalg.L2_norm(f, celltype=True)
    if recover is True:
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')
        b = np.bincount(cell[:, 0], weights=eta, minlength=NN)
        b += np.bincount(cell[:, 1], weights=eta, minlength=NN)
        valence = np.bincount(cell.flat, weights=np.ones(cell.shape).flat, minlength=NN)
        eta = b/valence
    return eta


n = int(sys.argv[1])
p = int(sys.argv[2])
maxit = int(sys.argv[3])
theta = int(sys.argv[4])
pde = Pde(100)
mesh = pde.init_mesh(n=n)


for i in range(maxit):
    print('step:', i)
    space = LagrangeFiniteElementSpace(mesh, p)
    A = space.stiff_matrix()
    b = space.source_vector(pde.source)
    
    bc = DirichletBC(space, pde.dirichlet)
    AD, b= bc.apply(A, b)
    
    uh = space.function()
    uh[:] = spsolve(AD, b)
    
    eta = post_error(pde, uh, space)
    markedCell = mark(eta,theta=theta,method='MAX')
    if i < maxit - 1:
        markedCell = mark(eta,theta=theta,method='MAX')
        mesh.refine(markedCell)

# 可视化数值解和真解
node = mesh.entity('node')
f = pde.source(node)
idx = np.argsort(node)
fig = plt.figure()
plt.plot(node[idx], uh[idx],"r*--",linewidth = 2,label = "$solution-u_h$")
plt.legend()
plt.show()