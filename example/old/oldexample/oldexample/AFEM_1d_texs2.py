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

from fealpy.tools.show import showmultirate, show_error_table

class Pde():
    def __init__(self, c):
        self.c = c

    def init_mesh(self, n=1):
        node = np.array([0.4, 0.6], dtype=np.float)
        cell = np.array([(0, 1)], dtype=np.int)
        mesh = IntervalMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    def solution(self, p):
        return np.exp(-self.c*(p - 0.5)**2)

    def gradient(self, p):
        return -2*self.c*(p-0.5)*np.exp(-self.c*(p - 0.5)**2)

    def source(self, p):
        return -(-2*self.c*(p-0.5))**2*np.exp(-self.c*(p - 0.5)**2) + 2*self.c*np.exp(-self.c*(p - 0.5)**2)

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

errorType = ['$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$']


Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    print('step:', i)
    space = LagrangeFiniteElementSpace(mesh, p)
    A = space.stiff_matrix()
    b = space.source_vector(pde.source)
    
    bc = DirichletBC(space, pde.dirichlet)
    AD, b= bc.apply(A, b)
    
    uh = space.function()
    uh[:] = spsolve(AD, b)
    
    Ndof[i] = space.mesh.number_of_nodes()
    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)
    
    eta = post_error(pde, uh, space, recover=False)
    markedCell = mark(eta,theta=theta,method='MAX')
    if i < maxit - 1:
        markedCell = mark(eta,theta=theta,method='MAX')
        mesh.refine(markedCell)

# 显示误差
show_error_table(Ndof, errorType, errorMatrix)
# 可视化误差收敛阶
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
# 可视化数值解和真解
node = mesh.entity('node')
u = pde.solution(node)
idx = np.argsort(node)
fig = plt.figure()
plt.plot(node[idx], uh[idx],"r*--",linewidth = 2,label = "$u_h$")
plt.plot(node[idx], u[idx],'k-',linewidth = 2,label = "$u$")
plt.legend()
plt.show()

