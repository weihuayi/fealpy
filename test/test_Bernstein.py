import numpy as np
import sys
import time
import pyamg
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from fealpy.mesh import MeshFactory, HalfEdgeMesh2d, PolygonMesh
from fealpy.functionspace import BernsteinFiniteElementSpace
from scipy.sparse.linalg import cg, inv, dsolve
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.pde.poisson_2d import CosCosData


maxit = 5
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

pde = CosCosData()
p = 5
mesh = pde.init_mesh(n=1)
for i in range(maxit):
    print("The {}-th computation:".format(i))

    ## 准备空间
    space = BernsteinFiniteElementSpace(mesh, p=p)
    NDof[i] = space.dof.number_of_global_dofs() 

    s = time.time()
    S = space.stiff_matrix()
    e = time.time()
    print("time = ", e-s)
    ## 右端项
    b = space.source_vector(pde.source)

    ## 边界条件处理
    uh = space.function()
    bc = DirichletBC(space, pde.dirichlet)
    S, b = bc.apply(S, b, uh)

    ## 求解
    uh[:] = spsolve(S, b)

    err = space.integralalg.L2_error(pde.solution, uh)
    err1 = space.integralalg.error(pde.gradient, uh.grad_value)

    # 计算误差
    errorMatrix[0, i] = err 
    errorMatrix[1, i] = err1
    if i < maxit-1:
        mesh.uniform_refine()

showmultirate(plt, 2, NDof, errorMatrix,  errorType, propsize=20)
show_error_table(NDof, errorType, errorMatrix)
plt.show()

