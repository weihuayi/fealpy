import matplotlib.pyplot as plt
import numpy as np
import sys

from fealpy.mesh.meshio import load_mat_mesh, write_mat_mesh, write_mat_linear_system
from fealpy.mesh.simple_mesh_generator import fishbone

from fealpy.functionspace.function import FiniteElementFunction
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace  
from fealpy.functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace  

from fealpy.femmodel.BiharmonicFEMModel1 import BiharmonicRecoveryFEMModel
from fealpy.boundarycondition.BoundaryCondition import DirichletBC
from fealpy.solver import solve

from fealpy.model.BiharmonicModel2d import SinSinData
from fealpy.tools.show import showmultirate

m = int(sys.argv[1]) 
meshtype = int(sys.argv[2])
n = int(sys.argv[3])
rtype = int(sys.argv[4])

if rtype == 1:
    rtype='simple'
elif rtype == 2:
    rtype='inv_area'

print('rtype:', rtype)


if m == 1:
    pde = SinSinData()
    box = [0, 1, 0, 1]

maxit = 4

Ndof = np.zeros((maxit,), dtype=np.int)

errorType = ['$\| u - u_h\|$',
             '$\|\\nabla u - \\nabla u_h\|$',
             '$\|\\nabla u_h - G(\\nabla u_h) \|$',
             '$\|\\nabla u - G(\\nabla u_h)\|$',
             '$\|\Delta u - \\nabla\cdot G(\\nabla u_h)\|$',
             '$\|\Delta u -  G(\\nabla\cdot G(\\nabla u_h))\|$',
             '$\|G(\\nabla\cdot G(\\nabla u_h)) - \\nabla\cdot G(\\nabla u_h)\|$'
         ]
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)


for i in range(maxit):
    if meshtype == 1: #fishbone mesh
        mesh = fishbone(box,n)   
    
    integrator = mesh.integrator(3)
    mesh.add_plot(plt,cellcolor='w')

    V = LagrangeFiniteElementSpace(mesh,p=1)
    V2 = VectorLagrangeFiniteElementSpace(mesh,p=1)
    uh = FiniteElementFunction(V)
    rgh = FiniteElementFunction(V2)
    rlh = FiniteElementFunction(V)
    
    fem = BiharmonicRecoveryFEMModel(mesh, pde,integrator,rtype=rtype)
    bc = DirichletBC(V, pde.dirichlet)
    solve(fem, uh, dirichlet=bc, solver='direct')
    
    fem.recover_grad() #TODO
    fem.recover_laplace()
    
    eta1 = fem.grad_recover_estimate()
    eta2 = fem.laplace_recover_estimate()

    Ndof[i] = V.number_of_global_dofs() 
    errorMatrix[0, i] = fem.error.L2_error(pde.solution, uh)
    errorMatrix[1, i] = fem.error.L2_error(pde.gradient, uh)
    errorMatrix[2, i] = np.sqrt(np.sum(eta1**2))
    errorMatrix[3, i] = fem.error.L2_error(pde.gradient, rgh)
    errorMatrix[4, i] = fem.error.L2_error(pde.laplace, rgh)
    errorMatrix[5, i] = fem.error.L2_error(pde.laplace, rlh)
    errorMatrix[6, i] = np.sqrt(np.sum(eta2**2)) 


showmultirate(plt, 1, Ndof, errorMatrix,errorType)
plt.show()
