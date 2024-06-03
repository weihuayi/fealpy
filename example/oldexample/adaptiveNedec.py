#!/usr/bin/env python3
# 
import sys
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import math

from fealpy.mesh import MeshFactory
from fealpy.pde.timeharmonic_2d import CosSinData
from fealpy.pde.timeharmonic_2d import LShapeRSinData
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d 
from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table



p = int(sys.argv[1]) # p=0 denotes the linear Nedelec element
n = int(sys.argv[2]) # n=1 denotes the n*n mesh
maxit = int(sys.argv[3]) # the number of calculating

#pde = CosSinData() #zhenjie 
pde = LShapeRSinData()
mf = MeshFactory()
mesh = mf.boxmesh2d([-1, 1, -1, 1], nx=n, ny=n, meshtype='tri', 
threshold=lambda p: (p[..., 0] > 0.0) & (p[..., 1] < 0.0))
#mesh = pde.init_mesh(n=n, meshtype='tri')  # mesh

mesh.add_plot(plt)
plt.savefig('./test-1.png')
plt.close()

cell2edge = mesh.ds.cell_to_edge() # (NC, 3) 
######################
#############################################
##############################################
#print("celledge:",cell2edge[0])
##############################################

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla\\times u - \\nabla\\times u_h||_{\Omega, 0}$',
             '$|| u - u_I||_{\Omega,0}$',
             '$||\\nabla\\times u - \\nabla\\times u_I||_{\Omega, 0}$',
             ]
errorMatrix = np.zeros((4, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float) 

lNDof = np.zeros(maxit, dtype=np.float) 
#the array to store "number of dof" for every matrit[i] 

for i in range(maxit):
    space = FirstKindNedelecFiniteElementSpace2d(mesh, p=p) #Nedelec space

    gdof = space.number_of_global_dofs() 
    
    NDof[i] = gdof 

    uh = space.function()
   
    uI = space.interpolation(pde.solution)
    

    A = space.curl_matrix() - space.mass_matrix()  # original stiffness matrix
    F = space.source_vector(pde.source) #original right-hand
    
    isDDof = space.set_dirichlet_bc(uh, pde.solution)
    isDDof = np.tile(isDDof, 2)
    F = F.T.flat
    x = uh.T.flat # 把 uh 按列展平
    F -= A@x
    F[isBdDof] = x[isBdDof] 
    
    isBdDof = space.boundary_dof() # boundary Dof   
    bdIdx = np.zeros(gdof, dtype=np.int)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)   
    A = T@A@T + Tbd  # the stiffness matrix after dealing boundary condition
    
    
       
    
    uh[:] = spsolve(A, F) # solution
        
    # u-uh l2error
    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh) 
    print("u-uh = ", errorMatrix[0, i])
    # \nabla\times(u-uh) l2error
    errorMatrix[1, i] = space.integralalg.L2_error(pde.curl, uh.curl_value)
    
    errorMatrix[2, i] = space.integralalg.L2_error(pde.solution, uI) 
    print("u-uI = ", errorMatrix[2, i])
    errorMatrix[3, i] = space.integralalg.L2_error(pde.curl, uI.curl_value) 
    print("\\nabla \\times (u-uI) = ", errorMatrix[3, i])
    if i < maxit-1:
        mesh.uniform_refine()
     
#show_error_table(NDof, errorType, errorMatrix)
# parameter "propsize"  can adjust the size of picture
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=10)
plt.show()
