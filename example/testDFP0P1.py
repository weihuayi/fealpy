import numpy as np
import matplotlib.pyplot as plt
#from fealpy.mg.DarcyForchheimerP0P1 import DarcyForchheimerP0P1
from fealpy.mg.DFP0P1 import DFP0P1
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.pde.darcy_forchheimer_2d import DarcyForchheimerdata1

np.set_printoptions(threshold=np.inf)

box = [0,1,0,1]
mu = 1
rho = 1
beta = 10
alpha = 1/beta
tol = 1e-6
level = 5
mg_maxN = 3
maxN = 2000
p = 1
n = 1

pde = DarcyForchheimerdata1(box,mu,rho,beta,alpha,level,tol,maxN,mg_maxN)
mesh = pde.init_mesh(1)
node = mesh.node
integrator1 = mesh.integrator(p+2)
integrator0 = mesh.integrator(p)

errorType = ['$|| u - u_h||_0$','$|| p - p_h||$', '$||\\nabla p - \\nabla p_h||_0$']

maxit = 2
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
test_errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
Ndof = np.zeros(maxit,dtype = np.int)


for i in range(maxit):
    test_fem = DFP0P1(pde, mesh, integrator0, integrator1)
    test_fem.solve()
    
    NC = mesh.number_of_cells()
    NN = mesh.number_of_edges()
    Ndof[i] = 2*NC+NN

    test_errorMatrix[0, i] = test_fem.get_uL2_error()
    test_errorMatrix[1, i] = test_fem.get_pL2_error()
    test_errorMatrix[2, i] = test_fem.get_H1_error()



    if i < maxit -1:
        mesh.uniform_refine()

#show_error_table(Ndof, errorType, errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
showmultirate(plt, 0, Ndof, test_errorMatrix, errorType)
plt.show()

