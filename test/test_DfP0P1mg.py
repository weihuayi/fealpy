import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, inv, spsolve
from fealpy.pde.darcy_forchheimer_2d import DarcyForchheimerdata1
from fealpy.mg.DarcyForchheimerP0P1MGModel import DarcyForchheimerP0P1MGModel

box = [-1, 1, -1, 1]
mu = 1
rho = 1
beta = 10
alpha = 1/beta
tol = 1e-6
level = 4
mg_maxN = 1
maxN = 2000
p = 1

pde = DarcyForchheimerdata1(box,mu,rho,beta,alpha,level,tol,maxN,mg_maxN)
mesh = pde.init_mesh(1)
NC = mesh.number_of_cells()
print('NC',NC)
NN = mesh.number_of_nodes()

integrator = mesh.integrator(p+2)
femmg = DarcyForchheimerP0P1MGModel(pde, mesh, level)
A = femmg.stiff_matrix()
B = femmg.compute_initial_value()
#m = 0
#error = np.ones(maxN, dtype=np.float)
#residual = np.ones(maxN, dtype=np.float)
#Ndof = np.zeros(maxN, dtype=np.int)
#u = np.zeros(2*NC + NN)
#while residual[m] > tol and m < maxN:
#    uold = np.zeros(u.shape)
#    uold[:] = u
#

