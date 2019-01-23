import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, inv, spsolve
from fealpy.pde.darcy_forchheimer_2d import DarcyForchheimerdata1
from fealpy.mg.DarcyForchheimerP0P1MGModel import DarcyForchheimerP0P1MGModel

np.set_printoptions(threshold = np.inf)

box = [-1, 1, -1, 1]
mu = 1
rho = 1
beta = 10
alpha = 1/beta
tol = 1e-6
level = 4
mg_maxN = 3
maxN = 2000
p = 1

pde = DarcyForchheimerdata1(box,mu,rho,beta,alpha,level,tol,maxN,mg_maxN)
mesh = pde.init_mesh(1)
integrator = mesh.integrator(p+2)
femmg = DarcyForchheimerP0P1MGModel(pde, mesh, level)
u = femmg.mg()

