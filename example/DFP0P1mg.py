import numpy as np
import matplotlib.pyplot as plt
import time

from fealpy.pde.darcy_forchheimer_2d import DarcyForchheimerdata1
from fealpy.mg.DarcyFEMModel import DarcyP0P1
from fealpy.tools.show import showmultirate

box = [-1,1,-1,1]
mu = 1
rho = 1
beta = 10
alpha = 1/beta
level = 5
tol = 1e-6
maxN = 2000
mg_maxN = 3
J = level - 2
p = 1

n = 2

pde = DarcyForchheimerdata1(box,mu,rho,beta,alpha,level,tol,maxN,mg_maxN,J)
mesh = pde.init_mesh(n)
integrator = mesh.integrator(3) 
## Initial guess
fem = DarcyP0P1(pde,mesh,p,integrator)
fem.solve()
