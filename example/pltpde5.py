import numpy as np
import matplotlib.pyplot as plt
import time

from fealpy.pde.darcy_forchheimer_2d import DeltaData
from fealpy.fdm.velocity import NonDarcyForchheimerFDMModel
#from fealpy.fdm.DarcyForchheimerFDMModel_pu import DarcyForchheimerFDMModel
#from fealpy.fdm.DarcyForchheimerFDMModelpu import DarcyForchheimerFDMModel
#from fealpy.fdm.testDarcyForchheimerFDMModelpu import DarcyForchheimerFDMModel

box = [0,1,0,1]
mu = 2
k = 1
rho = 1
beta = 5
tol = 1e-6
#hx = np.array([0.12,0.34,0.22,0.32])
#hy = np.array([0.25,0.13,0.33,0.29])
#hy = np.array([0.16,0.23,0.32,0.11,0.18])
hx = np.array([0.25,0.25,0.25,0.25])
hy = np.array([0.25,0.25,0.25,0.25])
#hy = np.array([0.2,0.2,0.2,0.2,0.2])
m = 8
hx = hx/8
hy = hy/8
hx = hx.repeat(8)
hy = hy.repeat(8)

pde = DeltaData(box,mu,k,rho,beta,tol)
t1 = time.time()
mesh = pde.init_mesh(hx,hy)
isYDEdge = mesh.ds.y_direction_edge_flag()
fdm = NonDarcyForchheimerFDMModel(pde,mesh)
count,uh = fdm.solve()
print('count',count)
nx = hx.shape[0]
ny = hy.shape[0]
X,Y = np.meshgrid(np.arange(0,1,hx[0]),np.arange(0,1,hy[0]))
u = uh[:sum(isYDEdge)]
U = u.reshape(ny,nx+1)
v = uh[sum(isYDEdge):]
V = v.reshape(ny+1,nx)
plt.figure()
Q = plt.quiver(X,Y,U,0)
W = plt.quiver(X,Y,0,V)
plt.show()
