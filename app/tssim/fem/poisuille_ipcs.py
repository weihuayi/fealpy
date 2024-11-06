#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: chorin.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 07 Mar 2024 09:53:05 AM CST
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.pde.navier_stokes_equation_2d import Poisuille
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.cfd import NSFEMSolver
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DirichletBC
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib
#get basic parameters
ns = 16
udegree= 2
pdegree = 1
T = 10
nt = 1000
pde = Poisuille()
rho = pde.rho
mu = pde.mu

#generate mesh space 
mesh = TriangleMesh.from_box(pde.domain(), nx = ns, ny = ns)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt
uspace = LagrangeFESpace(mesh, p=udegree, doforder='sdofs')
pspace = LagrangeFESpace(mesh, p=pdegree, doforder='sdofs')

solver = NSFEMSolver(mesh, dt, uspace, pspace, rho, mu, 6)

us = uspace.function(dim = 2)
u0 = uspace.function(dim = 2)
u1 = uspace.function(dim = 2)

p0 = pspace.function()
p1 = pspace.function()

source = uspace.interpolate(pde.source, dim=2)
ubc = DirichletBC(uspace, pde.velocity, pde.is_wall_boundary) 
pbc = DirichletBC(pspace, pde.pressure, pde.is_p_boundary) 
#ubc = DirichletBC(uspace, pde.velocity) 
#pbc = DirichletBC(pspace, pde.pressure) 
uso = uspace.interpolate(pde.velocity,2)
pso = pspace.interpolate(pde.pressure)
errorMatrix = np.zeros((4,nt),dtype=np.float64)

for i in range(0, 10):
    t1 = timeline.next_time_level()
    print("time=", t1)
    
    A0 = solver.ipcs_A_0(threshold=pde.is_p_boundary)
    b0 = solver.ipcs_b_0(u0, p0, source, threshold=pde.is_p_boundary)
    A0,b0 = ubc.apply(A0,b0)
    us[:]= spsolve(A0, b0).reshape((2,-1))
    print(np.sum(np.abs(us)))

    A1 = solver.ipcs_A_1() 
    b1 = solver.ipcs_b_1(us, p0)
    A1,b1 = pbc.apply(A1,b1)
    p1[:] = spsolve(A1,b1) 

    A2 = solver.ipcs_A_2() 
    b2 = solver.ipcs_b_2(us, p1, p0)
    u1[:] = spsolve(A2,b2).reshape((2,-1))
    
    u0[:] = u1
    p0[:] = p1
    errorMatrix[0,i] = mesh.error(pde.velocity, u1)
    errorMatrix[1,i] = mesh.error(pde.pressure, p1)
    errorMatrix[2,i] = np.abs(uso-u1).max()
    errorMatrix[3,i] = np.abs(pso-p1).max()
    timeline.advance()
    #print("max u:",errorMatrix[0,i])
    print("max u:",np.max(u1))

'''
ipoint = mesh.interpolation_points(2)
nx = ns
ny = ns
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
X = ipoint[..., 0]
Y = ipoint[..., 1]
Z = np.array((u1-uso)[0,...])
print(Z.shape)
plt.scatter(X, Y, c=Z, cmap='viridis')
plt.colorbar()
plt.show()
'''
'''
print(errorMatrix[0,-1])
print(errorMatrix[1,-1])
print(errorMatrix[2,-1])
print(errorMatrix[3,-1])
'''

'''
fig1 = plt.figure()
node = mesh.node
x = tuple(node[:,0])
y = tuple(node[:,1])
NN = mesh.number_of_nodes()
u = u1[:,:NN]
ux = tuple(u[0,:])
uy = tuple(u[1,:])

o = ux
norm = matplotlib.colors.Normalize()
cm = matplotlib.cm.copper
sm = matplotlib.cm.ScalarMappable(cmap=cm,norm=norm)
sm.set_array([])
plt.quiver(x,y,ux,uy,color=cm(norm(o)))
plt.colorbar(sm)
plt.show()
'''
