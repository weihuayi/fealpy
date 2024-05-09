#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 13 Apr 2024 04:34:02 PM CST
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.cfd import NSFlipSolver
from fealpy.mesh import UniformMesh2d
import matplotlib.pyplot as plt

R = 8.314 #普适气体常数
Cv = 4186 #比热容 
mu = 1 #剪切粘度
lam = -(2/3)*mu #体积粘度

dtype = [("position", "float64", (2, )), 
         ("velocity", "float64", (2, )),
         ("rho", "float64"),
         ("mass", "float64"),
         ("pressure", "float64"),
		 ("internal_energy", "float64"),]

num=10
np.random.seed(0)
random_points = np.random.rand(num, 2)
particles = np.zeros(num, dtype=dtype)
particles['position'] = random_points
particles['mass'] = 1 #临时给的值
particles['internal_energy'] = 1 #临时给的值
particles['velocity'] = np.array([0.0, 1.0]) #临时给的值
#print(random_points)

nt = 100
domain=[0,1,0,1]
nx = 4
ny = 4
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0,nx,0,ny],h=(hx,hy),origin=(0,0))
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.scatter(particles["position"][:,0], particles["position"][:,1])
#plt.show()

solver = NSFlipSolver(particles, mesh)
num_v = (mesh.ds.nx + 1)*(mesh.ds.ny + 1)
vertex = mesh.node.reshape(num_v,2)
#e = solver.e(particles["position"])
#solver.coordinate(particles["position"])
#solver.NGP(particles["position"],vertex)
Sv = solver.bilinear(particles["position"],vertex)
grad_Sv = Sv.todense() #形状有疑问？
solver.P2G_cell(particles)
rho_c ,I_c =  solver.P2G_cell(particles)
#solver.P2G_vertex(particles)
p = solver.pressure(rho_c,I_c,R,Cv)
Q = np.matmul(-(lam+2*mu)*grad_Sv.T,particles['velocity'])
phi = np.einsum('ij,ik->ijk', mu*grad_Sv, particles['velocity'])
