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

dtype = [("position", "float64", (2, )), 
         ("velocity", "float64", (2, )),
         ("rho", "float64"),
         ("mass", "float64"),
         ("pressure", "float64")]

num=10
np.random.seed(0)
random_points = np.random.rand(num, 2)
particles = np.zeros(num, dtype=dtype)
particles['position'] = random_points
print(random_points)

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
solver.e(particles["position"])
