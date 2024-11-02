#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test_node_set.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年10月25日 星期三 10时47分40秒
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.mesh import NodeSet 
from fealpy.functionspace import NodeSetKernelSpace 
import matplotlib.pyplot as plt

dx = 0.025
dy = 0.025
rho0 = 1000
H = 0.92*np.sqrt(dx**2+dy**2)
dt = 0.001
c0 = 10
g = np.array([0.0, -9.8])
gamma = 7
alpha = 0.3
maxstep = 10000

mesh = NodeSet.from_dam_break_domain(dx, dy)
fig, axes = plt.subplots()
color = np.where(mesh.is_boundary_node(), 'red', 'blue')
mesh.add_plot(axes,color=color)
#plt.show()

name = ["velocity", "rho", "mass", "pressure", "sound"]
dtype = [("float64", (2, )),
         ("float64"),
         ("float64"),
         ("float64"),
         ("float64")]


mesh.add_node_data(name, dtype)
mesh.set_node_data("rho", rho0)
NN = mesh.number_of_nodes()
NB = np.sum(mesh.is_boundary_node())
NF = NN - NB
mesh.set_node_data("mass", 2*rho0/NF)

space = NodeSetKernelSpace(mesh, H=H)

re = space.kernel(0.02)
print(re)
