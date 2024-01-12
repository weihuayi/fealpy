#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: poisson_quadranglemesh_fvm_dirichletbc_2d.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2024年01月02日 星期二 19时20分18秒
	@bref 
	@ref 
'''  
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import QuadrangleMesh
from fealpy.fvm import ScalarDiffusionIntegrator
def sincos(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sin(x)*np.cos(y)
    return val

ns = 5
mesh = QuadrangleMesh.from_box(box=[0,1,0,1],nx=ns,ny=ns)
DI = ScalarDiffusionIntegrator(mesh)
#DM,Db = DI.cell_center_matrix()

b = mesh.integral(sincos, celltype=True)
bcs = (np.array([1/2,1/2]),np.array([1/2,1/2]))
cpoint = mesh.bc_to_point(bcs)
edgemeasure = mesh.entity_measure('edge')
print(edgemeasure)

bpoint = mesh.bc_to_point([1/2,1/2])
edge = mesh.ds.boundary_edge_flag()
print(cpoint)

fig = plt.figure()
axes  = fig.gca()
#mesh.find_node(axes,node=point,markersize=20)
mesh.find_edge(axes,showindex=True, markersize=20)
mesh.add_plot(axes)
#plt.show()
