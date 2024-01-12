#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2024年01月12日 星期五 11时07分10秒
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh.polygon_mesh import PolygonMesh

from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace

from fealpy.fem.scalar_mass_integrator import ScalarMassIntegrator
from fealpy.fem.bilinear_form import BilinearForm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

node = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, 1.0],
                 [0.5, 0.0], [0.5, 0.5], [0.5, 1.0],
                 [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]], dtype=np.float64)
cell = np.array([0, 3, 4, 1, 3, 6, 7, 4, 1, 4, 5, 2, 4, 7, 8, 4, 8, 5], dtype=np.int_)
cellLocation = np.array([0, 4, 8, 12, 15, 18], dtype=np.int_)
mesh = PolygonMesh(node=node, cell=cell, cellLocation=cellLocation)
#mesh = PolygonMesh.from_box([0,1,0,1],nx=1,ny=1)


normal = mesh.edge_unit_normal()
flag = mesh.ds.cell_to_edge_sign()
print(normal)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, color='r', marker='o', markersize=8, fontsize=16, fontcolor='r')
mesh.find_cell(axes, showindex=True, color='b', marker='o', markersize=8, fontsize=16, fontcolor='b')
mesh.find_edge(axes, showindex=True, color='g', marker='o', markersize=8, fontsize=16, fontcolor='g')
plt.show()
