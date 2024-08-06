#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: generate_data.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 05 Aug 2024 03:22:05 PM CST
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
import matplotlib.pyplot as plt
def format_array(arr):
    rows = [np.array2string(row, separator=', ') for row in arr]
    formatted_rows = [f' [{row}],\n' for row in rows]
    return 'np.array([\n' + ''.join(formatted_rows) + '])'

mesh = TriangleMesh.from_box([0,1,0,1], 1, 1)
space = LagrangeFESpace(mesh, 2)

print(space.number_of_local_dofs())
print(space.number_of_global_dofs())
print(space.interpolation_points())
print(space.cell_to_dof())
print(space.face_to_dof())
print(space.edge_to_dof())


'''
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes,node=mesh.node,showindex=True,color='r')
mesh.find_edge(axes,showindex=True)
plt.show()
'''
