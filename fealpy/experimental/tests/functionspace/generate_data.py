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
from fealpy.decorator import barycentric, cartesian

def test_fun(point): 
    x = point[..., 0]
    y = point[..., 1]
    result = np.sin(x) * np.sin(y)
    return result

mesh = TriangleMesh.from_box([0,1,0,1], 1, 1)
space = LagrangeFESpace(mesh, 2)
uh = space.interpolate(test_fun)
bcs = np.array([[0, 0, 1], [0, 1, 0], [1/3, 1/3, 1/3]], dtype=np.float64)

#print(uh)
#print(space.value(uh, bcs).swapaxes(1,0))
print(space.grad_basis(bcs).swapaxes(0,1))
#print(space.grad_value(uh, bcs).shape)


'''
print(space.number_of_local_dofs())
print(space.number_of_global_dofs())
print(space.interpolation_points())
print(space.cell_to_dof())
print(space.face_to_dof())
print(space.edge_to_dof())
'''

'''
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes,node=mesh.node,showindex=True,color='r')
mesh.find_edge(axes,showindex=True)
plt.show()
'''
