#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: wpx_test.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Wed 31 Jul 2024 09:47:57 AM CST
	@bref 
	@ref 
'''  
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh, IntervalMesh, TetrahedronMesh
from fealpy.functionspace import LagrangeFiniteElementSpace

#mesh = TriangleMesh.from_box([0, 1, 0, 1], 1, 1)
#bcs = np.array([[1,0,0],[1/3,1/3,1/3]],dtype=np.float64)
#print(mesh.multi_index_matrix(p=2,etype=2))
#print(mesh.multi_index_matrix(p=2,etype=1))
#print(mesh.ds.cell_to_face())
#print(mesh.ds.cell_to_face().shape)
#print(mesh.ds.face_to_cell().shape)
#print(mesh.ds.face_to_cell())
#print(mesh.ds.face.shape)
#print(mesh.ds.face)
#print(mesh.ds.edge.shape)
#print(mesh.ds.edge)
#print(mesh.ds.cell.shape)
#print(mesh.node)
#print(mesh.entity('edge'))
#print(mesh.entity('node'))
#print(mesh.entity('cell'))
#print(mesh.entity_measure('edge'))
#print(mesh.entity_measure('cell'))
#print(mesh.edge_normal())
#print(mesh.edge_tangent())
#print(mesh.entity_barycenter('cell'))
#print(mesh.entity_barycenter('edge'))
#print(mesh.bc_to_point(bcs).swapaxes(0,1))
#print(mesh.shape_function(bcs, 2))
#print(mesh.grad_shape_function(bcs, 2, variables='u'))
#print(mesh.grad_lambda())

#mesh = IntervalMesh.from_interval([0,1], 3)
#print(mesh.grad_lambda())
#print(mesh.node)
#print(mesh.ds.cell)

#mesh = TriangleMesh.from_unit_sphere_surface()
#print(mesh.node)
#print(mesh.ds.cell)
#print(mesh.grad_lambda())
#print(mesh.entity_measure('cell'))

mesh = TetrahedronMesh.from_one_tetrahedron()
print(mesh.ds.localFace)
print(mesh.node)
print(mesh.ds.cell)
print(mesh.grad_lambda())
#print(mesh.entity_measure('cell'))

def format_array(arr):
    rows = [np.array2string(row, separator=', ') for row in arr]
    formatted_rows = [f' [{row}],\n' for row in rows]
    return 'np.array([\n' + ''.join(formatted_rows) + '])'


'''
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes,node=mesh.node,showindex=True,color='r')
mesh.find_edge(axes,showindex=True)
plt.show()
'''
