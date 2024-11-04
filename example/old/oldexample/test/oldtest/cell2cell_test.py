import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from fealpy.common import ranges
from fealpy.mesh.simple_mesh_generator import squaremesh
from meshpy.triangle import MeshInfo, build                                     
from fealpy.mesh.TriangleMesh import TriangleMesh,TriangleMeshDataStructure

x0 = 0
x1 = 1
y0 = 0
y1 = 1
r = 0

mesh = squaremesh(x0, x1, y0, y1, r)
f = plt.figure()
axes = f.gca()

N = mesh.number_of_points()
NC = mesh.number_of_cells()
NE = mesh.number_of_edges()
print(NE)
div = mesh.geom_dimension()
print("Point:\n", mesh.point)
print("Cell:\n", mesh.ds.cell)
print("Edge:\n", mesh.ds.edge)

mesh.add_plot(axes,pointcolor='k', edgecolor='k',cellcolor='grey', aspect='equal',
        linewidths=2, markersize=20, showaxis=False, showcolorbar=False)
mesh.find_point(axes, point=None,index=None, showindex=True,color='r',
        markersize=200,fontsize=24, fontcolor='k')
mesh.find_edge(axes, index=None, showindex=True,color='g', markersize=400, 
        fontsize=24, fontcolor='k')
mesh.find_cell(axes, index=None, showindex=True,color='y', markersize=800, 
        fontsize=24, fontcolor='k')

l = mesh.ds.cell_to_point()
m = mesh.ds.cell_to_edge(sparse=False)
s = mesh.ds.cell_to_cell(return_sparse=False,return_boundary=True, return_array=False)
print("cell2Point:\n", l)
print("Cell2edge:\n", m)
print("cell2cell:\n", s)
plt.show()
