import numpy as np
from fealpy.mesh import HexahedronMesh
from icecream import ic

node = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], 
                 [2, 1, 0], [1, 1, 0], [0, 1, 0],
                 [0, 0, 1], [1, 0, 1], [2, 0, 1], 
                 [2, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float_)
cell = np.array([[0, 1, 4, 5, 6, 7, 10, 11], [1, 2, 3, 4, 7, 8, 9, 10]],
        dtype=np.int_)

#node = np.array([[0, 0, 0], [1, 0, 0],  
#                 [1, 1, 0], [0, 1, 0],
#                 [0, 0, 1], [1, 0, 1], 
#                 [1, 1, 1], [0, 1, 1]], dtype=np.float_)
#cell = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int_)
mesh = HexahedronMesh(node, cell)


edge = mesh.entity('edge')
face = mesh.entity('face')
face2edge = mesh.ds.face_to_edge()
ic(edge)
ic(face)
ic(face2edge)

cell2ipoint = mesh.cell_to_ipoint(3)
face2ipoint = mesh.face_to_ipoint(3)
ic(cell2ipoint)
ic(face2ipoint)

point = mesh.interpolation_points(3)
ic(point)

ic(cell2ipoint[0, ::4])
ic(cell2ipoint[0, 3::4])
ic(cell2ipoint[0, :16])
ic(cell2ipoint[0, -16:])

idx = np.arange(4)*16
idx = np.linspace(idx, idx+3, 4, endpoint=True).T.flatten().astype(np.int_)
ic(idx)
ic(cell2ipoint[0, idx])

idx = np.arange(4)*16+12
idx = np.linspace(idx, idx+3, 4, endpoint=True).T.flatten().astype(np.int_)
ic(idx)
ic(cell2ipoint[0, idx])



import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

axes = a3.Axes3D(pl.figure())
mesh.add_plot(axes, alpha=0.1)
mesh.find_node(axes, node = point, showindex=True, markersize=300, color='r')
mesh.find_cell(axes, showindex=True, markersize=600, color='k')
pl.show()
