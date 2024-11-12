import numpy as np
import matplotlib.pyplot as plt
from meshpy.triangle import MeshInfo, build

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.functionspace.mixed_fem_space import RaviartThomasFiniteElementSpace2d 
from fealpy.functionspace.mixed_fem_space import BDMFiniteElementSpace2d
from fealpy.functionspace.mixed_fem_space import FirstNedelecFiniteElement2d 

def mesh_2dpy(mesh_info,h):
    mesh = build(mesh_info,max_volume=(h)**2)
    point = np.array(mesh.points,dtype=np.float)
    cell = np.array(mesh.elements,dtype=np.int)
    tmesh =TriangleMesh(point,cell)
    return tmesh

mesh_info = MeshInfo()  

mesh_info.set_points([(0,0), (1,0), (1,1), (0,1)])
mesh_info.set_facets([[0,1], [1,2], [2,3], [3,0]])
h = 0.5
tmesh = mesh_2dpy(mesh_info,h)

#point = np.array([
#    (0, 0), 
#    (1, 0), 
#    (1, 1),
#    (0, 1)], dtype=np.float)
#cell = np.array([
#    (1, 2, 0), 
#    (3, 0, 2)], dtype=np.int)
#tmesh = TriangleMesh(point, cell)

space = RaviartThomasFiniteElementSpace2d(tmesh)
#space = BDMFiniteElementSpace2d(tmesh)
#space = FirstNedelecFiniteElement2d(tmesh)

ldof = space.number_of_local_dofs()

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
tmesh.find_point(axes, showindex=True)
tmesh.find_edge(axes, showindex=True)
tmesh.find_cell(axes, showindex=True)
tmesh.print()
print('cell2edgeSign:\n', space.cell_to_edge_sign())


cell2edge = tmesh.ds.cell_to_edge()
n = tmesh.edge_unit_normal()

cell = tmesh.ds.cell
point = tmesh.point
a = np.zeros((9, 2), dtype=np.float)
a[:, 1]= np.arange(0.2, 1, 0.2)
a[:, 0] = 1 - a[:, 1]

for li, lj in a:
    p = li*point[edge[:, 0]] + lj*point[edge[:, 1]]

plt.show()
