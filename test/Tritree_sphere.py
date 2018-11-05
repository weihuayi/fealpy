import sys
import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl
from fealpy.mesh.level_set_function import Sphere
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.tree_data_structure import Tritree

#m = int(sys.argv[1])
class AdaptiveMarker():
    def __init__(self):
        pass
         
    def refine_marker(self, tmesh):
        node = tmesh.entity('node')
        cell = tmesh.entity('cell')
        isLeafCell = tmesh.is_leaf_cell()
        flag = (np.sum(cell == 0, axis=1) == 1) & isLeafCell
        idx, = np.where(flag)
        return idx

    def coarsen_marker(self, qtmesh):
        pass
surface = Sphere()
mesh = surface.init_mesh()
mesh.uniform_refine(n=2, surface = surface)
node = mesh.node
cell = mesh.ds.cell
tmesh = Tritree(node, cell, irule=1)
marker = AdaptiveMarker()
for i in range(4):
    tmesh.refine(marker, surface=surface)
print(tmesh.node)
fig0 = pl.figure()
axes0 = a3.Axes3D(fig0)
tmesh.add_plot(axes0)

pmesh = tmesh.to_conformmesh()
fig1 = pl.figure()
axes1 = a3.Axes3D(fig1)
pmesh.add_plot(axes1)
pl.show()
