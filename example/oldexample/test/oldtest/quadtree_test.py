import sys

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

from fealpy.mesh.tree_data_structure import Quadtree
from fealpy.mesh.PolygonMesh import PolygonMesh
from fealpy.mesh.level_set_function import dcircle
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh


class AdaptiveMarker():
    def __init__(self, phi):
        self.phi = phi

    def refine_marker(self, qtmesh):
        phi = self.phi

        idx = qtmesh.leaf_cell_index()
        pmesh = qtmesh.to_pmesh()
        cell2node = pmesh.ds.cell_to_node()

        node = qtmesh.node
        value = phi(node)
        valueSign = np.sign(value)
        valueSign[np.abs(value) < 1e-8] = 0
        NV = pmesh.number_of_vertices_of_cells()
        isNeedCutCell = np.abs(cell2node*valueSign).reshape(-1) != NV

        return idx[isNeedCutCell]

    def coarsen_marker(self, qtmesh):
        phi = self.phi
        NC = qtmesh.number_of_cells()
        cell = qtmesh.ds.cell
        child = qtmesh.child
        parent = qtmesh.parent

        isRootCell = qtmesh.is_root_cell()
        if np.all(isRootCell):
            return None
        else:
            point = qtmesh.point
            value = phi(node)
            valueSign = np.sign(value)
            valueSign[np.abs(value) < 1e-12] = 0

            isLeafCell = qtmesh.is_leaf_cell()
            isBranchCell = np.zeros(NC, dtype=np.bool_)
            isBranchCell[parent[isLeafCell, 0]] = True 

            branchCell = cell[isBranchCell, :]
            isCoarsenCell = np.abs(np.sum(valueSign[branchCell], axis=1) 
                    + valueSign[cell[child[isBranchCell, 0], 2]]) == 5 
            idx, = np.nonzero(isBranchCell)

            return idx[isCoarsenCell]

box = [-1, 1, -1, 1]
cxy = (0.12, 0.1)
r = 0.5
phi = lambda p: dcircle(p, cxy, r)
circle = Circle(cxy, r, edgecolor='g', fill=False, linewidth=2)

n = 1
mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')

qtree = Quadtree(mesh.node, mesh.ds.cell)

marker = AdaptiveMarker(phi) 

for i in range(3):
    qtree.uniform_refine()

for i in range(2):
    qtree.refine(marker)


f0 = plt.figure()
axes0 = f0.gca()
qtree.add_plot(axes0)

pmesh = qtree.to_pmesh()
f1 = plt.figure()
axes1 = f1.gca()
pmesh.add_plot(axes1)
plt.show()

