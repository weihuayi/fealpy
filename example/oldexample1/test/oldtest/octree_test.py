
import sys

import numpy as np

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

from fealpy.mesh.tree_data_structure import Octree
from fealpy.mesh.level_set_function import  Sphere
from fealpy.mesh.vtkMeshIO import write_vtk_mesh


class AdaptiveMarker():
    def __init__(self, phi):
        self.phi = phi

    def refine_marker(self, octree):
        phi = self.phi

        cell = octree.ds.cell
        idx = octree.leaf_cell_index()
        cell = cell[idx, :]

        point = octree.point
        value = phi(point)
        valueSign = np.sign(value)
        valueSign[np.abs(value) < 1e-12] = 0
        isNeedCutCell = np.abs(np.sum(valueSign[cell], axis=1)) < 8

        return idx[isNeedCutCell]

    def coarsen_marker(self, octree):
        phi = self.phi
        NC = octree.number_of_cells()
        cell = octree.ds.cell
        tree = octree.tree

        point = octree.point
        value = phi(point)
        valueSign = np.sign(value)
        valueSign[np.abs(value) < 1e-12] = 0

        isLeafCell = octree.is_leaf_cell()
        isBranchCell = np.zeros(NC, dtype=np.bool_)
        isBranchCell[tree[isLeafCell, 0]] = True 

        branchCell = cell[isBranchCell, :]
        isCoarsenCell = np.abs(np.sum(valueSign[branchCell], axis=1) 
                + valueSign[cell[tree[isBranchCell, 1], 1]]) == 5 
        idx, = np.nonzero(isBranchCell)

        return idx[isCoarsenCell]
    

point = 2*np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1]], dtype=np.float)

cell = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int)
octree = Octree(point, cell)
phi = Sphere()
marker = AdaptiveMarker(phi)

for i in range(1):
    octree.uniform_refine()

for i in range(8):
    octree.refine(marker)

pmesh = octree.to_pmesh()
write_vtk_mesh(pmesh, 'pmesh.vtk')
print(pmesh.check())
