import numpy as np 
from fealpy.mesh.curve import Curve1, msign
from fealpy.mesh.tree_data_structure import Quadtree
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh

import  matplotlib.pyplot as plt

class AdaptiveMarker():
    def __init__(self, phi):
        self.phi = phi

    def refine_marker(self, qtmesh):
        idx = qtmesh.leaf_cell_index()
        return idx[markedIdx]

    def coarsen_marker(self, qtmesh):
        pass

    def mark(qmesh):
        idx = qmesh.leaf_cell_index()
        polymesh = qmesh.to_polygonmesh()

        phiValue = phi(qmesh.point)
        phiSign = msign(phiValue)

        cell = qmesh.ds.cell[idx, :]
        eta1 = np.abs(phiSign[cell].sum(axis=1))
        eta2 = np.abs(phiSign[cell]).sum(axis=1)
        eta3 = np.abs(phiSign[cell[:, [0, 2]]]).sum(axis=1)
        eta4 = np.abs(phiSign[cell[:, [1, 3]]]).sum(axis=1)
        isInterfaceCell = (eta1 <= 1 ) | ((eta1 == 2) & ((eta3 == 0) | (eta4 == 0) |
            (eta2 == 4)))
        idx0, = np.nonzero(isInterfaceCell)

        NC = polymesh.number_of_cells()
        N = polymesh.number_of_points()
        isInterfaceCell = np.zeros(NC, dtype=np.int)
        isInterfaceCell[idx0] = 1 
        isInterfacePoint = np.zeros(N, dtype=np.int)
        cell = polymesh.ds.cell[idx0]
        isInterfacePoint[cell] = 1

        # Case 1
        edge = polymesh.ds.edge
        edge2cell = polymesh.ds.edge_to_cell()
        isBdInterfaceEdge = isInterfaceCell[edge2cell[:, 0]] & (~isInterfaceCell[edge2cell[:, 1]]) 
        isBdInterfaceEdge = isBdInterfaceEdge | ((~isInterfaceCell[edge2cell[:, 0]]) & isInterfaceCell[edge2cell[:, 1]])
        edge0 = edge[isBdInterfaceEdge]
        isInterfaceBdPoint = np.zeros(N, dtype=np.int) 
        isInterfaceBdPoint[edge0] = 1
        p2p = polymesh.ds.point_to_point_in_edge(N, edge0)
        isInterfaceLinkPoint = np.asarray(p2p@isInterfaceBdPoint) > 2 

        # Case 2
        isInterfaceInPoint = isInterfacePoint & (~isInterfaceBdPoint)

        c2p = polymesh.ds.cell_to_point()

        isMarkedCell = np.asarray(c2p@(isInterfaceLinkPoint | isInterfaceInPoint))


def is_interface_cell(phi, qmesh):
    idx = qmesh.leaf_cell_index()
    phiValue = phi(qmesh.point)
    phiSign = msign(phiValue)

    return idx[idx0]

phi = Curve1(a=6)
mesh = rectangledomainmesh(phi.box, nx=10, ny=10, meshtype='quad')
qmesh = Quadtree(mesh.point, mesh.ds.cell)
qmesh.uniform_refine(1)

idx = is_interface_cell(phi, qmesh)
NC = qmesh.number_of_cells()
isInterfaceCell = np.zeros(NC, dtype=np.int)
isInterfaceCell[idx] = 1

fig = plt.figure()
axes = fig.gca()
qmesh.add_plot(axes, cellcolor=isInterfaceCell)
plt.show()

