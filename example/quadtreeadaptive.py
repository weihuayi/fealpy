import numpy as np 
import sys
from fealpy.mesh.curve import Curve1, msign
from fealpy.mesh.tree_data_structure import Quadtree
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh

import  matplotlib.pyplot as plt

class AdaptiveMarker():
    def __init__(self, phi, maxh=0.01):
        self.phi = phi
        self.maxh = maxh

    def refine_marker(self, qtmesh):
        idx0, idx1 = self.mark(qtmesh)
        return idx0

    def coarsen_marker(self, qtmesh):
        pass

    def mark(self, qmesh):
        # Get the index of the leaf cells
        idx = qmesh.leaf_cell_index()

        polymesh = qmesh.to_polygonmesh()
        phi = self.phi

        phiValue = phi(qmesh.point)
        phiSign = msign(phiValue)

        cell = qmesh.ds.cell[idx, :] # the quad cell in quadtree

        eta1 = np.abs(phiSign[cell].sum(axis=1))
        eta2 = np.abs(phiSign[cell]).sum(axis=1)
        eta3 = np.abs(phiSign[cell[:, [0, 2]]]).sum(axis=1)
        eta4 = np.abs(phiSign[cell[:, [1, 3]]]).sum(axis=1)
        isInterfaceCell = (eta1 <= 1 ) | ((eta1 == 2) & ((eta3 == 0) | (eta4 == 0) | (eta2 == 4)))
        idx0, = np.nonzero(isInterfaceCell)

        N = polymesh.number_of_points()
        isInterfacePoint = np.zeros(N, dtype=np.bool)
        isInterfacePoint[cell[idx0]] = True 


        NC = polymesh.number_of_cells()
        isMarkedCell = np.zeros(NC, dtype=np.bool)
        c2p = polymesh.ds.cell_to_point()

        # Case 1
        NE = polymesh.number_of_edges()
        edge = polymesh.ds.edge # the edge in polymesh
        edge2cell = polymesh.ds.edge_to_cell()
        isInterfaceBdEdge = isInterfaceCell[edge2cell[:, 0]] & (~isInterfaceCell[edge2cell[:, 1]]) 
        isInterfaceBdEdge = isInterfaceBdEdge | ((~isInterfaceCell[edge2cell[:, 0]]) & isInterfaceCell[edge2cell[:, 1]])
        edge0 = edge[isInterfaceBdEdge]
        isInterfaceBdPoint = np.zeros(N, dtype=np.int) 
        isInterfaceBdPoint[edge0] = 1
        p2p = polymesh.ds.point_to_point_in_edge(N, edge0)
        isInterfaceLinkPoint = np.asarray(p2p@isInterfaceBdPoint) > 2 
        nlink = np.sum(isInterfaceLinkPoint) 

        if nlink > 0:
            isMarkedCell = isMarkedCell | (np.asarray(c2p@(isInterfaceLinkPoint) > 0))

        # Case 2
        nc = np.asarray(c2p.sum(axis=0)).reshape(-1)
        NV = polymesh.number_of_vertices_of_cells()
        isInterfaceInPoint = isInterfacePoint & (~isInterfaceBdPoint) & (nc == 4)
        isMarkedCell = isMarkedCell | (np.asarray(c2p@(isInterfaceInPoint) > 0))
        isInterfaceInPoint = isInterfacePoint & (~isInterfaceBdPoint) & (nc == 3)
        isMarkedCell = isMarkedCell | (np.asarray(c2p@(isInterfaceInPoint) > 0) & (NV > 4))

        # Case 3
        if nlink == 0:
            point = polymesh.point
            eps = 1e-8
            nbd = np.sum(isInterfaceBdPoint)
            normal = np.zeros((nbd, 2), dtype=np.float)
            xeps = np.array([(eps, 0)])
            yeps = np.array([(0, eps)])
            normal[:, 0] = (phi(point[isInterfaceBdPoint==1]+xeps) - phi(point[isInterfaceBdPoint==1] - xeps))/(2*eps)
            normal[:, 1] = (phi(point[isInterfaceBdPoint==1]+yeps) - phi(point[isInterfaceBdPoint==1] - yeps))/(2*eps)
            idxMap = np.zeros(N, dtype=np.int)
            idxMap[isInterfaceBdPoint==1] = np.arange(nbd)
            edge00 = idxMap[edge0]
            l = np.sqrt(np.sum(normal**2, axis=1))
            cosa = np.sum(normal[edge00[:, 0]]*normal[edge00[:, 1]], axis=1)/(l[edge00[:, 0]]*l[edge00[:, 1]])
            a = np.arccos(cosa)*180/np.pi
            isBigCurvatureEdge =  np.zeros(NE, dtype=np.bool)
            isBigCurvatureEdge[isInterfaceBdEdge] = (a > 5)
            isMarkedCell0 = np.zeros(NC, dtype=np.bool)
            isMarkedCell0[edge2cell[isBigCurvatureEdge, 0:2]] = True
            isMarkedCell = isMarkedCell | (isMarkedCell0 & isInterfaceCell)
        
        return idx[isMarkedCell], idx[isInterfaceCell]

n = int(sys.argv[1])
phi = Curve1(a=8)
mesh = rectangledomainmesh(phi.box, nx=10, ny=10, meshtype='quad')
qmesh = Quadtree(mesh.point, mesh.ds.cell)
qmesh.uniform_refine(1)

marker = AdaptiveMarker(phi)


for i in range(n):
    qmesh.refine(marker=marker)

NC = qmesh.number_of_cells()
idx0, idx1 = marker.mark(qmesh) 

isInterfaceCell = np.zeros(NC, dtype=np.int)
isInterfaceCell[idx1] = 1
isInterfaceCell[idx0] = 2

fig = plt.figure()
axes = fig.gca()
qmesh.add_plot(axes, cellcolor=isInterfaceCell)
plt.show()

