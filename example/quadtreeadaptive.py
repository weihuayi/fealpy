import numpy as np 
import sys
from fealpy.mesh.curve import Curve1, msign
from fealpy.mesh.tree_data_structure import Quadtree
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
from fealpy.mesh.interface_mesh_generator import find_cut_point 

import  matplotlib.pyplot as plt

class AdaptiveMarker():
    def __init__(self, phi, maxh=0.01, maxa=5):
        self.phi = phi
        self.maxh = maxh
        self.maxa = maxa

    def refine_marker(self, qtmesh):
        idx = qtmesh.leaf_cell_index()
        polymesh = qtmesh.to_polygonmesh()
        isMarkedCell = self.mark(polymesh)
        return idx[isMarkedCell]

    def coarsen_marker(self, qtmesh):
        pass

    def interface_cell_flag(self, polymesh):
        phi = self.phi
        c2p = polymesh.ds.cell_to_point()
        phiValue = phi(qmesh.point)
        phiSign = msign(phiValue)
        NV = polymesh.number_of_vertices_of_cells()

        eta1 = np.abs(c2p*phiSign)
        eta2 = c2p*np.abs(phiSign)

        isInterfaceCell = (eta1 < eta2) | ((eta1 == eta2) & ((NV - eta2) > 2))

        return isInterfaceCell 

    def mark(self, polymesh):
        # Get the index of the leaf cells
        phi = self.phi
        c2p = polymesh.ds.cell_to_point()

        phiValue = phi(qmesh.point)
        phiSign = msign(phiValue)
        NV = polymesh.number_of_vertices_of_cells()

        eta1 = np.abs(c2p*phiSign)
        eta2 = c2p*np.abs(phiSign)

        isInterfaceCell = (eta1 < eta2) | ((eta1 == eta2) & ((NV - eta2) > 2))

        idx0, = np.nonzero(isInterfaceCell)

        N = polymesh.number_of_points()
        isInterfacePoint= np.asarray(isInterfaceCell@c2p).reshape(-1)


        NC = polymesh.number_of_cells()
        isMarkedCell = np.zeros(NC, dtype=np.bool)

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
        return isMarkedCell

phi = Curve1(a=3)
mesh = rectangledomainmesh(phi.box, nx=10, ny=10, meshtype='quad')
qmesh = Quadtree(mesh.point, mesh.ds.cell)
qmesh.uniform_refine(1)

marker = AdaptiveMarker(phi)


flag = True
while flag:
    flag = qmesh.refine(marker=marker)
polymesh = qmesh.to_polygonmesh()

N = polymesh.number_of_points()
NE = polymesh.number_of_edges()
NC = polymesh.number_of_cells()

# find the interface points 
edge = polymesh.ds.edge
point = polymesh.point
phiSign = msign(phi(point))
isInterfaceCell = marker.interface_cell_flag(polymesh)
edge2cell = polymesh.ds.edge2cell

isCutEdge0 = (phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0)
isCutEdge1 = (phiSign[edge[:, 0]] == 0)
isCutEdge2 = (phiSign[edge[:, 1]] == 0)
cutEdge0 = edge[isCutEdge0]
cutPoint = find_cut_point(phi, point[cutEdge0[:, 0]], point[cutEdge0[:, 1]])
newPoint = np.append(point, cutPoint, axis=0)

# find the cut location in each interface cell
NV = polymesh.number_of_vertices_of_cells()
NIC = isInterfaceCell.sum()
cellIdxMap = np.zeros(NC, dtype=np.int)
cellIdxMap[isInterfaceCell] = range(NIC)

cell = polymesh.ds.cell
cellLocation = polymesh.ds.cellLocation

location0 = -np.ones(NIC, dtype=np.int)
location1 = -np.ones(NIC, dtype=np.int)

location0[cellIdxMap[edge2cell[isCutEdge0, 0]]] = edge2cell[isCutEdge0, 2]
location1[cellIdxMap[edge2cell[isCutEdge0, 1]]] = edge2cell[isCutEdge0, 3]

isCase = isCutEdge1 & isInterfaceCell[edge2cell[:, 0]] 
location0[cellIdxMap[edge2cell[isCase, 0]]] = edge2cell[isCase, 2]
isCase = isCutEdge1 & isInterfaceCell[edge2cell[:, 1]] 
location1[cellIdxMap[edge2cell[isCase, 1]]] = (edge2cell[isCase, 3] + 1)%NV[edge2cell[isCase, 1]]

isCase = isCutEdge2 & isInterfaceCell[edge2cell[:, 0]]
location0[cellIdxMap[edge2cell[isCase, 0]]] = (edge2cell[isCase, 2] + 1)%NV[edge2cell[isCase, 0]]
isCase = isCutEdge2 & isInterfaceCell[edge2cell[:, 1]] 
location1[cellIdxMap[edge2cell[isCase, 1]]] = edge2cell[isCase, 3] 

location = np.append(location0.reshape(-1, 1), location1.reshape(-1, 1), axis=1)
print(location)

idx0, idx1= np.nonzero(location == -1)
print(idx0.shape[0])
print(idx1.shape[0])

NV0 = NV[isInterfaceCell]
initLocation = cellLocation[:-1][isInterfaceCell]


fig = plt.figure()
axes = fig.gca()
polymesh.add_plot(axes, cellcolor=isInterfaceCell)
plt.show()

