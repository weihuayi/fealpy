import numpy as np 
import sys
from .PolygonMesh import PolygonMesh 
from .tree_data_structure import Quadtree
from .tree_data_structure import Octree
from .simple_mesh_generator import rectangledomainmesh
from .interface_mesh_generator import find_cut_point 

def msign(x):
    flag = np.sign(x)
    flag[np.abs(x) < 1e-8] = 0
    return flag

class AdaptiveMarkerBase():
    def __init__(self, phi, maxh, maxa):
        self.phi = phi
        self.maxh = maxh
        self.maxa = maxa

    def refine_marker(self, treemesh):
        idx = treemesh.leaf_cell_index()
        pmesh = treemesh.to_pmesh()
        isMarkedCell = self.refine_mark(pmesh, treemesh)
        return idx[isMarkedCell]

    def coarsen_marker(self, treemesh):
        idx = treemesh.leaf_cell_index()
        pmesh = treemesh.to_pmesh()
        isMarkedCell = self.coarsen_mark(pmesh, treemesh)
        return idx[isMarkedCell]

    def interface_cell_flag(self, pmesh):
        pass

    def refine_mark(self, pmesh, treemesh):
        pass

    def coarsen_mark(self, pmesh, treemesh):
        pass


class AdaptiveMarker2d(AdaptiveMarkerBase):
    def __init__(self, phi, maxh=0.01, maxa=5):
        super(AdaptiveMarker2d, self).__init__(phi, maxh, maxa)

    def interface_cell_flag(self, pmesh):
        phi = self.phi
        cell2node = pmesh.ds.cell_to_node()
        phiValue = phi(pmesh.node)
        phiSign = msign(phiValue)
        NV = pmesh.number_of_vertices_of_cells()

        eta1 = np.abs(cell2node*phiSign)
        eta2 = cell2node*np.abs(phiSign)

        isInterfaceCell = (eta1 < eta2) | ((eta1 == eta2) & ((NV - eta2) > 2))

        return isInterfaceCell 

    def refine_mark(self, pmesh, treemesh=None):
        phi = self.phi
        cell2node = pmesh.ds.cell_to_node()

        phiValue = phi(pmesh.node)
        phiSign = msign(phiValue)
        NV = pmesh.number_of_vertices_of_cells()

        eta1 = np.abs(cell2node*phiSign)
        eta2 = cell2node*np.abs(phiSign)

        isInterfaceCell = (eta1 < eta2) | ((eta1 == eta2) & ((NV - eta2) > 2))

        idx0, = np.nonzero(isInterfaceCell)

        N = pmesh.number_of_nodes()
        isInterfaceNode= np.asarray(isInterfaceCell@cell2node).reshape(-1)


        NC = pmesh.number_of_cells()
        isMarkedCell = np.zeros(NC, dtype=np.bool_)

        # Case 1
        NE = pmesh.number_of_edges()
        edge = pmesh.ds.edge # the edge in polymesh
        edge2cell = pmesh.ds.edge_to_cell()
        isInterfaceBdEdge = isInterfaceCell[edge2cell[:, 0]] & (~isInterfaceCell[edge2cell[:, 1]]) 
        isInterfaceBdEdge = isInterfaceBdEdge | ((~isInterfaceCell[edge2cell[:, 0]]) & isInterfaceCell[edge2cell[:, 1]])
        edge0 = edge[isInterfaceBdEdge]
        isInterfaceBdNode = np.zeros(N, dtype=np.int) 
        isInterfaceBdNode[edge0] = 1
        node2node = pmesh.ds.node_to_node_in_edge(N, edge0)
        isInterfaceLinkNode = np.asarray(node2node@isInterfaceBdNode) > 2 
        nlink = np.sum(isInterfaceLinkNode) 

        if nlink > 0:
            isMarkedCell = isMarkedCell | (np.asarray(cell2node@(isInterfaceLinkNode) > 0))

        # Case 2
        nc = np.asarray(cell2node.sum(axis=0)).reshape(-1)
        NV = pmesh.number_of_vertices_of_cells()
        isInterfaceInNode = isInterfaceNode & (~isInterfaceBdNode) & (nc == 4)
        isMarkedCell = isMarkedCell | (np.asarray(cell2node@(isInterfaceInNode) > 0))
        isInterfaceInNode = isInterfaceNode & (~isInterfaceBdNode) & (nc == 3)
        isMarkedCell = isMarkedCell | (np.asarray(cell2node@(isInterfaceInNode) > 0) & (NV > 4))

        # Case 3
        if nlink == 0:
            node = pmesh.node
            eps = 1e-8
            nbd = np.sum(isInterfaceBdNode)
            normal = np.zeros((nbd, 2), dtype=np.float)
            xeps = np.array([(eps, 0)])
            yeps = np.array([(0, eps)])
            normal[:, 0] = (phi(node[isInterfaceBdNode==1]+xeps) -
                    phi(node[isInterfaceBdNode==1] - xeps))/(2*eps)
            normal[:, 1] = (phi(node[isInterfaceBdNode==1]+yeps) -
                    phi(node[isInterfaceBdNode==1] - yeps))/(2*eps)
            idxMap = np.zeros(N, dtype=np.int)
            idxMap[isInterfaceBdNode==1] = np.arange(nbd)
            edge00 = idxMap[edge0]
            l = np.sqrt(np.sum(normal**2, axis=1))
            cosa = np.sum(normal[edge00[:, 0]]*normal[edge00[:, 1]], axis=1)/(l[edge00[:, 0]]*l[edge00[:, 1]])
            a = np.arccos(cosa)*180/np.pi
            isBigCurvatureEdge =  np.zeros(NE, dtype=np.bool_)
            isBigCurvatureEdge[isInterfaceBdEdge] = (a > 5)
            isMarkedCell0 = np.zeros(NC, dtype=np.bool_)
            isMarkedCell0[edge2cell[isBigCurvatureEdge, 0:2]] = True
            isMarkedCell = isMarkedCell | (isMarkedCell0 & isInterfaceCell)

        # Case 4
        a = pmesh.area()
        maxh = self.maxh
        isMarkedCell = isMarkedCell | ((a > maxh**2) & isInterfaceCell) 


        return isMarkedCell


class QuadtreeInterfaceMesh2d():
    def __init__(self, mesh, marker):
        if mesh.meshType is 'quad':
            self.treemesh =  Quadtree(mesh.node, mesh.ds.cell)
        elif mesh.meshType is 'quadtree':
            self.treemesh = mesh

        self.treemesh.uniform_refine() # here one should refine one time 
                                    # TODO: there is a bug in Quadtree
        self.marker = marker
        self.adaptive_refine()

    def adaptive_refine(self):
        flag = True
        while flag:
            flag = self.treemesh.refine(marker=self.marker)

    def get_interface_mesh(self):
        phi = self.marker.phi
        pmesh = self.treemesh.to_pmesh()

        N = pmesh.number_of_nodes()
        NE = pmesh.number_of_edges()
        NC = pmesh.number_of_cells()

        # find the interface nodes 
        edge = pmesh.ds.edge
        node = pmesh.node
        phiSign = msign(phi(node))
        isInterfaceCell = self.marker.interface_cell_flag(pmesh)
        edge2cell = pmesh.ds.edge2cell

        isCutEdge0 = (phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0)
        cutEdge0 = edge[isCutEdge0]
        cutNode = find_cut_point(phi, node[cutEdge0[:, 0]], node[cutEdge0[:, 1]])

        edge2cutNode = np.zeros(NE, dtype=np.int)
        edge2cutNode[isCutEdge0] = range(N, N+cutEdge0.shape[0])

        isSpecialEdge = (~isCutEdge0) & isInterfaceCell[edge2cell[:, 0]] \
                & isInterfaceCell[edge2cell[:, 1]]

        if np.any(isSpecialEdge):
            print(isSpecialEdge.sum())
            isSpecialEdge0 = isSpecialEdge & (phiSign[edge[:, 0]] == 0)
            edge2cutNode[isSpecialEdge0] = edge[isSpecialEdge0, 0]
            isCutEdge0[isSpecialEdge0] = True
            isSpecialEdge0 = isSpecialEdge & (phiSign[edge[:, 1]] == 0)
            edge2cutNode[isSpecialEdge0] = edge[isSpecialEdge0, 1]
            isCutEdge0[isSpecialEdge0] = True

        edge2cutNode = edge2cutNode[isCutEdge0]

        # find the cut location in each interface cell
        NV = pmesh.number_of_vertices_of_cells()
        NIC = isInterfaceCell.sum() # the number of interface cell
        cellIdxMap = np.zeros(NC, dtype=np.int)
        cellIdxMap[isInterfaceCell] = range(NIC) # renumbering the interface cell 

        cell = pmesh.ds.cell
        cellLocation = pmesh.ds.cellLocation


        # find the neighbor cell idx of cut edges
        cellIdx = edge2cell[isCutEdge0, 0:2].reshape(-1)

        idx = np.argsort(cellIdx)
        location = edge2cell[isCutEdge0, 2:4].reshape(-1)[idx].reshape(-1, 2)
        interfaceCell2cutNode = np.repeat(edge2cutNode, 2)[idx].reshape(-1, 2) 
        idx = np.argsort(location, axis=1)
        location = location[np.arange(NIC).reshape(-1, 1), idx]
        interfaceCell2cutNode = interfaceCell2cutNode[np.arange(NIC).reshape(-1, 1), idx]


        NV0 = NV[isInterfaceCell]
        initLocation = cellLocation[:-1][isInterfaceCell]

        NNC = NV0.shape[0]

        NV1 = location[:, 1] - location[:, 0] + 2

         
        newCell1 = np.zeros(np.sum(NV1), dtype=np.int)
        newCellLocation1 = np.zeros(NNC+1, dtype=np.int)
        newCellLocation1[1:] = np.cumsum(NV1)
        newCell1[newCellLocation1[:-1]] = interfaceCell2cutNode[:, 0]
        newCell1[newCellLocation1[:-1]+NV1-1] = interfaceCell2cutNode[:, 1]
        for i in range(NNC):
            idx0 =np.arange(newCellLocation1[i]+1, newCellLocation1[i]+NV1[i]-1) 
            idx1 = np.arange(initLocation[i]+location[i, 0]+1, 
                    initLocation[i]+location[i, 1]+1)
            newCell1[idx0] = cell[idx1]

        NV2 = NV0 - NV1 + 4
        newCell2 = np.zeros(np.sum(NV2), dtype=np.int)
        newCellLocation2 = np.zeros(NNC+1, dtype=np.int)
        newCellLocation2[1:] = np.cumsum(NV2)
        newCell2[newCellLocation2[:-1]+ location[:, 0] + 1] = interfaceCell2cutNode[:, 0]
        newCell2[newCellLocation2[:-1] + location[:, 0] + 2] = interfaceCell2cutNode[:, 1] 
        for i in range(NNC): 
            idx0 = np.arange(newCellLocation2[i], newCellLocation2[i] + location[i, 0] +1)
            idx1 = np.arange(initLocation[i], initLocation[i] + location[i, 0] + 1)
            newCell2[idx0] = cell[idx1]
            idx0 = np.arange(newCellLocation2[i] + location[i, 0] + 3, newCellLocation2[i] + NV2[i]) 
            idx1 = np.arange(initLocation[i] + location[i, 1] + 1, initLocation[i] + NV0[i])
            newCell2[idx0] = cell[idx1]

        np.repeat(isInterfaceCell, NV)
        NV3 = NV[~isInterfaceCell]
        newCell3 = cell[np.repeat(~isInterfaceCell, NV)]
        newCellLocation3 = np.zeros(NV3.shape[0]+1, dtype=np.int)
        newCellLocation3[1:] = np.cumsum(NV3)

        cell = np.concatenate((newCell3, newCell1, newCell2))
        cellLocation = np.concatenate(
                (newCellLocation3[0:-1],
                newCellLocation3[-1] + newCellLocation1[0:-1], 
                newCellLocation3[-1] + newCellLocation1[-1] + newCellLocation2), axis=0)
        node = np.append(node, cutNode, axis=0)
        pmesh1 = PolygonMesh(node, cell, cellLocation)
        return pmesh1

class AdaptiveMarker3d():
    def __init__(self, phi, maxh=0.01, maxa=5):
        self.phi = phi
        self.maxh = maxh
        self.maxa = 5

    def interface_cell_flag(self, pmesh):
        isInterfaceFace = self.interface_face_flag(pmesh)
        face2cell = pmesh.ds.face2cell
        NC = pmesh.number_of_cells()
        isInterfaceCell = np.zeros(NC, dtype=np.bool_)
        isInterfaceCell[face2cell[isInterfaceFace, 0:2]] = True
        return isInterfaceCell 

    def interface_face_flag(self, pmesh):
        phi = self.phi
        face2node = pmesh.ds.face_to_node()
        phiValue = phi(pmesh.node)
        phiSign = msign(phiValue)
        NFV = pmesh.ds.number_of_vertices_of_faces()

        eta1 = np.abs(face2node*phiSign)
        eta2 = face2node*np.abs(phiSign)

        isInterfaceFace = (eta1 < eta2) | ((eta1 == eta2) & ((NV - eta2) > 2))

        return isInterfaceFace 


    def refine_mark(self, treemesh):
        pmesh = treemesh.to_pmesh() 

        N  = pmesh.number_of_nodes()
        NF = pmesh.number_of_faces()
        NC = pmesh.number_of_cells()

        isInterfaceCell = self.interface_cell_flag(self, pmesh)
        cell2node = pmesh.ds.cell_to_node()
        isInterfaceNode= np.asarray(isInterfaceCell@cell2node).reshape(-1)

        isMarkedCell = np.zeros(NC, dtype=np.bool_)

        # Case 1
        isLeafCell = treemesh.is_leaf_cell()
        node = treemesh.node
        cell = treemesh.ds.cell

        h = np.sqrt(np.sum((node[cell[isLeafCell, 0]] - node[cell[isLeafCell, 6]])**2, axis=1))
        maxh = self.maxh
        isMarkedCell = isMarkedCell | ((h > maxh) & isInterfaceCell) 

        return isMarkedCell 

    def coarsen_mark(self, pmesh):
        pass

class OctreeInterfaceMesh3d():
    def __init__(self, mesh, marker):
        pass

