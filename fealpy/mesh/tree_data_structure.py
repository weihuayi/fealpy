import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .QuadrangleMesh import QuadrangleMesh 
from .HexahedronMesh import HexahedronMesh 
from .PolygonMesh import PolygonMesh
from ..common import ranges

class Quadtree(QuadrangleMesh):

    localEdge2childCell = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.int)

    def __init__(self, point, cell, dtype=np.float):
        super(Quadtree, self).__init__(point, cell, dtype=dtype)
        self.dtype = dtype
        NC = self.number_of_cells()
        self.parent = -np.ones((NC, 2), dtype=np.int) 
        self.child = -np.ones((NC, 4), dtype=np.int)

    def leaf_cell_index(self):
        child = self.child
        idx, = np.nonzero(child[:, 0] == -1)
        return idx

    def is_leaf_cell(self, idx=None):
        if idx is None:
            return self.child[:, 0] == -1
        else:
            return self.child[idx, 0] == -1

    def is_root_cell(self, idx=None):
        if idx is None:
            return self.parent[:, 0] == -1
        else:
            return self.parent[idx, 0] == -1
    
    def uniform_refine(self, r=1):
        for i in range(r):
            self.refine()

    def sizing_adaptive(self, eta):
        ```
        ```
        pass 


    def refine(self, marker=None):
        if marker == None:
            idx = self.leaf_cell_index()
        else:
            idx = marker.refine_marker(self)

        if idx is None:
            return False

        if len(idx) > 0:
            # Prepare data
            N = self.number_of_points()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            point = self.point
            edge = self.ds.edge
            cell = self.ds.cell


            parent = self.parent
            child = self.child

            isLeafCell = self.is_leaf_cell()

            # Construct 
            isNeedCutCell = np.zeros(NC, dtype=np.bool)
            isNeedCutCell[idx] = True
            isNeedCutCell = isNeedCutCell & isLeafCell

            # Find the cutted edge  
            cell2edge = self.ds.cell_to_edge()

            isCutEdge = np.zeros(NE, dtype=np.bool)
            isCutEdge[cell2edge[isNeedCutCell, :]] = True

            isCuttedEdge = np.zeros(NE, dtype=np.bool)
            isCuttedEdge[cell2edge[~isLeafCell, :]] = True
            isCuttedEdge = isCuttedEdge & isCutEdge

            isNeedCutEdge = (~isCuttedEdge) & isCutEdge 

            # 找到每条非叶子边对应的单元编号， 及在该单元中的局部编号 
            I, J = np.nonzero(isCuttedEdge[cell2edge])
            cellIdx = np.zeros(NE, dtype=np.int)
            localIdx = np.zeros(NE, dtype=np.int)
            I1 = I[~isLeafCell[I]]
            J1 = J[~isLeafCell[I]]
            cellIdx[cell2edge[I1, J1]] = I1 # the cell idx 
            localIdx[cell2edge[I1, J1]] = J1 #
            del I, J, I1, J1

            # 找到该单元相应孩子单元编号， 及对应的中点编号
            cellIdx = cellIdx[isCuttedEdge]
            localIdx = localIdx[isCuttedEdge]
            cellIdx = child[cellIdx, self.localEdge2childCell[localIdx, 0]]
            localIdx = self.localEdge2childCell[localIdx, 1]

            edge2center = np.zeros(NE, dtype=np.int)
            edge2center[isCuttedEdge] = cell[cellIdx, localIdx]  

            edgeCenter = 0.5*np.sum(point[edge[isNeedCutEdge]], axis=1) 
            cellCenter = 0.5*np.sum(point[cell[isNeedCutCell][:, [0, 2]].reshape(-1, 2)], axis=1)

            NEC = len(edgeCenter)
            NCC = len(cellCenter)

            edge2center[isNeedCutEdge] = np.arange(N, N+NEC) 

            cp = [cell[isNeedCutCell, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[isNeedCutCell, i]].reshape(-1, 1) for i in range(4)]
            cc = np.arange(N + NEC, N + NEC + NCC).reshape(-1, 1)
            
            newCell = np.zeros((4*NCC, 4), dtype=np.int)
            newChild = -np.ones((4*NCC, 4), dtype=np.int)
            newParent = -np.ones((4*NCC, 2), dtype=np.int)
            newCell[0::4, :] = np.concatenate((cp[0], ep[0], cc, ep[3]), axis=1) 
            newCell[1::4, :] = np.concatenate((ep[0], cp[1], ep[1], cc), axis=1)
            newCell[2::4, :] = np.concatenate((cc, ep[1], cp[2], ep[2]), axis=1)
            newCell[3::4, :] = np.concatenate((ep[3], cc, ep[2], cp[3]), axis=1)
            newParent[:, 0] = np.repeat(idx, 4)
            newParent[:, 1] = ranges(4*np.ones(NCC, dtype=np.int)) 
            child[idx, :] = np.arange(NC, NC + 4*NCC).reshape(NCC, 4)

            cell = np.concatenate((cell, newCell), axis=0)
            self.point = np.concatenate((point, edgeCenter, cellCenter), axis=0)
            self.parent = np.concatenate((parent, newParent), axis=0)
            self.child = np.concatenate((child, newChild), axis=0)
            self.ds.reinit(N + NEC + NCC, cell)
            return True
        else:
            return False

    def coarsen(self, marker):
        """ marker will marke the leaf cells which will be coarsen
        """
        idx = marker.coarsen_marker(self)
        if idx is None:
            return False

        if len(idx) > 0:
            N = self.number_of_points()
            NC = self.number_of_cells()

            point = self.point
            cell = self.ds.cell

            parent = self.parent
            child = self.child

            isRemainCell = np.ones(NC, dtype=np.bool)
            isRemainCell[child[idx, :]] = False

            isNotRootCell = (~self.is_root_cell())
            NNC0 = isRemainCell.sum()
            while True:
                isRemainCell[parent[isRemainCell & isNotRootCell, 0]] = True
                isRemainCell[child[parent[isRemainCell & isNotRootCell, 0], :]] = True
                NNC1 = isRemainCell.sum()
                if NNC1 == NNC0:
                    break
                else:
                    NNC0 = NNC1

            isRemainPoint = np.zeros(N, dtype=np.bool)
            isRemainPoint[cell[isRemainCell, :]] = True

            cell = cell[isRemainCell]
            child = child[isRemainCell]
            parent = parent[isRemainCell]

            # 子单元不需要保留的单元， 是新的叶子单元 
            childIdx, = np.nonzero(child[:, 0] > -1)
            isNewLeafCell = np.sum(isRemainCell[child[childIdx, :]], axis=1) == 0 
            child[childIdx[isNewLeafCell], :] = -1

            cellIdxMap = np.zeros(NC, dtype=np.int)
            NNC = isRemainCell.sum()
            cellIdxMap[isRemainCell] = np.arange(NNC)
            child[child > -1] = cellIdxMap[child[child > -1]]
            parent[parent > -1] = cellIdxMap[parent[parent > -1]]
            self.child = child
            self.parent = parent

            pointIdxMap = np.zeros(N, dtype=np.int)
            NN = isRemainPoint.sum()
            pointIdxMap[isRemainPoint] = np.arange(NN)
            cell = pointIdxMap[cell]
            self.point = point[isRemainPoint]
            self.ds.reinit(NN, cell)

            if cell.shape[0] == NC:
                return False 
            else:
                return True
        else:
            return False

    def to_polygonmesh(self):
        """ Transform the quadtree data structure to polygonmesh datastructure
        """

        isRootCell = self.is_root_cell()

        if np.all(isRootCell):
            NC = self.number_of_cells()

            point = self.point
            cell = self.ds.cell
            NV = cell.shape[1]

            pcell = cell.reshape(-1) 
            pcellLocation = np.arange(0, NV*NC, NV)

            return PolygonMesh(point, pcell, pcellLocation, dtype=self.dtype) 
        else:
            N = self.number_of_points()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            cell = self.ds.cell
            edge = self.ds.edge
            edge2cell = self.ds.edge_to_cell()
            cell2cell = self.ds.cell_to_cell()
            cell2edge = self.ds.cell_to_edge()

            parent = self.parent
            child = self.child


            isLeafCell = self.is_leaf_cell()
            isLeafEdge = isLeafCell[edge2cell[:, 0]] & isLeafCell[edge2cell[:, 1]]

            pedge2cell = edge2cell[isLeafEdge, :]
            pedge = edge[isLeafEdge, :]

            isRootCell = self.is_root_cell()
            isLevelBdEdge =  (pedge2cell[:, 0] == pedge2cell[:, 1]) 

            # Find the index of all boundary edges on each tree level
            pedgeIdx, = np.nonzero(isLevelBdEdge)
            while len(pedgeIdx) > 0:
                cellIdx = pedge2cell[pedgeIdx, 1] 
                localIdx = pedge2cell[pedgeIdx, 3]

                parentCellIdx = parent[cellIdx, 0] 
                
                neighborCellIdx = cell2cell[parentCellIdx, localIdx]
                
                isFound = isLeafCell[neighborCellIdx] | isRootCell[neighborCellIdx]
                pedge2cell[pedgeIdx[isFound], 1] = neighborCellIdx[isFound]

                edgeIdx = cell2edge[parentCellIdx, localIdx]

                isCase = (edge2cell[edgeIdx, 0] != parentCellIdx) & isFound
                pedge2cell[pedgeIdx[isCase], 3] = edge2cell[edgeIdx[isCase], 2] 

                isCase = (edge2cell[edgeIdx, 0] == parentCellIdx) & isFound
                pedge2cell[pedgeIdx[isCase], 3] = edge2cell[edgeIdx[isCase], 3] 

                isSpecial = isFound & (parentCellIdx == neighborCellIdx) 
                pedge2cell[pedgeIdx[isSpecial], 1] =  pedge2cell[pedgeIdx[isSpecial], 0]
                pedge2cell[pedgeIdx[isSpecial], 3] =  pedge2cell[pedgeIdx[isSpecial], 2]

                pedgeIdx = pedgeIdx[~isFound]
                pedge2cell[pedgeIdx, 1] = parentCellIdx[~isFound]


            PNC = isLeafCell.sum()
            cellIdxMap = np.zeros(NC, dtype=np.int)
            cellIdxMap[isLeafCell] = np.arange(PNC)
            cellIdxInvMap, = np.nonzero(isLeafCell)

            pedge2cell[:, 0:2] = cellIdxMap[pedge2cell[:, 0:2]]

            # 计算每个叶子四边形单元的每条边上有几条叶子边
            # 因为叶子单元的边不一定是叶子边
            isInPEdge = (pedge2cell[:, 0] != pedge2cell[:, 1])
            cornerLocation = np.zeros((PNC, 5), dtype=np.int)
            np.add.at(cornerLocation.ravel(), 5*pedge2cell[:, 0] + pedge2cell[:, 2] + 1, 1)
            np.add.at(cornerLocation.ravel(), 5*pedge2cell[isInPEdge, 1] + pedge2cell[isInPEdge, 3] + 1, 1)
            cornerLocation = cornerLocation.cumsum(axis=1)


            pcellLocation = np.zeros(PNC+1, dtype=np.int)
            pcellLocation[1:] = cornerLocation[:, 4].cumsum()
            pcell = np.zeros(pcellLocation[-1], dtype=np.int)
            cornerLocation += pcellLocation[:-1].reshape(-1, 1) 
            pcell[cornerLocation[:, 0:-1]] = cell[isLeafCell, :]

            PNE = pedge.shape[0]
            val = np.ones(PNE, dtype=np.bool)
            p2pe = coo_matrix(
                    (val, (pedge[:,0], range(PNE))),
                    shape=(N, PNE), dtype=np.bool)
            p2pe += coo_matrix(
                    (val, (pedge[:,1], range(PNE))),
                    shape=(N, PNE), dtype=np.bool)
            p2pe = p2pe.tocsr()
            NES = np.asarray(p2pe.sum(axis=1)).reshape(-1) 
            isPast = np.zeros(PNE, dtype=np.bool)
            for i in range(4):
                currentIdx = cornerLocation[:, i]
                endIdx = cornerLocation[:, i+1]
                cellIdx = np.arange(PNC)
                while True:
                    isNotOK = ((currentIdx + 1) < endIdx)
                    currentIdx = currentIdx[isNotOK]
                    endIdx = endIdx[isNotOK]
                    cellIdx = cellIdx[isNotOK]
                    if len(currentIdx) == 0:
                        break
                    pointIdx = pcell[currentIdx] 
                    _, J = p2pe[pointIdx].nonzero()
                    isEdge = (pedge2cell[J, 1] == np.repeat(cellIdx, NES[pointIdx])) \
                            & (pedge2cell[J, 3] == i) & (~isPast[J])
                    isPast[J[isEdge]] = True
                    pcell[currentIdx + 1] = pedge[J[isEdge], 0]
                    currentIdx += 1

            return PolygonMesh(self.point,  pcell, pcellLocation)


class Octree(HexahedronMesh):
    localFace2childCell = np.array([
        (0, 2), (4, 6), 
        (0, 7), (1, 6),
        (0, 5), (2, 7)], dtype=np.int)
    localEdge2childCell = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 0), (5, 1), (6, 2), (7, 3),
        (4, 5), (5, 6), (6, 7), (7, 4)], dtype=np.int)

    def __init__(self, point, cell, dtype=np.float):
        super(Octree, self).__init__(point, cell, dtype=dtype)
        self.dtype = dtype
        NC = self.number_of_cells()
        self.parent = -np.ones((NC, 2), dtype=np.int) 
        self.child = -np.ones((NC, 8), dtype=np.int)

    def leaf_cell_index(self):
        child = self.child
        idx, = np.nonzero(child[:, 0] == -1)
        return idx

    def is_leaf_cell(self, idx=None):
        if idx is None:
            return self.child[:, 0] == -1
        else:
            return self.child[idx, 0] == -1

    def is_root_cell(self, idx=None):
        if idx is None:
            return self.parent[:, 0] == -1
        else:
            return self.parent[idx, 0] == -1
    
    def uniform_refine(self):
        self.refine()

    def refine(self, marker=None):
        if marker == None:
            idx = self.leaf_cell_index()
        else:
            idx = marker.refine_marker(self)

        if idx is None:
            return False

        if len(idx) > 0:
            N = self.number_of_points()
            NE = self.number_of_edges()
            NF = self.number_of_faces()
            NC = self.number_of_cells()

            point = self.point
            edge = self.ds.edge
            face = self.ds.face
            cell = self.ds.cell

            parent = self.parent
            child = self.child

            isLeafCell = self.is_leaf_cell()

            # Construct cellCenter
            isNeedCutCell = np.zeros(NC, dtype=np.bool)
            isNeedCutCell[idx] = True
            isNeedCutCell = isNeedCutCell & isLeafCell

            # Construct edgeCenter 
            cell2edge = self.ds.cell_to_edge()

            isCutEdge = np.zeros(NE, dtype=np.bool)
            isCutEdge[cell2edge[isNeedCutCell, :]] = True

            isCuttedEdge = np.zeros(NE, dtype=np.bool)
            isCuttedEdge[cell2edge[~isLeafCell, :]] = True
            isCuttedEdge = isCuttedEdge & isCutEdge

            isNeedCutEdge = ~isCuttedEdge & isCutEdge

            edge2center = np.zeros(NE, dtype=np.int)

            I, J = np.nonzero(isCuttedEdge[cell2edge])
            cellIdx = np.zeros(NE, dtype=np.int)
            localIdx = np.zeros(NE, dtype=np.int)
            I1 = I[~isLeafCell[I]]
            J1 = J[~isLeafCell[I]]
            cellIdx[cell2edge[I1, J1]] = I1
            localIdx[cell2edge[I1, J1]] = J1
            del I, J, I1, J1
            cellIdx = cellIdx[isCuttedEdge]
            localIdx = localIdx[isCuttedEdge]
            cellIdx = child[cellIdx, self.localEdge2childCell[localIdx, 0]]
            localIdx = self.localEdge2childCell[localIdx, 1]
            edge2center[isCuttedEdge] = cell[cellIdx, localIdx]  

            edgeCenter = 0.5*np.sum(point[edge[isNeedCutEdge]], axis=1)
            NEC = len(edgeCenter)
            edge2center[isNeedCutEdge] = np.arange(N, N + NEC)

            # Construct faceCenter and face2center
            cell2face = self.ds.cell_to_face()
            isCutFace = np.zeros(NF, dtype=np.bool)
            isCutFace[cell2face[isNeedCutCell, :]] = True

            isCuttedFace = np.zeros(NF, dtype=np.bool)
            isCuttedFace[cell2face[~isLeafCell, :]] = True 
            isCuttedFace = isCuttedFace & isCutFace

            isNeedCutFace = ~isCuttedFace & isCutFace 
            

            face2center = np.zeros(NF, dtype=np.int)

            I, J = np.nonzero(isCuttedFace[cell2face])
            cellIdx = np.zeros(NF, dtype=np.int)
            localIdx = np.zeros(NF, dtype=np.int)
            I1 = I[~isLeafCell[I]]
            J1 = J[~isLeafCell[I]]
            cellIdx[cell2face[I1, J1]] = I1
            localIdx[cell2face[I1, J1]] = J1
            del I, J, I1, J1
            cellIdx = cellIdx[isCuttedFace]
            localIdx = localIdx[isCuttedFace]
            cellIdx = child[cellIdx, self.localFace2childCell[localIdx, 0]]
            localIdx = self.localFace2childCell[localIdx, 1]
            face2center[isCuttedFace] = cell[cellIdx, localIdx]

            faceCenter = 0.5*np.sum(point[face[isNeedCutFace][:, [0, 2]]], axis=1)
            NFC = len(faceCenter)

            face2center[isNeedCutFace] = np.arange(N + NEC, N + NEC + NFC)


            cellCenter = 0.5*np.sum(point[cell[isNeedCutCell][:, [0, 6]]], axis=1)
            NCC = len(cellCenter) 

            cp = [cell[isNeedCutCell, i].reshape(-1, 1) for i in range(8)]

            ep = [edge2center[cell2edge[isNeedCutCell, i]].reshape(-1, 1) for i in range(12)]

            fp = [face2center[cell2face[isNeedCutCell, i]].reshape(-1, 1) for i in range(6)]

            cc = np.arange(N+NEC+NFC, N+NEC+NFC+NCC).reshape(-1, 1)

            newParent = np.zeros((8*NCC, 2), dtype=np.int)
            newParent[:, 0] = np.repeat(idx, 8)
            newParent[:, 1] = ranges(8*np.ones(NCC, dtype=np.int)) 
            newChild = -np.ones((8*NCC, 8), dtype=np.int)

            newCell = np.zeros((8*NCC, 8), dtype=np.int)
            newCell[0::8, :] = np.concatenate(
                    (cp[0], ep[0], fp[0], ep[3], ep[4], fp[4], cc, fp[2]), axis=1)
            newCell[1::8, :] = np.concatenate(
                    (ep[0], cp[1], ep[1], fp[0], fp[4], ep[5], fp[3], cc), axis=1)
            newCell[2::8, :] = np.concatenate(
                    (fp[0], ep[1], cp[2], ep[2], cc, fp[3], ep[6], fp[5]), axis=1)
            newCell[3::8, :] = np.concatenate(
                    (ep[3], fp[0], ep[2], cp[3], fp[2], cc, fp[5], ep[7]), axis=1)
            newCell[4::8, :] = np.concatenate(
                    (ep[4], fp[4], cc, fp[2], cp[4], ep[8], fp[1], ep[11]), axis=1)
            newCell[5::8, :] = np.concatenate(
                    (fp[4], ep[5], fp[3], cc, ep[8], cp[5], ep[9], fp[1]), axis=1)
            newCell[6::8, :] = np.concatenate(
                    (cc, fp[3], ep[6], fp[5], fp[1], ep[9], cp[6], ep[10]), axis=1)
            newCell[7::8, :] = np.concatenate(
                    (fp[2], cc, fp[5], ep[7], ep[11], fp[1], ep[10], cp[7]), axis=1)

            
            cell = np.concatenate((cell, newCell), axis=0)
            self.point = np.concatenate((point, edgeCenter, faceCenter, cellCenter), axis=0)
            self.parent = np.concatenate((parent, newParent), axis=0)
            self.child = np.concatenate((child, newChild), axis=0)
            self.child[newParent[:, 0], newParent[:, 1]] = np.arange(NC, NC + 8*NCC) 
            self.ds.reinit(N + NEC + NCC, cell)

            return True
        else:
            return False

    def coarsen(self, marker):
        """ marker will mark the leaf cells which will be coarsen
        """
        idx = marker.coarsen_marker(self)
        if idx is None:
            return False

        if len(idx) > 0:
            N = self.number_of_points()
            NC = self.number_of_cells()

            point = self.point
            cell = self.ds.cell
            parent = self.parent
            child = self.child

            
            isRemainCell = np.ones(NC, dtype=np.bool)
            isRemainCell[idx] = False
            isRemainCell[child[parent[idx, 0], :]] = False

            isNotRootCell = (~self.is_root_cell())
            NNC0 = isRemainCell.sum()
            while True:
                isRemainCell[parent[isRemainCell & isNotRootCell, 0]] = True
                isRemainCell[child[parent[isRemainCell & isNotRootCell, 0], :]] = True
                NNC1 = isRemainCell.sum()
                if NNC1 == NNC0:
                    break
                else:
                    NNC0 = NNC1

            isRemainPoint = np.zeros(N, dtype=np.bool)
            isRemainPoint[cell[isRemainCell, :]] = True

            cell = cell[isRemainCell]
            child = child[isRemainCell]
            parent = parent[isRemainCell]
            childIdx, = np.nonzero(child[:, 0] > -1)
            isNewLeafCell = np.sum(isRemainCell[child[childIdx, :]], axis=1) == 0 
            child[childIdx[isNewLeafCell], :] = -1

            cellIdxMap = np.zeros(NC, dtype=np.int)
            NNC = isRemainCell.sum()
            cellIdxMap[isRemainCell] = np.arange(NNC)
            child[child > -1] = cellIdxMap[child[child > -1]]
            parent[parent > -1] = cellIdxMap[parent[parent > -1]]
            self.child = child
            self.parent = parent

            pointIdxMap = np.zeros(N, dtype=np.int)
            NN = isRemainPoint.sum()
            pointIdxMap[isRemainPoint] = np.arange(NN)
            cell = pointIdxMap[cell]
            self.point = point[isRemainPoint]
            self.ds.reinit(NN, cell)

            if cell.shape[0] == NC:
                return False 
            else:
                return True
        else:
            return False
