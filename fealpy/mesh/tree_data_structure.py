import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .QuadrangleMesh import QuadrangleMesh 
from .HexahedronMesh import HexahedronMesh 
from .PolygonMesh import PolygonMesh
from .PolyhedronMesh import PolyhedronMesh 
from ..common import ranges

class Quadtree(QuadrangleMesh):
    localEdge2childCell = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.int)

    def __init__(self, node, cell, dtype=np.float):
        super(Quadtree, self).__init__(node, cell, dtype=dtype)
        self.dtype = dtype
        NC = self.number_of_cells()
        self.parent = -np.ones((NC, 2), dtype=np.int) 
        self.child = -np.ones((NC, 4), dtype=np.int)
        self.meshType = 'quadtree'

    def leaf_cell_index(self):
        child = self.child
        idx, = np.nonzero(child[:, 0] == -1)
        return idx

    def leaf_cell(self, celltype='quad'):
        child = self.child
        cell = self.ds.cell[child[:, 0] == -1]
        if celltype is 'quad':
            return cell
        elif celltype is 'tri':
            return np.r_['0', cell[:, [1, 2, 0]], cell[:, [3, 0, 2]]]

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
        pass 

    def refine(self, marker=None, u=None):
        if marker == None:
            idx = self.leaf_cell_index()
        else:
            idx = marker.refine_marker(self)

        if idx is None:
            return False

        if len(idx) > 0:
            # Prepare data
            N = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            node = self.node
            edge = self.ds.edge
            cell = self.ds.cell


            parent = self.parent
            child = self.child

            isLeafCell = self.is_leaf_cell()

            # Construct 
            isNeedCutCell = np.zeros(NC, dtype=np.bool_)
            isNeedCutCell[idx] = True
            isNeedCutCell = isNeedCutCell & isLeafCell

            # Find the cutted edge  
            cell2edge = self.ds.cell_to_edge()

            isCutEdge = np.zeros(NE, dtype=np.bool_)
            isCutEdge[cell2edge[isNeedCutCell, :]] = True

            isCuttedEdge = np.zeros(NE, dtype=np.bool_)
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

            edgeCenter = 0.5*np.sum(node[edge[isNeedCutEdge]], axis=1) 
            cellCenter = self.entity_barycenter('cell', isNeedCutCell)
            if u is not None:
                isNeedCutEdge = (~isCuttedEdge) & isCutEdge 
                eu = 0.5*np.sum(u[edge[isNeedCutEdge]], axis=1)
                
                cu = np.sum(u[cell[isNeedCutCell], :], axis=1)/4
                Iu = np.concatenate((u, eu, cu), axis=0)
            
                

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
            self.node = np.concatenate((node, edgeCenter, cellCenter), axis=0)
            self.parent = np.concatenate((parent, newParent), axis=0)
            self.child = np.concatenate((child, newChild), axis=0)
            self.ds.reinit(N + NEC + NCC, cell)
            if u is None:
                return True
            else:
                return (Iu,True)
        else:
            return False
        


    def coarsen(self, marker):
        """ marker will marke the leaf cells which will be coarsen
        """
        idx = marker.coarsen_marker(self)
        if idx is None:
            return False

        if len(idx) > 0:
            N = self.number_of_nodes()
            NC = self.number_of_cells()

            node = self.node
            cell = self.ds.cell

            parent = self.parent
            child = self.child

            isRemainCell = np.ones(NC, dtype=np.bool_)
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

            isRemainNode = np.zeros(N, dtype=np.bool_)
            isRemainNode[cell[isRemainCell, :]] = True

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

            nodeIdxMap = np.zeros(N, dtype=np.int)
            NN = isRemainNode.sum()
            nodeIdxMap[isRemainNode] = np.arange(NN)
            cell = nodeIdxMap[cell]
            self.node = node[isRemainNode]
            self.ds.reinit(NN, cell)

            if cell.shape[0] == NC:
                return False 
            else:
                return True
        else:
            return False

        """ Transform the quadtree data structure to polygonmesh datastructure
        """

        isRootCell = self.is_root_cell()

        if np.all(isRootCell):
            NC = self.number_of_cells()

            node = self.node
            cell = self.ds.cell
            NV = cell.shape[1]

            pcell = cell.reshape(-1) 
            pcellLocation = np.arange(0, NV*(NC+1), NV)

            return PolygonMesh(node, pcell, pcellLocation, dtype=self.dtype) 
        else:
            N = self.number_of_nodes()
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
            val = np.ones(PNE, dtype=np.bool_)
            p2pe = coo_matrix(
                    (val, (pedge[:,0], range(PNE))),
                    shape=(N, PNE), dtype=np.bool_)
            p2pe += coo_matrix(
                    (val, (pedge[:,1], range(PNE))),
                    shape=(N, PNE), dtype=np.bool_)
            p2pe = p2pe.tocsr()
            NES = np.asarray(p2pe.sum(axis=1)).reshape(-1) 
            isPast = np.zeros(PNE, dtype=np.bool_)
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
                    nodeIdx = pcell[currentIdx] 
                    _, J = p2pe[nodeIdx].nonzero()
                    isEdge = (pedge2cell[J, 1] == np.repeat(cellIdx, NES[nodeIdx])) \
                            & (pedge2cell[J, 3] == i) & (~isPast[J])
                    isPast[J[isEdge]] = True
                    pcell[currentIdx + 1] = pedge[J[isEdge], 0]
                    currentIdx += 1

            return PolygonMesh(self.node,  pcell, pcellLocation)

    def bc_to_point(self, bc):
        node = self.node
        cell = self.ds.cell
        isLeafCell = self.is_leaf_cell()
        p = np.einsum('...j, ijk->...ik', bc, node[cell[isLeafCell]])
        return p 


class Octree(HexahedronMesh):
    localFace2childCell = np.array([
        (0, 2), (4, 6), 
        (0, 7), (1, 6),
        (0, 5), (2, 7)], dtype=np.int)
    localEdge2childCell = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 0), (5, 1), (6, 2), (7, 3),
        (4, 5), (5, 6), (6, 7), (7, 4)], dtype=np.int)

    def __init__(self, node, cell, dtype=np.float):
        super(Octree, self).__init__(node, cell, dtype=dtype)
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
            N = self.number_of_nodes()
            NE = self.number_of_edges()
            NF = self.number_of_faces()
            NC = self.number_of_cells()

            node = self.node
            edge = self.ds.edge
            face = self.ds.face
            cell = self.ds.cell

            parent = self.parent
            child = self.child

            isLeafCell = self.is_leaf_cell()

            # Construct cellCenter
            isNeedCutCell = np.zeros(NC, dtype=np.bool_)
            isNeedCutCell[idx] = True
            isNeedCutCell = isNeedCutCell & isLeafCell

            # Construct edgeCenter 
            cell2edge = self.ds.cell_to_edge()

            isCutEdge = np.zeros(NE, dtype=np.bool_)
            isCutEdge[cell2edge[isNeedCutCell, :]] = True

            isCuttedEdge = np.zeros(NE, dtype=np.bool_)
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

            edgeCenter = 0.5*np.sum(node[edge[isNeedCutEdge]], axis=1)
            NEC = len(edgeCenter)
            edge2center[isNeedCutEdge] = np.arange(N, N + NEC)

            # Construct faceCenter and face2center
            cell2face = self.ds.cell_to_face()
            isCutFace = np.zeros(NF, dtype=np.bool_)
            isCutFace[cell2face[isNeedCutCell, :]] = True

            isCuttedFace = np.zeros(NF, dtype=np.bool_)
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

            faceCenter = 0.5*np.sum(node[face[isNeedCutFace][:, [0, 2]]], axis=1)
            NFC = len(faceCenter)

            face2center[isNeedCutFace] = np.arange(N + NEC, N + NEC + NFC)


            cellCenter = 0.5*np.sum(node[cell[isNeedCutCell][:, [0, 6]]], axis=1)
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
            self.node = np.concatenate((node, edgeCenter, faceCenter, cellCenter), axis=0)
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
            N = self.number_of_nodes()
            NC = self.number_of_cells()

            node = self.node
            cell = self.ds.cell
            parent = self.parent
            child = self.child

            
            isRemainCell = np.ones(NC, dtype=np.bool_)
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

            isRemainNode = np.zeros(N, dtype=np.bool_)
            isRemainNode[cell[isRemainCell, :]] = True

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

            nodeIdxMap = np.zeros(N, dtype=np.int)
            NN = isRemainNode.sum()
            nodeIdxMap[isRemainNode] = np.arange(NN)
            cell = nodeIdxMap[cell]
            self.node = node[isRemainNode]
            self.ds.reinit(NN, cell)

            if cell.shape[0] == NC:
                return False 
            else:
                return True
        else:
            return False

    def to_pmesh1(self):
        NF = self.number_of_faces()
        NC = self.number_of_cells()
        node = self.node
        face = self.ds.face
        NV = face.shape[1]
        face2cell = self.ds.face2cell
        faceLocation = np.arange(0, NV*(NF+1), NV)
        return PolyhedronMesh(node, face.reshape(-1), faceLocation, face2cell, NC=NC)

    def to_pmesh(self):
        '''transform the Octree data structure to Polyhedral data structure

            * node, pface, pfaceLocation, pface2cell 
        '''
        isRootCell = self.is_root_cell()
        if np.all(isRootCell):
            NF = self.number_of_faces()
            NC = self.number_of_cells()
            node = self.node
            face = self.ds.face
            NV = face.shape[1]
            face2cell = self.ds.face2cell
            faceLocation = np.arange(0, NV*(NF+1), NV)
            return PolyhedronMesh(node, face.reshape(-1), faceLocation, face2cell, NC=NC)
        else:
            face2cell = self.ds.face2cell
            cell2cell = self.ds.cell_to_cell()
            cell2face = self.ds.cell_to_face()
            face = self.ds.face
            parent = self.parent
            child = self.child

            isLeafCell = self.is_leaf_cell()
            isLeafFace = (isLeafCell[face2cell[:, 0]] & isLeafCell[face2cell[:, 1]])

            pface2cell = face2cell[isLeafFace]
            pface = face[isLeafFace]

            isLeafBdFace = (pface2cell[:, 0] == pface2cell[:, 1])
            pfaceIdx, = np.nonzero(isLeafBdFace)
            while len(pfaceIdx) > 0:
                # the right hand side cell of current face 
                cellIdx = pface2cell[pfaceIdx, 1]  
                localIdx = pface2cell[pfaceIdx, 3]

                parentCellIdx = parent[cellIdx, 0] 
                # the parent cell's 'localIdx'-th neighbor
                neighborCellIdx = cell2cell[parentCellIdx, localIdx]
                
                # if this neighbor cell is a leaf cell or a root cell, it is
                # what we want 
                isFound = isLeafCell[neighborCellIdx] | isRootCell[neighborCellIdx]
                pface2cell[pfaceIdx[isFound], 1] = neighborCellIdx[isFound]

                # find the global face idx of this parent cell's  `localIdx`-th face 
                faceIdx = cell2face[parentCellIdx, localIdx]

                # if the left cell of `faceIdx`-th face is not this parent
                # cell, we update the local idx of 
                isCase = (face2cell[faceIdx, 0] == neighborCellIdx) & isFound
                pface2cell[pfaceIdx[isCase], 3] = face2cell[faceIdx[isCase], 2] 

                # If the left cell of `faceIdx`-th face is this parent cell, we
                # set the local idx in its right cell of this `pfaceidx`
                isCase = (face2cell[faceIdx, 1] == neighborCellIdx) & isFound
                pface2cell[pfaceIdx[isCase], 3] = face2cell[faceIdx[isCase], 3] 

                # If this parent cell is a boundary cell
                isSpecial = isFound & (parentCellIdx == neighborCellIdx) 
                pface2cell[pfaceIdx[isSpecial], 1] =  pface2cell[pfaceIdx[isSpecial], 0]
                pface2cell[pfaceIdx[isSpecial], 3] =  pface2cell[pfaceIdx[isSpecial], 2]

                pfaceIdx = pfaceIdx[~isFound]
                pface2cell[pfaceIdx, 1] = parentCellIdx[~isFound]

            NC = self.number_of_cells()
            PNC = isLeafCell.sum()
            cellIdxMap = np.zeros(NC, dtype=np.int)
            cellIdxMap[isLeafCell] = np.arange(PNC)
            pface2cell[:, 0:2] = cellIdxMap[pface2cell[:, 0:2]]

            # pface and pfaceLocation
            edge = self.ds.edge
            N = self.number_of_nodes()
            NE = self.number_of_edges()
            face2edge = self.ds.face_to_edge()
            isLeafEdge = np.zeros(NE, dtype=np.bool_)
            pface2edge = face2edge[isLeafFace]
            isLeafEdge[pface2edge] = True
            pedge = edge[isLeafEdge]
            idxMap = -np.ones(NE, dtype=np.int)
            idxMap[isLeafEdge] = range(np.sum(isLeafEdge))
            pface2edge = idxMap[pface2edge]

            PNF = pface.shape[0]
            NE = pedge.shape[0]
            I = pedge.flatten()
            J = np.repeat(range(NE), 2)
            val = np.ones((NE, 2), dtype=np.int8)
            val[:, 1] = 2
            p2e = csr_matrix((val.flatten(), (I, J)), shape=(N, NE), dtype=np.int8)

            NV = np.zeros(PNF, dtype=np.int)
            node = self.node
            mp = (node[pedge[:, 0]] + node[pedge[:, 1]])/2
            l2 = np.sqrt(np.sum((node[pedge[:, 0]] - node[pedge[:, 1]])**2, axis=1))
            for i in range(4):
                print('Current:', i)
                NV += 1
                fidx = np.arange(PNF)
                p0 = pface[:, i]
                p1 = pface[:, (i+1)%4]
                ei = pface2edge[:, i]
                pe = -np.ones(PNF) 
                fp = pface[:, i] 
                k = 0
                while True:
                    k += 1
                    p2e0 = p2e[fp, :]
                    I, J = p2e0.nonzero()
                    lidx = np.asarray(p2e0[I, J]).reshape(-1)

                    l20 = np.sqrt(np.sum((mp[J] - node[p0[I]])**2, axis=1))
                    l21 = np.sqrt(np.sum((mp[J] - node[p1[I]])**2, axis=1))
                    flag0 = (np.abs(l20 + l21 - l2[ei[I]]) < 1e-12)    
                    flag1 = (pedge[J, lidx%2] != p1[I]) 
                    flag2 = l2[J] < l2[ei[I]]
                    ldot = np.einsum('ij, ij->i', 
                            node[pedge[J, lidx%2]] - node[pedge[J, lidx-1]],
                            node[p1[I]] - node[p0[I]])
                    flag3 = (np.abs(ldot - l2[J]*l2[ei[I]]) < 1e-12)
                    isNotOK = flag0 & flag1 & flag2 & flag3
                    if isNotOK.sum() == 0:
                        break

                    I = I[isNotOK]
                    J = J[isNotOK]

                    p2e1 = csr_matrix((1/l2[J], (I, J)), shape=(len(fp), NE))
                    J = np.asarray(p2e1.argmax(axis=1)).reshape(-1)
                    I = np.arange(len(fp))
                    isNotOK = np.asarray(p2e1[I, J]).reshape(-1) > 0
                    if isNotOK.sum() == 0:
                        break

                    lidx = np.asarray(p2e0[I[isNotOK], J[isNotOK]]).reshape(-1)
                    fidx = fidx[I[isNotOK]]
                    p0 = pface[fidx, i] 
                    p1 = pface[fidx, (i+1)%4] 
                    ei = pface2edge[fidx, i]
                    pe = J[isNotOK]
                    fp = pedge[pe, lidx%2]
                    NV[fidx] += 1

            pfaceLocation = np.zeros(PNF+1, dtype=np.int)
            pfaceLocation[1:] = np.cumsum(NV)
            pface0 = np.zeros(pfaceLocation[-1], dtype=np.int)
            currentLocation = pfaceLocation[:-1].copy()
            for i in range(4):
                print("Current ", i)
                pface0[currentLocation] = pface[:, i]
                currentLocation += 1
                fidx = np.arange(PNF)
                p0 = pface[:, i]
                p1 = pface[:, (i+1)%4]
                ei = pface2edge[:, i]
                pe = -np.ones(PNF) 
                fp = p0
                while True:
                    p2e0 = p2e[fp]
                    I, J = p2e0.nonzero()
                    lidx = np.asarray(p2e0[I, J]).reshape(-1)
                    l20 = np.sqrt(np.sum((mp[J] - node[p0[I]])**2, axis=1))
                    l21 = np.sqrt(np.sum((mp[J] - node[p1[I]])**2, axis=1))
                    flag0 = (np.abs(l20 + l21 - l2[ei[I]]) < 1e-12) & (J != pe[I])   
                    flag1 = (pedge[J, lidx%2] != p1[I]) 
                    flag2 = l2[J] < l2[ei[I]]
                    ldot = np.einsum('ij, ij->i', 
                            node[pedge[J, lidx%2]] - node[pedge[J, lidx-1]],
                            node[p1[I]] - node[p0[I]])
                    flag3 = (np.abs(ldot - l2[J]*l2[ei[I]]) < 1e-12)
                    isNotOK = flag0 & flag1 & flag2 & flag3
                    if isNotOK.sum() == 0:
                        break

                    I = I[isNotOK]
                    J = J[isNotOK]
                    lidx = lidx[isNotOK]
                    p2e1 = csr_matrix((1/l2[J], (I, J)), shape=(len(fp), NE))
                    J = np.asarray(p2e1.argmax(axis=1)).reshape(-1)
                    I = np.arange(len(fp))
                    isNotOK = np.asarray(p2e1[I, J]).reshape(-1) > 0
                    if isNotOK.sum() == 0:
                        break

                    lidx = np.asarray(p2e0[I[isNotOK], J[isNotOK]]).reshape(-1)
                    fidx = fidx[I[isNotOK]]
                    p0 = pface[fidx, i] 
                    p1 = pface[fidx, (i+1)%4] 
                    ei = pface2edge[fidx, i]
                    pe = J[isNotOK]
                    fp = pedge[pe, lidx%2]
                    pface0[currentLocation[fidx]] = fp
                    currentLocation[fidx] += 1

            return PolyhedronMesh(node, pface0, pfaceLocation, pface2cell, NC=PNC)
