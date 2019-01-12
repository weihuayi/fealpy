
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .QuadrangleMesh import QuadrangleMesh 
from .PolygonMesh import PolygonMesh
from ..common import ranges

class Quadtree(QuadrangleMesh):
    localEdge2childCell = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.int)

    def __init__(self, node, cell):
        super(Quadtree, self).__init__(node, cell)
        NC = self.number_of_cells()
        self.parent = -np.ones((NC, 2), dtype=self.itype) 
        self.child = -np.ones((NC, 4), dtype=self.itype)
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
            cellIdx = np.zeros(NE, dtype=self.itype)
            localIdx = np.zeros(NE, dtype=self.itype)
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

            edge2center = np.zeros(NE, dtype=self.itype)
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
            
            newCell = np.zeros((4*NCC, 4), dtype=self.itype)
            newChild = -np.ones((4*NCC, 4), dtype=self.itype)
            newParent = -np.ones((4*NCC, 2), dtype=self.itype)
            newCell[0::4, :] = np.concatenate((cp[0], ep[0], cc, ep[3]), axis=1) 
            newCell[1::4, :] = np.concatenate((ep[0], cp[1], ep[1], cc), axis=1)
            newCell[2::4, :] = np.concatenate((cc, ep[1], cp[2], ep[2]), axis=1)
            newCell[3::4, :] = np.concatenate((ep[3], cc, ep[2], cp[3]), axis=1)
            newParent[:, 0] = np.repeat(idx, 4)
            newParent[:, 1] = ranges(4*np.ones(NCC, dtype=self.itype)) 
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
            NN = self.number_of_nodes()
            NC = self.number_of_cells()

            node = self.node
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

            isRemainNode = np.zeros(N, dtype=np.bool)
            isRemainNode[cell[isRemainCell, :]] = True

            cell = cell[isRemainCell]
            child = child[isRemainCell]
            parent = parent[isRemainCell]

            # 子单元不需要保留的单元， 是新的叶子单元

            childIdx, = np.nonzero(child[:, 0] > -1)
            isNewLeafCell = np.sum(isRemainCell[child[childIdx, :]], axis=1) == 0 
            child[childIdx[isNewLeafCell], :] = -1

            cellIdxMap = np.zeros(NC, dtype=self.itype)
            NNC = isRemainCell.sum()
            cellIdxMap[isRemainCell] = np.arange(NNC)
            child[child > -1] = cellIdxMap[child[child > -1]]
            parent[parent > -1] = cellIdxMap[parent[parent > -1]]
            self.child = child
            self.parent = parent

            nodeIdxMap = np.zeros(NN, dtype=self.itype)
            N = isRemainNode.sum()
            nodeIdxMap[isRemainNode] = np.arange(N)
            cell = nodeIdxMap[cell]
            self.node = node[isRemainNode]
            self.ds.reinit(N, cell)

            if cell.shape[0] == NC:
                return False 
            else:
                return True
        else:
            return False

    def to_pmesh(self):
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

            return PolygonMesh(node, pcell, pcellLocation) 
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
            cellIdxMap = np.zeros(NC, dtype=self.itype)
            cellIdxMap[isLeafCell] = np.arange(PNC)
            cellIdxInvMap, = np.nonzero(isLeafCell)

            pedge2cell[:, 0:2] = cellIdxMap[pedge2cell[:, 0:2]]

            # 计算每个叶子四边形单元的每条边上有几条叶子边
            # 因为叶子单元的边不一定是叶子边
            isInPEdge = (pedge2cell[:, 0] != pedge2cell[:, 1])
            cornerLocation = np.zeros((PNC, 5), dtype=self.itype)
            np.add.at(cornerLocation.ravel(), 5*pedge2cell[:, 0] + pedge2cell[:, 2] + 1, 1)
            np.add.at(cornerLocation.ravel(), 5*pedge2cell[isInPEdge, 1] + pedge2cell[isInPEdge, 3] + 1, 1)
            cornerLocation = cornerLocation.cumsum(axis=1)


            pcellLocation = np.zeros(PNC+1, dtype=self.itype)
            pcellLocation[1:] = cornerLocation[:, 4].cumsum()
            pcell = np.zeros(pcellLocation[-1], dtype=self.itype)
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
