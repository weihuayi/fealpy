import numpy as np

from .TriangleMesh import TriangleMesh 
from .adaptive_tools import mark

class Tritree(TriangleMesh):
    localEdge2childCell = np.array([(1, 2), (2, 0), (0, 1)], dtype=np.int32)
    def __init__(self, node, cell, irule=1):
        super(Tritree, self).__init__(node, cell)
        NC = self.number_of_cells()
        self.parent = -np.ones((NC, 2), dtype=self.itype)
        self.child = -np.ones((NC, 4), dtype=self.itype)
        self.irule = irule              # irregular rule  
        self.meshtype = 'tritree'

    def leaf_cell_index(self):
        child = self.child
        idx, = np.nonzero(child[:, 0] == -1)
        return idx

    def leaf_cell(self):
        child = self.child
        cell = self.ds.cell[child[:, 0] == -1]
        return cell

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

    def adaptive(self, estimator, surface=None):
        while ~estimator.is_uniform():
            isMarkedCell = self.marker(self.eta, self.theta, 'MAX')
            refineFlag = self.refine(isMarkedCell, surface=surface)
            NN1 = self.number_of_nodes()
            NN0 = len(estimator.rho)
            if NN1 > NN0:
                rho = np.zeros(NN1, dtype=self.ftype)
                rho[:NN0] = self.rho
                edge = self.entity('edge')
                rho[NN0:] = (self.rho[edge[refineFlag, 0]] + self.rho[edge[refineFlag, 1]])/2.0
                mesh = self.to_conformmesh()
                estimator.update(rho, mesh)

    def marker(self, eta, theta, method):
        leafCellIdx = self.leaf_cell_index()
        NC = self.number_of_cells()
        if 'idxmap' in self.celldata.keys(): 
            eta0 = np.zeros(NC, dtype=self.ftype)
            idxmap = self.celldata['idxmap']
            np.add.at(eta0, idxmap, eta)
        else:
            eta0 = eta

        isMarked = mark(eta0[leafCellIdx], theta, method)
        isMarkedCell = np.zeros(NC, dtype=np.bool)
        isMarkedCell[leafCellIdx[isMarked]] = True
        return isMarkedCell 
<<<<<<< HEAD

    def adaptive(self, eta, theta, beta, surface=None, data=None):
        isMarkedCell = self.marker(eta, theta, method='MAX')
        self.refine(isMarkedCell, surface=None, data=None)
        ec = self.entity_barycenter('edge')


||||||| merged common ancestors

    def adaptive(self, eta, theta, beta, surface=None, data=None):
        pass

=======
>>>>>>> 75b60ac13e19b30735d8d972e741adf04d44cf70
        

    def refine(self, isMarkedCell, surface=None):
        if sum(isMarkedCell) > 0:
            # Prepare data
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()
            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')


            # expand the marked cell
            isLeafCell = self.is_leaf_cell()
            edge2cell = self.ds.edge_to_cell()
            flag0 = isLeafCell[edge2cell[:, 0]] & (~isLeafCell[edge2cell[:, 1]])
            flag1 = isLeafCell[edge2cell[:, 1]] & (~isLeafCell[edge2cell[:, 0]])

            LCell = edge2cell[flag0, 0]
            RCell = edge2cell[flag1, 1]
            idx0 = self.localEdge2childCell[edge2cell[flag0, 3]]
            if len(idx0) > 0:
                idx0 = self.child[edge2cell[flag0, [1]].reshape(-1, 1), idx0] 
                 
            idx1 = self.localEdge2childCell[edge2cell[flag1, 2]]
            if len(idx1) > 0:
                idx1 = self.child[edge2cell[flag1, [0]].reshape(-1, 1), idx1]

            assert self.irule == 1      # TODO: add support for  general k irregular rule case 
            cell2cell = self.ds.cell_to_cell()
            isCell0 = isMarkedCell | (~isLeafCell)
            isCell1 = isLeafCell & (~isMarkedCell)
            flag = (isCell1) & (np.sum(isCell0[cell2cell], axis=1) > 1)
            flag2 = ( (~isMarkedCell[LCell]) & isLeafCell[LCell] ) & (isMarkedCell[idx0[:, 0]] | isMarkedCell[idx0[:, 1]])
            flag3 = ( (~isMarkedCell[RCell]) & isLeafCell[RCell] ) & (isMarkedCell[idx1[:, 0]] | isMarkedCell[idx1[:, 1]])
            while np.any(flag) | np.any(flag2) | np.any(flag3):
                isMarkedCell[flag] = True
                isMarkedCell[LCell[flag2]] = True
                isMarkedCell[RCell[flag3]] = True
                isCell0 = isMarkedCell | (~isLeafCell)
                isCell1 = isLeafCell & (~isMarkedCell)
                flag = (isCell1) & (np.sum(isCell0[cell2cell], axis=1) > 1)
                flag2 = ( (~isMarkedCell[LCell]) & isLeafCell[LCell] ) & (isMarkedCell[idx0[:, 0]] | isMarkedCell[idx0[:, 1]])
                flag3 = ( (~isMarkedCell[RCell]) & isLeafCell[RCell] ) & (isMarkedCell[idx1[:, 0]] | isMarkedCell[idx1[:, 1]])
                
            cell2edge = self.ds.cell_to_edge()
            refineFlag = np.zeros(NE, dtype=np.bool)
            refineFlag[cell2edge[isMarkedCell]] = True
            refineFlag[flag0 | flag1] = False

            NNN = refineFlag.sum()
            edge2newNode = np.zeros(NE, dtype=self.itype)
            edge2newNode[refineFlag] = NN + np.arange(NNN)

            edge2newNode[flag0] = cell[self.child[edge2cell[flag0, 1], 3], edge2cell[flag0, 3]]
            edge2newNode[flag1] = cell[self.child[edge2cell[flag1, 0], 3], edge2cell[flag1, 2]]

            # red cell 
            idx, = np.where(isMarkedCell)                                 
            NCC = len(idx) 
            cell4 = np.zeros((4*NCC, 3), dtype=self.itype)
            child4 = -np.ones((4*NCC, 4), dtype=self.itype)
            parent4 = -np.ones((4*NCC, 2), dtype=self.itype) 
            cell4[:NCC, 0] = cell[isMarkedCell, 0] 
            cell4[:NCC, 1] = edge2newNode[cell2edge[isMarkedCell, 2]] 
            cell4[:NCC, 2] = edge2newNode[cell2edge[isMarkedCell, 1]] 
            parent4[:NCC, 0] = idx 
            parent4[:NCC, 1] = 0
            self.child[idx, 0] = NC + np.arange(0, NCC)

            cell4[NCC:2*NCC, 0] = cell[isMarkedCell, 1] 
            cell4[NCC:2*NCC, 1] = edge2newNode[cell2edge[isMarkedCell, 0]] 
            cell4[NCC:2*NCC, 2] = edge2newNode[cell2edge[isMarkedCell, 2]] 
            parent4[NCC:2*NCC, 0] = idx 
            parent4[NCC:2*NCC, 1] = 1
            self.child[idx, 1] = NC + np.arange(NCC, 2*NCC)

            cell4[2*NCC:3*NCC, 0] = cell[isMarkedCell, 2] 
            cell4[2*NCC:3*NCC, 1] = edge2newNode[cell2edge[isMarkedCell, 1]] 
            cell4[2*NCC:3*NCC, 2] = edge2newNode[cell2edge[isMarkedCell, 0]] 
            parent4[2*NCC:3*NCC, 0] = idx 
            parent4[2*NCC:3*NCC, 1] = 2
            self.child[idx, 2] = NC + np.arange(2*NCC, 3*NCC)

            cell4[3*NCC:4*NCC, 0] = edge2newNode[cell2edge[isMarkedCell, 0]] 
            cell4[3*NCC:4*NCC, 1] = edge2newNode[cell2edge[isMarkedCell, 1]] 
            cell4[3*NCC:4*NCC, 2] = edge2newNode[cell2edge[isMarkedCell, 2]]
            parent4[3*NCC:4*NCC, 0] = idx
            parent4[3*NCC:4*NCC, 1] = 3
            self.child[idx, 3] = NC + np.arange(3*NCC, 4*NCC)
            ec = self.entity_barycenter('edge', refineFlag)

            if surface is not None:
                ec, _ = surface.project(ec)

            self.node = np.r_['0', node, ec]
            cell = np.r_['0', cell, cell4]                      
            self.parent = np.r_['0', self.parent, parent4]           
            self.child = np.r_['0', self.child, child4]              
            self.ds.reinit(NN + NNN, cell)
            return refineFlag 
        else:
            return

    def coarsen(self, isMarkedCell, data=None):
        pass


    def to_conformmesh(self):

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        node = self.entity('node')
        edge = self.entity('edge')
        cell = self.entity('cell')
        
        isLeafCell = self.is_leaf_cell()
        edge2cell = self.ds.edge_to_cell()
        
        flag0 = isLeafCell[edge2cell[:, 0]] & (~isLeafCell[edge2cell[:, 1]])
        flag1 = isLeafCell[edge2cell[:, 1]] & (~isLeafCell[edge2cell[:, 0]])

        LCell = edge2cell[flag0, 0]
        RCell = edge2cell[flag1, 1]
        isLeafCell = self.is_leaf_cell()
        idx0 = edge2cell[flag0, 0]
        N0 = len(idx0)
        cell20 =np.zeros((2*N0, 3), dtype=self.itype)

        cell20[0:N0, 0] = edge[flag0, 0]
        cell20[0:N0, 1] = cell[self.child[edge2cell[flag0, 1], 3], edge2cell[flag0, 3]]
        cell20[0:N0, 2] = cell[idx0, edge2cell[flag0, 2]]

        cell20[N0:2*N0, 0] = edge[flag0, 1]
        cell20[N0:2*N0, 1] = cell[idx0, edge2cell[flag0, 2]]
        cell20[N0:2*N0, 2] = cell[self.child[edge2cell[flag0, 1], 3], edge2cell[flag0, 3]]

        idx1 = edge2cell[flag1, 1]
        N1 = len(idx1)
        cell21 =np.zeros((2*N1, 3), dtype=self.itype)

        cell21[0:N1, 0] = edge[flag1, 1]
        cell21[0:N1, 1] = cell[self.child[edge2cell[flag1, 0], 3], edge2cell[flag1, 2]]
        cell21[0:N1, 2] = cell[idx1, edge2cell[flag1, 3]]

        cell21[N1:2*N1, 0] = edge[flag1, 0]
        cell21[N1:2*N1, 1] = cell[idx1, edge2cell[flag1, 3]]
        cell21[N1:2*N1, 2] = cell[self.child[edge2cell[flag1, 0], 3], edge2cell[flag1, 2]]
        
        isLRCell = np.zeros(NC, dtype=np.bool)
        isLRCell[LCell] = True
        isLRCell[RCell] = True

        idx = np.arange(NC)
        cell0 = cell[(~isLRCell) & (isLeafCell)] 
        idx0 = idx[(~isLRCell) & (isLeafCell)]
        cell = np.r_['0', cell0, cell20, cell21]                      
        tmesh = TriangleMesh(node, cell) 
        self.celldata['idxmap'] = np.r_['0', idx0, LCell, RCell, LCell, RCell]        #现在的cell单元与原来树结构中cell的对应关系.
        return tmesh
