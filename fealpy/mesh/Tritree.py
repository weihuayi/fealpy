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
        self.irule = irule
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

    def adaptive_options(
            self,
            method='mean',
            maxrefine=3,
            maxcoarsen=0,
            theta=1.0,
            maxsize=None,
            minsize=None,
            data=None,
            HB=None,
            imatrix=False,
            disp=True
            ):

        options = {
                'method': method,
                'maxrefine': maxrefine,
                'maxcoarsen': maxcoarsen,
                'theta': theta,
                'maxsize': maxsize,
                'minsize': minsize,
                'data': data,
                'HB': HB,
                'imatrix': imatrix,
                'disp': disp
            }
        return options

    def uniform_refine(self, n=1, surface=None):
        for i in range(n):
            self.refine_1(surface=surface)

    def adaptive(self, eta, options, surface=None):
        """
        """
        leafCellIdx = self.leaf_cell_index()

        NC = self.number_of_cells()
        if 'idxmap' in self.celldata.keys():
            eta0 = np.zeros(NC, dtype=self.ftype)
            idxmap = self.celldata['idxmap']
            np.add.at(eta0, idxmap, eta)
            eta = eta0[leafCellIdx]

        options['numrefine'] = np.zeros(NC, dtype=np.int8)
        theta = options['theta']
        if options['method'] is 'mean':
            options['numrefine'][leafCellIdx] = np.around(
                    np.log2(eta/(theta*np.mean(eta)))
                )
        elif options['method'] is 'max':
            options['numrefine'][leafCellIdx] = np.around(
                    np.log2(eta/(theta*np.max(eta)))
                )
        elif options['method'] is 'median':
            options['numrefine'][leafCellIdx] = np.around(
                    np.log2(eta/(theta*np.median(eta)))
                )
        elif options['method'] is 'min':
            options['numrefine'][leafCellIdx] = np.around(
                    np.log2(eta/(theta*np.min(eta)))
                )
        elif isinstance(options['method'], float):
            val = options['method']
            options['numrefine'][leafCellIdx] = np.around(
                    np.log2(eta/val)
                )
        else:
            raise ValueError(
                    "I don't know anyting about method %s!".format(
                        options['method']))

        flag = options['numrefine'] > options['maxrefine']
        options['numrefine'][flag] = options['maxrefine']
        flag = options['numrefine'] < -options['maxcoarsen']
        options['numrefine'][flag] = -options['maxcoarsen']

        h = np.sqrt(self.entity_measure('cell'))
        if options['minsize'] is not None:
            flag = (0.5*h < options['minsize']) & (options['numrefine'] > 0)
            options['numrefine'][flag] = 0

        if options['disp'] is True:
            print(
                    '\n',
                    '\n number of cells: ', len(leafCellIdx),
                    '\n max size of cells: ', np.max(h[leafCellIdx]),
                    '\n min size of cells: ', np.min(h[leafCellIdx]),
                    '\n mean size of cells: ', np.mean(h[leafCellIdx]),
                    '\n median size of cells: ', np.median(h[leafCellIdx]),
                    '\n std size of cells: ', np.std(h[leafCellIdx]),
                    '\n max val of eta: ', np.max(eta),
                    '\n min val of eta: ', np.min(eta),
                    '\n mean val of eta: ', np.mean(eta),
                    '\n median val of eta: ', np.median(eta),
                    '\n std val of eta: ', np.std(eta)
                )

        # refine
        isMarkedCell = (options['numrefine'] > 0)
        while sum(isMarkedCell) > 0:
            self.refine_1(isMarkedCell, options, surface=surface)

            h = np.sqrt(self.entity_measure('cell'))
            if options['minsize'] is not None:
                flag = (0.5*h < options['minsize']) & (options['numrefine'] > 0)
                options['numrefine'][flag] = 0

            if options['disp'] is True:
                leafCellIdx = self.leaf_cell_index()
                print(
                        '\n',
                        '\n number of cells: ', len(leafCellIdx),
                        '\n max size of cells: ', np.max(h[leafCellIdx]),
                        '\n min size of cells: ', np.min(h[leafCellIdx]),
                        '\n mean size of cells: ', np.mean(h[leafCellIdx]),
                        '\n median size of cells: ', np.median(h[leafCellIdx]),
                        '\n std size of cells: ', np.std(h[leafCellIdx])
                    )
            isMarkedCell = (options['numrefine'] > 0)

        # coarsen
        if options['maxcoarsen'] > 0:
            h = np.sqrt(self.entity_measure('cell'))
            if options['maxsize'] is not None:
                flag = (2*h > options['maxsize']) & (options['numrefine'] < 0)
                options['numrefine'][flag] = 0

            isMarkedCell = (options['numrefine'] < 0)
            while sum(isMarkedCell) > 0:
                NN0 = self.number_of_cells()
                self.coarsen_1(isMarkedCell, options)
                NN = self.number_of_cells()
                if NN == NN0:
                    break
                h = np.sqrt(self.entity_measure('cell'))
                if options['maxsize'] is not None:
                    flag = (2*h > options['maxsize']) & (options['numrefine'] < 0)
                    options['numrefine'][flag] = 0

                if options['disp'] is True:
                    leafCellIdx = self.leaf_cell_index()
                    print(
                            '\n',
                            '\n number of cells: ', len(leafCellIdx),
                            '\n max size of cells: ', np.max(h[leafCellIdx]),
                            '\n min size of cells: ', np.min(h[leafCellIdx]),
                            '\n mean size of cells: ', np.mean(h[leafCellIdx]),
                            '\n median size of cells: ', np.median(h[leafCellIdx]),
                            '\n std size of cells: ', np.std(h[leafCellIdx])
                        )

                flag = (2*h > options['maxsize']) & (options['numrefine'] < 0)
                options['numrefine'][flag] = 0
                isMarkedCell = (options['numrefine'] < 0)

    def refine_1(
            self,
            isMarkedCell=None,
            options={'disp': True},
            surface=None):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        if isMarkedCell is None:
            idx = self.leaf_cell_index()
            isMarkedCell = np.zeros(NC, dtype=np.bool)
            isMarkedCell[idx] = True

        if sum(isMarkedCell) > 0:
            # Prepare data
            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')

            # expand the marked cell
            isLeafCell = self.is_leaf_cell()
            edge2cell = self.ds.edge_to_cell()
            flag0 = isLeafCell[edge2cell[:, 0]] & (
                    ~isLeafCell[edge2cell[:, 1]])
            flag1 = isLeafCell[edge2cell[:, 1]] & (
                    ~isLeafCell[edge2cell[:, 0]])

            LCell = edge2cell[flag0, 0]
            RCell = edge2cell[flag1, 1]
            idx0 = self.localEdge2childCell[edge2cell[flag0, 3]]
            if len(idx0) > 0:
                idx0 = self.child[edge2cell[flag0, [1]].reshape(-1, 1), idx0]

            idx1 = self.localEdge2childCell[edge2cell[flag1, 2]]
            if len(idx1) > 0:
                idx1 = self.child[edge2cell[flag1, [0]].reshape(-1, 1), idx1]

            # TODO: add support for  general k irregular rule case 
            assert self.irule == 1

            cell2cell = self.ds.cell_to_cell()
            isCell0 = isMarkedCell | (~isLeafCell)
            isCell1 = isLeafCell & (~isMarkedCell)
            flag = (isCell1) & (np.sum(isCell0[cell2cell], axis=1) > 1)
            flag2 = ((~isMarkedCell[LCell]) & isLeafCell[LCell]) & (isMarkedCell[idx0[:, 0]] | isMarkedCell[idx0[:, 1]])
            flag3 = ((~isMarkedCell[RCell]) & isLeafCell[RCell]) & (isMarkedCell[idx1[:, 0]] | isMarkedCell[idx1[:, 1]])
            while np.any(flag) | np.any(flag2) | np.any(flag3):
                isMarkedCell[flag] = True
                isMarkedCell[LCell[flag2]] = True
                isMarkedCell[RCell[flag3]] = True
                isCell0 = isMarkedCell | (~isLeafCell)
                isCell1 = isLeafCell & (~isMarkedCell)
                flag = (isCell1) & (np.sum(isCell0[cell2cell], axis=1) > 1)
                flag2 = ((~isMarkedCell[LCell]) & isLeafCell[LCell]) & (isMarkedCell[idx0[:, 0]] | isMarkedCell[idx0[:, 1]])
                flag3 = ((~isMarkedCell[RCell]) & isLeafCell[RCell]) & (isMarkedCell[idx1[:, 0]] | isMarkedCell[idx1[:, 1]])

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

            if ('numrefine' in options) and (options['numrefine'] is not None):
                num = options['numrefine'][idx] - 1
                newCellRefine = np.zeros(4*NCC)
                newCellRefine[:NCC] = num
                newCellRefine[NCC:2*NCC] = num
                newCellRefine[2*NCC:3*NCC] = num
                newCellRefine[3*NCC:] = num
                options['numrefine'][idx] = 0
                options['numrefine'] = np.r_[options['numrefine'], newCellRefine]

            if ('data' in options) and (options['data'] is not None):
                for key, value in options['data'].items():
                    t = 0.5*(value[edge[refineFlag, 0]] + value[edge[refineFlag, 1]])
                    options['data'][key] = np.r_['0', value, t]

            if surface is not None:
                ec, _ = surface.project(ec)

            self.node = np.r_['0', node, ec]
            cell = np.r_['0', cell, cell4]
            self.parent = np.r_['0', self.parent, parent4]
            self.child = np.r_['0', self.child, child4]
            self.ds.reinit(NN + NNN, cell)

    def coarsen_1(self, isMarkedCell=None, options={'disp': True}):

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        if isMarkedCell is None:
            idx = self.leaf_cell_index()
            isMarkedCell = np.zeros(NC, dtype=np.bool)
            isMarkedCell[idx] = True

        if sum(isMarkedCell) > 0:
            node = self.entity('node')
            cell = self.entity('cell')

            parent = self.parent
            child = self.child

            isRootCell = self.is_root_cell()
            isLeafCell = self.is_leaf_cell()

            isNotLeafCell = ~isLeafCell

            isMarkedCell[isRootCell] = False

            isMarkedParentCell = np.zeros(NC, dtype=np.bool)
            isMarkedParentCell[parent[isMarkedCell, 0]] = True

            cell2cell = self.ds.cell_to_cell()
            while True:
                flag = (~isMarkedParentCell[cell2cell]) & isNotLeafCell[cell2cell]
                flag = flag.sum(axis=-1) > 1
                if isMarkedParentCell[flag].sum() > 0:
                    isMarkedParentCell[flag] = False
                else:
                    break

            isNeedRemovedCell = np.zeros(NC, dtype=np.bool)
            isNeedRemovedCell[child[isMarkedParentCell, :]] = True

            isRemainNode = np.zeros(NN, dtype=np.bool)
            isRemainNode[cell[~isNeedRemovedCell, :]] = True

            if ('numrefine' in options) and (options['numrefine'] is not None):
                options['numrefine'][isMarkedParentCell] = np.max(
                        options['numrefine'][child[isMarkedParentCell]],
                        axis=-1
                    ) + 1

            cell = cell[~isNeedRemovedCell]
            child = child[~isNeedRemovedCell]
            parent = parent[~isNeedRemovedCell]

            childIdx, = np.nonzero(child[:, 0] > -1)
            isNewLeafCell = np.sum(
                    ~isNeedRemovedCell[child[childIdx, :]], axis=1
                ) == 0
            child[childIdx[isNewLeafCell], :] = -1

            cellIdxMap = np.zeros(NC, dtype=np.int)
            NNC = (~isNeedRemovedCell).sum()
            cellIdxMap[~isNeedRemovedCell] = np.arange(NNC)
            child[child > -1] = cellIdxMap[child[child > -1]]
            parent[parent > -1] = cellIdxMap[parent[parent > -1]]
            self.child = child
            self.parent = parent

            nodeIdxMap = np.zeros(NN, dtype=np.int)
            NN = isRemainNode.sum()
            nodeIdxMap[isRemainNode] = np.arange(NN)
            cell = nodeIdxMap[cell]
            self.node = node[isRemainNode]
            self.ds.reinit(NN, cell)

            if ('numrefine' in options) and (options['numrefine'] is not None):
                options['numrefine'] = options['numrefine'][~isNeedRemovedCell]

            if ('data' in options) and (options['data'] is not None):
                for key, value in options['data'].items():
                    options['data'][key] = value[isRemainNode]

    def adaptive_refine(self, estimator, surface=None, data=None):
        i = 0
        if data is not None:
            if 'rho' not in data:
                data['rho'] = estimator.rho
        else:
            data = {'rho': estimator.rho}

        while True:
            i += 1
            isMarkedCell = self.refine_marker(
                    estimator.eta, estimator.theta, estimator.ep)
            if sum(isMarkedCell) == 0 or i > 3:
                break
            self.refine(isMarkedCell, surface=surface, data=data)
            mesh = self.to_conformmesh()
            estimator.update(data['rho'], mesh, smooth=True)

    def refine_marker(self, eta, theta, method):
        leafCellIdx = self.leaf_cell_index()
        NC = self.number_of_cells()
        if 'idxmap' in self.celldata.keys():
            eta0 = np.zeros(NC, dtype=self.ftype)
            idxmap = self.celldata['idxmap']
            np.add.at(eta0, idxmap, eta)
            eta = eta0[leafCellIdx]

        isMarked = mark(eta, theta, method)
        isMarkedCell = np.zeros(NC, dtype=np.bool)
        isMarkedCell[leafCellIdx[isMarked]] = True
        return isMarkedCell

    def refine(self, isMarkedCell, surface=None, data=None):
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
            flag0 = isLeafCell[edge2cell[:, 0]] & (
                    ~isLeafCell[edge2cell[:, 1]])
            flag1 = isLeafCell[edge2cell[:, 1]] & (
                    ~isLeafCell[edge2cell[:, 0]])

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
            flag2 = ((~isMarkedCell[LCell]) & isLeafCell[LCell]) & (isMarkedCell[idx0[:, 0]] | isMarkedCell[idx0[:, 1]])
            flag3 = ((~isMarkedCell[RCell]) & isLeafCell[RCell]) & (isMarkedCell[idx1[:, 0]] | isMarkedCell[idx1[:, 1]])
            while np.any(flag) | np.any(flag2) | np.any(flag3):
                isMarkedCell[flag] = True
                isMarkedCell[LCell[flag2]] = True
                isMarkedCell[RCell[flag3]] = True
                isCell0 = isMarkedCell | (~isLeafCell)
                isCell1 = isLeafCell & (~isMarkedCell)
                flag = (isCell1) & (np.sum(isCell0[cell2cell], axis=1) > 1)
                flag2 = ((~isMarkedCell[LCell]) & isLeafCell[LCell]) & (isMarkedCell[idx0[:, 0]] | isMarkedCell[idx0[:, 1]])
                flag3 = ((~isMarkedCell[RCell]) & isLeafCell[RCell]) & (isMarkedCell[idx1[:, 0]] | isMarkedCell[idx1[:, 1]])

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

            if data is not None:
                I = cell[edge2cell[refineFlag, 0], edge2cell[refineFlag, 2]]
                J = cell[edge2cell[refineFlag, 1], edge2cell[refineFlag, 3]]
                for key, value in data.items():
                    t = (3*value[edge[refineFlag, 0]] + 3*value[edge[refineFlag, 1]] + 
                            value[I] + value[J])/8
                    data[key] = np.r_['0', value, t]

            if surface is not None:
                ec, _ = surface.project(ec)

            self.node = np.r_['0', node, ec]
            cell = np.r_['0', cell, cell4]
            self.parent = np.r_['0', self.parent, parent4]
            self.child = np.r_['0', self.child, child4]
            self.ds.reinit(NN + NNN, cell)

    def adaptive_coarsen(self, estimator, surface=None, data=None):

        if data is not None:
            data['rho'] = estimator.rho
        else:
            data = {'rho': estimator.rho}

        i = 0
        while True:
            i += 1
            isMarkedCell = self.coarsen_marker(
                    estimator.eta,
                    estimator.beta,
                    estimator.ep)

            if (sum(isMarkedCell) == 0) or i > 3:
                break

            isRemainNode = self.coarsen(isMarkedCell)
            mesh = self.to_conformmesh()
            for key, value in data.items():
                data[key] = value[isRemainNode]
                estimator.update(data['rho'], mesh, smooth=False)

            isRootCell = self.is_root_cell()
            NC = self.number_of_cells()

            if isRootCell.sum() == NC:
                break

    def coarsen_marker(self, eta, beta, method):
        leafCellIdx = self.leaf_cell_index()
        NC = self.number_of_cells()
        if 'idxmap' in self.celldata.keys(): 
            eta0 = np.zeros(NC, dtype=self.ftype)
            idxmap = self.celldata['idxmap']
            np.add.at(eta0, idxmap, eta)
        else:
            eta0 = eta
        isMarked = mark(eta0[leafCellIdx], beta, method)
        isMarkedCell = np.zeros(NC, dtype=np.bool)
        isMarkedCell[leafCellIdx[isMarked]] = True
        return isMarkedCell 

    def coarsen(self, isMarkedCell, data=None):
        if sum(isMarkedCell) > 0:
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            node = self.entity('node')
            cell = self.entity('cell')

            parent = self.parent
            child = self.child

            isRootCell = self.is_root_cell()
            isLeafCell = self.is_leaf_cell()

            isNotLeafCell = ~isLeafCell 

            isMarkedCell[isRootCell] = False

            isMarkedParentCell = np.zeros(NC, dtype=np.bool)
            isMarkedParentCell[parent[isMarkedCell, 0]] = True

            cell2cell = self.ds.cell_to_cell()
            while True:
                flag = (~isMarkedParentCell[cell2cell]) & isNotLeafCell[cell2cell]
                flag = flag.sum(axis=-1) > 1
                if isMarkedParentCell[flag].sum() > 0:
                    isMarkedParentCell[flag] = False
                else:
                    break

            isNeedRemovedCell = np.zeros(NC, dtype=np.bool)
            isNeedRemovedCell[child[isMarkedParentCell, :]] = True

            isRemainNode = np.zeros(NN, dtype=np.bool)
            isRemainNode[cell[~isNeedRemovedCell, :]] = True

            cell = cell[~isNeedRemovedCell]
            child = child[~isNeedRemovedCell]
            parent = parent[~isNeedRemovedCell]

            childIdx, = np.nonzero(child[:, 0] > -1)
            isNewLeafCell = np.sum(~isNeedRemovedCell[child[childIdx, :]], axis=1) == 0 
            child[childIdx[isNewLeafCell], :] = -1

            cellIdxMap = np.zeros(NC, dtype=np.int)
            NNC = (~isNeedRemovedCell).sum()
            cellIdxMap[~isNeedRemovedCell] = np.arange(NNC)
            child[child > -1] = cellIdxMap[child[child > -1]]
            parent[parent > -1] = cellIdxMap[parent[parent > -1]]
            self.child = child
            self.parent = parent

            nodeIdxMap = np.zeros(NN, dtype=np.int)
            NN = isRemainNode.sum()
            nodeIdxMap[isRemainNode] = np.arange(NN)
            cell = nodeIdxMap[cell]
            self.node = node[isRemainNode]
            self.ds.reinit(NN, cell)
            return isRemainNode
        else:
            return 

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
