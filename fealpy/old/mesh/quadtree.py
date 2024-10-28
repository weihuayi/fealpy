import numpy as np
from scipy.sparse import coo_matrix
from .quadrangle_mesh import QuadrangleMesh
from fealpy.common import ranges
from .adaptive_tools import mark
from .polygon_mesh import PolygonMesh

class Quadtree(QuadrangleMesh):
    localEdge2childCell = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.int_)

    def __init__(self, node, cell):
        super(Quadtree, self).__init__(node, cell)
        NC = self.number_of_cells()
        self.parent = -np.ones((NC, 2), dtype=self.itype)
        self.child = -np.ones((NC, 4), dtype=self.itype)
        self.cellLocation = np.array([0,NC],dtype=np.int_)
        self.meshType = 'quadtree'

    def leaf_cell_index(self):
        child = self.child
        idx, = np.nonzero(child[:, 0] == -1)
        return idx

    def leaf_cell(self, celltype='quad'):
        child = self.child
        cell = self.ds.cell[child[:, 0] == -1]
        if celltype == 'quad':
            return cell
        elif celltype == 'tri':
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

    def adaptive_options(
            self,
            method='mean',
            maxrefine=3,
            maxcoarsen=3,
            theta=1.0,
            maxsize=1e-2,
            minsize=1e-12,
            data=None,
            HB=True,
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

    def uniform_refine(self, n=1):
        for i in range(n):
            self.refine_1()

    def adaptive(self, eta, options):

        if options['HB'] is True:
            HB = np.zeros((len(eta), 2), dtype=np.int_)
            HB[:, 0] = np.arange(len(eta))
            HB[:, 1] = np.arange(len(eta))
            options['HB'] = HB

        leafCellIdx = self.leaf_cell_index()
        NC = self.number_of_cells()
        options['numrefine'] = np.zeros(NC, dtype=np.int8)
        theta = options['theta']
        if options['method'] == 'mean':
            options['numrefine'][leafCellIdx] = np.around(
                    np.log2(eta/(theta*np.mean(eta)))
                )
            # options['numrefine'][leafCellIdx] = eta
        elif options['method'] == 'max':
            options['numrefine'][leafCellIdx] = np.around(
                    np.log2(eta/(theta*np.max(eta)))
                )
        elif options['method'] == 'median':
            options['numrefine'][leafCellIdx] = np.around(
                    np.log2(eta/(theta*np.median(eta)))
                )
        elif options['method'] == 'min':
            options['numrefine'][leafCellIdx] = np.around(
                    np.log2(eta/(theta*np.min(eta)))
                )
        elif options['method'] == 'numrefine':
            options['numrefine'][leafCellIdx]= eta
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
            self.refine_1(isMarkedCell, options)

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

                if options['maxsize'] is not None:
                    flag = (2*h > options['maxsize']) & (options['numrefine'] < 0)
                    options['numrefine'][flag] = 0
                isMarkedCell = (options['numrefine'] < 0)

    def refine_1(self, isMarkedCell=None, options={'disp': True}):
        if isMarkedCell is None:
            # 默认加密所有的叶子单元
            idx = self.leaf_cell_index()
            if ('HB' in options) and (options['HB'] is not None):
                isLeafCell = self.is_leaf_cell()
                NC = isLeafCell.sum()
                HB = np.zeros((4*NC, 2), dtype=self.itype)
                HB[:, 0] = range(4*NC)
                HB[:, 1] = np.repeat(options['HB'][:, 1], 4)
        else:
            if ('HB' in options) and (options['HB'] is not None):
                isNonMarkedCell = ~isMarkedCell
                isLeafCell = self.is_leaf_cell()
                flag0 = isNonMarkedCell[isLeafCell]
                flag1 = isMarkedCell[isLeafCell]
                NHB0 = flag0.sum()
                NHB1 = flag1.sum()
                NHB = NHB0 + 4*NHB1
                HB = np.zeros((NHB, 2), dtype=np.int_)
                HB[:, 0] = range(NHB)
                HB[0:NHB0, 1] = options['HB'][flag0, 1]
                HB[NHB0:,  1] = np.repeat(options['HB'][flag1, 1], 4)
                options['HB'] = HB
            idx, = np.nonzero(isMarkedCell)
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
            cellLocation = self.cellLocation

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
            cellIdx = np.zeros(NE, dtype=self.itype)
            localIdx = np.zeros(NE, dtype=self.itype)
            I1 = I[~isLeafCell[I]]
            J1 = J[~isLeafCell[I]]
            cellIdx[cell2edge[I1, J1]] = I1  # the cell idx
            localIdx[cell2edge[I1, J1]] = J1  #
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

            if ('data' in options) and (options['data'] is not None):
                isNeedCutEdge = (~isCuttedEdge) & isCutEdge
                for key, value in options['data'].items():
                    evalue = 0.5*np.sum(value[edge[isNeedCutEdge]], axis=1)
                    cvalue = np.sum(value[cell[isNeedCutCell]], axis=1)/4
                    options['data'][key] = np.concatenate((value, evalue, cvalue), axis=0)


            NEC = len(edgeCenter)
            NCC = len(cellCenter)
            edge2center[isNeedCutEdge] = np.arange(N, N+NEC)

            cp = [cell[isNeedCutCell, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[isNeedCutCell, i]].reshape(-1, 1) for i in range(4)]
            cc = np.arange(N + NEC, N + NEC + NCC).reshape(-1, 1)

            if ('numrefine' in options) and (options['numrefine'] is not None):
                num = options['numrefine'][idx] - 1
                newCellRefine = np.zeros(4*NCC)
                newCellRefine[0::4] = num
                newCellRefine[1::4] = num
                newCellRefine[2::4] = num
                newCellRefine[3::4] = num
                options['numrefine'][idx] = 0
                options['numrefine'] = np.r_[options['numrefine'], newCellRefine]

            newCell = np.zeros((4*NCC, 4), dtype=self.itype)
            newChild = -np.ones((4*NCC, 4), dtype=self.itype)
            newParent = -np.ones((4*NCC, 2), dtype=self.itype)
            newCell[0::4, :] = np.concatenate(
                    (cp[0], ep[0], cc, ep[3]), axis=1)
            newCell[1::4, :] = np.concatenate(
                    (ep[0], cp[1], ep[1], cc), axis=1)
            newCell[2::4, :] = np.concatenate(
                    (cc, ep[1], cp[2], ep[2]), axis=1)
            newCell[3::4, :] = np.concatenate(
                    (ep[3], cc, ep[2], cp[3]), axis=1)
            newParent[:, 0] = np.repeat(idx, 4)
            newParent[:, 1] = ranges(4*np.ones(NCC, dtype=self.itype))
            child[idx, :] = np.arange(NC, NC + 4*NCC).reshape(NCC, 4)

            cell = np.concatenate((cell, newCell), axis=0)
            self.node = np.concatenate((node, edgeCenter, cellCenter), axis=0)
            self.parent = np.concatenate((parent, newParent), axis=0)
            self.child = np.concatenate((child, newChild), axis=0)
            self.cellLocation = np.append(cellLocation,cellLocation[-1]+4*NCC)
            self.ds.reinit(N + NEC + NCC, cell)

    def coarsen_1(self, isMarkedCell=None, options={'disp': True}):
        """ marker will marke the leaf cells which will be coarsen
        """
        isRootCell = self.is_root_cell()
        if np.all(isRootCell):
            return False

        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        parent = self.parent
        child = self.child

        isLeafCell = self.is_leaf_cell()
        isBranchCell = np.zeros(NC, dtype=np.bool_)
        isBranchCell[parent[isLeafCell & (~isRootCell), 0]] = True

        idx, = np.nonzero(isBranchCell)
        isCoarsenCell = np.sum(isMarkedCell[child[isBranchCell]], axis=1) == 4
        idx = idx[isCoarsenCell] # 需要被粗化的单元

        if len(idx) > 0:

            node = self.node
            cell = self.ds.cell
            if ('numrefine' in options) and (options['numrefine'] is not None):
                options['numrefine'][idx] = np.max(
                        options['numrefine'][child[idx]], axis=-1
                    ) + 1

            isRemainCell = np.ones(NC, dtype=np.bool_)
            isRemainCell[child[idx, :]] = False

            isRemainNode = np.zeros(NN, dtype=np.bool_)
            isRemainNode[cell[isRemainCell, :]] = True

            NNC = isRemainCell.sum()
            if NNC == NC: # 如果保留单元个数与原始单元个数相同
                return

            cell = cell[isRemainCell]
            child = child[isRemainCell]
            parent = parent[isRemainCell]

            # 子单元不需要保留的单元， 是新的叶子单元
            childIdx, = np.nonzero(child[:, 0] > -1)
            isNewLeafCell = (
                    np.sum(isRemainCell[child[childIdx, :]], axis=1) == 0
                )
            child[childIdx[isNewLeafCell], :] = -1

            # 老单元与新单元的编号映射关系
            cellIdxMap = np.zeros(NC, dtype=self.itype)
            cellIdxMap[isRemainCell] = np.arange(NNC)
            # 下面的代码有 bug 
            # cellIdxMap[child[idx, :]] = np.reshape(idx, (-1, 1)) 
            cellIdxMap[self.child[idx, :]] = np.reshape(idx, (-1, 1))
            child[child > -1] = cellIdxMap[child[child > -1]]
            parent[parent > -1] = cellIdxMap[parent[parent > -1]]

            if ('HB' in options) and (options['HB'] is not None):
                # 粗化要更新 HB[:, 0]
                leafIdx = self.leaf_cell_index()
                idx0 = leafIdx[options['HB'][:, 0]] # 叶子单元在所有单元中对应的编号
                idx0 = cellIdxMap[idx0] # 在保留单元中的编号
                isLeafCell = (child[:, 0] == -1) # 新网格的叶子单元
                idxMap = np.zeros(NNC, dtype=self.itype)
                idxMap[isLeafCell] = range(isLeafCell.sum())
                options['HB'][:, 0] = idxMap[idx0]

            self.child = child
            self.parent = parent

            nodeIdxMap = np.zeros(NN, dtype=self.itype)
            N = isRemainNode.sum()
            nodeIdxMap[isRemainNode] = np.arange(N)
            cell = nodeIdxMap[cell]
            self.node = node[isRemainNode]
            self.ds.reinit(N, cell)

            if ('numrefine' in options) and (options['numrefine'] is not None):
                options['numrefine'] = options['numrefine'][isRemainCell]

            if ('data' in options) and (options['data'] is not None):
                # 这里只是做线性插值
                for key, value in options['data'].items():
                    options['data'][key] = value[isRemainNode]

    def adaptive_refine(self, estimator, data=None):
        i = 0
        if data is not None:
            if 'rho' not in data:
                data['rho'] = estimator.rho
        else:
            data = {'rho': estimator.rho}
        while True:

            i += 1
            isMarkedCell = self.refine_marker(
                    estimator.eta,
                    estimator.theta)
            if sum(isMarkedCell) == 0:
                break
            self.refine(isMarkedCell, data=data)
            mesh = self.to_pmesh()
            estimator.update(data['rho'], mesh, smooth=True)
            if i > 3:
                break

    def refine_marker(self, eta, theta, method="L2"):
        leafCellIdx = self.leaf_cell_index()
        NC = self.number_of_cells()
        isMarked = mark(eta, theta, method=method)
        isMarkedCell = np.zeros(NC, dtype=np.bool_)
        isMarkedCell[leafCellIdx[isMarked]] = True
        return isMarkedCell

    def refine(self, isMarkedCell=None, data=None):
        if isMarkedCell is None:
            idx = self.leaf_cell_index()
        else:
            idx, = np.nonzero(isMarkedCell)

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
            cellLocation = self.cellLocation

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
            cellIdx = np.zeros(NE, dtype=self.itype)
            localIdx = np.zeros(NE, dtype=self.itype)
            I1 = I[~isLeafCell[I]]
            J1 = J[~isLeafCell[I]]
            cellIdx[cell2edge[I1, J1]] = I1  # the cell idx
            localIdx[cell2edge[I1, J1]] = J1  #
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

            if data is not None:
                isNeedCutEdge = (~isCuttedEdge) & isCutEdge
                for key, value in data.items():
                    evalue = 0.5*np.sum(value[edge[isNeedCutEdge]], axis=1)
                    cvalue = np.sum(value[cell[isNeedCutCell]], axis=1)/4
                    data[key] = np.concatenate((value, evalue, cvalue), axis=0)

            NEC = len(edgeCenter)
            NCC = len(cellCenter)

            edge2center[isNeedCutEdge] = np.arange(N, N+NEC)

            cp = [cell[isNeedCutCell, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[isNeedCutCell, i]].reshape(-1, 1) for i in range(4)]
            cc = np.arange(N + NEC, N + NEC + NCC).reshape(-1, 1)

            newCell = np.zeros((4*NCC, 4), dtype=self.itype)
            newChild = -np.ones((4*NCC, 4), dtype=self.itype)
            newParent = -np.ones((4*NCC, 2), dtype=self.itype)
            newCell[0::4, :] = np.concatenate(
                    (cp[0], ep[0], cc, ep[3]), axis=1)
            newCell[1::4, :] = np.concatenate(
                    (ep[0], cp[1], ep[1], cc), axis=1)
            newCell[2::4, :] = np.concatenate(
                    (cc, ep[1], cp[2], ep[2]), axis=1)
            newCell[3::4, :] = np.concatenate(
                    (ep[3], cc, ep[2], cp[3]), axis=1)
            newParent[:, 0] = np.repeat(idx, 4)
            newParent[:, 1] = ranges(4*np.ones(NCC, dtype=self.itype))
            child[idx, :] = np.arange(NC, NC + 4*NCC).reshape(NCC, 4)

            cell = np.concatenate((cell, newCell), axis=0)
            self.node = np.concatenate((node, edgeCenter, cellCenter), axis=0)
            self.parent = np.concatenate((parent, newParent), axis=0)
            self.child = np.concatenate((child, newChild), axis=0)
            self.cellLocation = np.append(cellLocation,cellLocation[-1]+4*NCC)
            self.ds.reinit(N + NEC + NCC, cell)

    def adaptive_coarsen(self, estimator, data=None):
        i = 0
        if data is not None:
                data['rho'] = estimator.rho
        else:
            data = {'rho': estimator.rho}
        while True:
            i += 1
            isMarkedCell = self.coarsen_marker(
                    estimator.eta,
                    estimator.beta)

            if sum(isMarkedCell) == 0 | i > 3:
                break

            isRemainNode = self.coarsen(isMarkedCell)
            for key, value in data.items():
                data[key] = value[isRemainNode]

            mesh = self.to_pmesh()
            estimator.update(data['rho'], mesh, smooth=False)

            isRootCell = self.is_root_cell()
            NC = self.number_of_cells()
            if isRootCell.sum() == NC:
                break

    def coarsen_marker(self, eta, beta):
        leafCellIdx = self.leaf_cell_index()
        isMarked = mark(eta, beta, method="COARSEN")
        NC = self.number_of_cells()
        isMarkedCell = np.zeros(NC, dtype=np.bool_)
        isMarkedCell[leafCellIdx[isMarked]] = True
        return isMarkedCell

    def coarsen(self, isMarkedCell, data=None):
        """ marker will marke the leaf cells which will be coarsen
        """
        isRootCell = self.is_root_cell()
        if np.all(isRootCell):
            return False

        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        parent = self.parent
        child = self.child

        isLeafCell = self.is_leaf_cell()
        isBranchCell = np.zeros(NC, dtype=np.bool_)
        isBranchCell[parent[isLeafCell & (~isRootCell), 0]] = True

        idx, = np.nonzero(isBranchCell)
        isCoarsenCell = np.sum(isMarkedCell[child[isBranchCell]], axis=1) == 4

        idx = idx[isCoarsenCell]

        if len(idx) > 0:

            node = self.node
            cell = self.ds.cell

            isRemainCell = np.ones(NC, dtype=np.bool_)
            isRemainCell[child[idx, :]] = False

            isRemainNode = np.zeros(NN, dtype=np.bool_)
            isRemainNode[cell[isRemainCell, :]] = True

            cell = cell[isRemainCell]
            child = child[isRemainCell]
            parent = parent[isRemainCell]

            # 子单元不需要保留的单元， 是新的叶子单元

            childIdx, = np.nonzero(child[:, 0] > -1)
            isNewLeafCell = np.sum(
                    isRemainCell[child[childIdx, :]], axis=1
                ) == 0
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
