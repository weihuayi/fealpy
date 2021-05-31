import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, bmat, eye
from scipy.spatial import KDTree
from .Mesh2d import Mesh2d, Mesh2dDataStructure
from ..quadrature import TriangleQuadrature
from ..quadrature import GaussLegendreQuadrature

class TriangleMeshDataStructure(Mesh2dDataStructure):
    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    localFace = np.array([(1, 2), (2, 0), (0, 1)])
    ccw = np.array([0, 1, 2])
    NVC = 3
    NEC = 3
    NFC = 3
    def __init__(self, NN, cell):
        super(TriangleMeshDataStructure, self).__init__(NN, cell)

class TriangleMesh(Mesh2d):
    def __init__(self, node, cell):

        self.node = node
        N = node.shape[0]
        self.ds = TriangleMeshDataStructure(N, cell)

        if node.shape[1] == 2:
            self.meshtype = 'tri'
        elif node.shape[1] == 3:
            self.meshtype = 'stri'

        self.itype = cell.dtype
        self.ftype = node.dtype
        self.p = 1 # 平面三角形

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.meshdata = {}

    def number_of_corner_nodes(self):
        return self.ds.NN

    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_TRIANGLE = 5
            return VTK_TRIANGLE
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
        """
        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from .vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell]
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)

    def integrator(self, q, etype='cell'):
        if etype in {'cell', 2}:
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            return GaussLegendreQuadrature(q)

    def copy(self):
        return TriangleMesh(self.node.copy(), self.ds.cell.copy());

    def delete_cell(self, threshold):
        NN = self.number_of_nodes()

        cell = self.entity('cell')
        node = self.entity('node')

        bc = self.entity_barycenter('cell')
        isKeepCell = ~threshold(bc)
        cell = cell[isKeepCell]

        isValidNode = np.zeros(NN, dtype=np.bool_)
        isValidNode[cell] = True
        node = node[isValidNode]

        idxMap = np.zeros(NN, dtype=self.itype)
        idxMap[isValidNode] = range(isValidNode.sum())
        cell = idxMap[cell]
        self.node = node
        NN = len(node)
        self.ds.reinit(NN, cell)

    def to_quadmesh(self):
        from ..mesh import QuadrangleMesh

        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        node0 = self.entity('node')
        cell0 = self.entity('cell')
        ec = self.entity_barycenter('edge')
        cc = self.entity_barycenter('cell')
        cell2edge = self.ds.cell_to_edge()

        node = np.r_['0', node0, ec, cc]
        cell = np.zeros((3*NC, 4), dtype=self.itype)
        idx = np.arange(NC)
        cell[:NC, 0] = NN + NE + idx
        cell[:NC, 1] = cell2edge[:, 0] + NN
        cell[:NC, 2] = cell0[:, 2]
        cell[:NC, 3] = cell2edge[:, 1] + NN

        cell[NC:2*NC, 0] = cell[:NC, 0]
        cell[NC:2*NC, 1] = cell2edge[:, 1] + NN
        cell[NC:2*NC, 2] = cell0[:, 0]
        cell[NC:2*NC, 3] = cell2edge[:, 2] + NN

        cell[2*NC:3*NC, 0] = cell[:NC, 0]
        cell[2*NC:3*NC, 1] = cell2edge[:, 2] + NN
        cell[2*NC:3*NC, 2] = cell0[:, 1]
        cell[2*NC:3*NC, 3] = cell2edge[:, 0] + NN
        return QuadrangleMesh(node, cell)

    def egde_merge(self, h0):
        edge = self.entity('edge')
        h = self.entity_measure('edge')
        isShortEdge = h < h0


    def is_crossed_cell(self, point, segment):
        """

        Notes
        -----

        给定一组线段，找到这些线段的一个邻域单元集合, 且这些单元要满足一定的连通
        性
        """

        nx = np.array([1, 2, 0])
        pr = np.array([2, 0, 1])

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        node = self.entity('node')
        cell = self.entity('cell')
        cell2cell = self.ds.cell_to_cell()

        # 用于标记被线段穿过的网格节点，这些节点周围的单元都会被标记为
        # 穿过单元，这样保持了加密单元的连通性
        isCrossedNode = np.zeros(NN, dtype=np.bool_)
        isCrossedCell = np.zeros(NC, dtype=np.bool_)

        # 找到线段端点所属的网格单元， 并标记为穿过单元
        location = self.location(point)
        isCrossedCell[location] = True


        # 从一个端点所在的单元出发，走到另一个端点所在的单元
        p0 = point[segment[:, 0]] # 线段起点
        p1 = point[segment[:, 1]] # 线段终点
        v = p0 - p1

        start = location[segment[:, 0]] # 出发单元
        end = location[segment[:, 1]] # 终止单元

        isNotOK = np.ones(len(segment), dtype=np.bool_)
        jdx = 3
        while isNotOK.any():
            idx = start[isNotOK] # 当前单元 

            pp0 = p0[isNotOK]
            pp1 = p1[isNotOK]
            vv = v[isNotOK]

            a = np.zeros((len(idx), 3), dtype=self.ftype)
            v0 = node[cell[idx, 0]] - pp1 # 所在单元的三个顶点
            v1 = node[cell[idx, 1]] - pp1
            v2 = node[cell[idx, 2]] - pp1
            a[:, 0] = np.cross(v0, vv)
            a[:, 1] = np.cross(v1, vv)
            a[:, 2] = np.cross(v2, vv)

            b = np.zeros((len(idx), 3), dtype=self.ftype)
            b[:, 0] = np.cross(v1, v2)
            b[:, 1] = np.cross(v2, v0)
            b[:, 2] = np.cross(v0, v1)

            isOK = np.sum(b >=0, axis=-1) == 3
            idx0, = np.nonzero(isNotOK)

            for i in range(3):
                flag = np.abs(a[:, i]) < 1e-12
                isCrossedNode[cell[idx[flag], i]] = True

            lidx = np.zeros(len(idx), dtype=np.int_)
            for i in range(3):
                j = nx[i]
                k = pr[i]
                flag0 = (a[:, j] <= 0) & (a[:, k] >=0) & (jdx!=i)
                lidx[flag0] = i

            # 移动到下一个单元
            tmp = start[idx0[~isOK]]
            start[idx0[~isOK]] = cell2cell[idx[~isOK], lidx[~isOK]]
            isNotOK[idx0[isOK]] = False 
            _, jdx = np.where((cell2cell[start[isNotOK]].T==tmp).T)

            # 这些单元标记为穿过单元
            isCrossedCell[start] = True



        # 处理被线段穿过的网格点的连通性

        edge = self.entity('edge')
        edge2cell = self.ds.edge_to_cell()
        isFEdge0 = isCrossedCell[edge2cell[:, 0]] & (~isCrossedCell[edge2cell[:, 1]])
        isFEdge1 = (~isCrossedCell[edge2cell[:, 0]]) & isCrossedCell[edge2cell[:, 1]]
        flag = isFEdge0 | isFEdge1

        if np.any(flag):
            valence = np.zeros(NN, dtype=self.itype)
            np.add.at(valence, edge[flag], 1)
            isCrossedNode[valence > 2] = True
            for i in range(3):
                np.logical_or.at(isCrossedCell, range(NC), isCrossedNode[cell[:, i]])

        return isCrossedCell

    def location(self, points):
        """
        Notes
        -----
        给定一组点 p ， 找到这些点所在的单元

        这里假设：

        1. 所有点在网格内部，
        2. 网格中没有洞
        3. 区域还要是凸的
        """

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NP = points.shape[0]
        node = self.entity('node')
        cell = self.entity('cell')
        cell2cell = self.ds.cell_to_cell()

        start = np.zeros(NN, dtype=self.itype)
        start[cell[:, 0]] = range(NC)
        start[cell[:, 1]] = range(NC)
        start[cell[:, 2]] = range(NC)
        tree = KDTree(node)
        _, loc = tree.query(points)
        start = start[loc] # 设置一个初始单元位置

        isNotOK = np.ones(NP, dtype=np.bool)
        while np.any(isNotOK):
            idx = start[isNotOK]
            pp = points[isNotOK]

            v0 = node[cell[idx, 0]] - pp # 所在单元的三个顶点
            v1 = node[cell[idx, 1]] - pp 
            v2 = node[cell[idx, 2]] - pp

            a = np.zeros((len(idx), 3), dtype=self.ftype)
            a[:, 0] = np.cross(v1, v2)
            a[:, 1] = np.cross(v2, v0)
            a[:, 2] = np.cross(v0, v1)
            lidx = np.argmin(a, axis=-1) 

            # 最小面积小于 0, 说明点在单元外
            isOutCell = a[range(a.shape[0]), lidx] < 0.0 

            idx0, = np.nonzero(isNotOK)
            start[idx0[isOutCell]] = cell2cell[idx[isOutCell], lidx[isOutCell]]
            isNotOK[idx0[~isOutCell]] = False

        return start 

    def circumcenter(self):
        node = self.node
        cell = self.ds.cell
        GD = self.geo_dimension()

        v0 = node[cell[:,2],:] - node[cell[:,1],:]
        v1 = node[cell[:,0],:] - node[cell[:,2],:]
        v2 = node[cell[:,1],:] - node[cell[:,0],:]
        nv = np.cross(v2, -v1)
        if GD == 2:
            area = nv/2.0
            x2 = np.sum(node**2, axis=1, keepdims=True)
            w0 = x2[cell[:,2]] + x2[cell[:,1]]
            w1 = x2[cell[:,0]] + x2[cell[:,2]]
            w2 = x2[cell[:,1]] + x2[cell[:,0]]
            W = np.array([[0, -1],[1, 0]], dtype=self.ftype)
            fe0 = w0*v0@W
            fe1 = w1*v1@W
            fe2 = w2*v2@W
            c = 0.25*(fe0 + fe1 + fe2)/area.reshape(-1,1)
            R = np.sqrt(np.sum((c-node[cell[:,0], :])**2,axis=1))
        elif GD == 3:
            length = np.sqrt(np.sum(nv**2, axis=1))
            n = nv/length.reshape((-1, 1))
            l02 = np.sum(v1**2, axis=1, keepdims=True)
            l01 = np.sum(v2**2, axis=1, keepdims=True)
            d = 0.5*(l02*np.cross(n, v2) + l01*np.cross(-v1, n))/length.reshape(-1, 1)
            c = node[cell[:, 0]] + d
            R = np.sqrt(np.sum(d**2, axis=1))
        return c, R

    def angle(self):
        NC = self.number_of_cells()
        cell = self.ds.cell
        node = self.node
        localEdge = self.ds.local_edge()
        angle = np.zeros((NC, 3), dtype=self.ftype)
        for i,(j,k) in zip(range(3),localEdge):
            v0 = node[cell[:,j]] - node[cell[:,i]]
            v1 = node[cell[:,k]] - node[cell[:,i]]
            angle[:, i] = np.arccos(np.sum(v0*v1, axis=1)/np.sqrt(np.sum(v0**2, axis=1) * np.sum(v1**2, axis=1)))
        return angle

    def edge_swap(self):
        while True:
            # Construct necessary data structure
            edge2cell = self.ds.edge_to_cell()
            cell2edge = self.ds.cell_to_edge()

            # Find non-Delaunay edges
            angle = self.angle()
            asum = np.sum(angle[edge2cell[:, 0:2], edge2cell[:, 2:4]], axis=1)
            isNonDelaunayEdge = (asum > np.pi) & (edge2cell[:,0] != edge2cell[:,1])

            return isNonDelaunayEdge
            
            if np.sum(isNonDelaunayEdge) == 0:
                break
            # Find dependent set of swap edges
            isCheckCell = np.sum(isNonDelaunayEdge[cell2edge], axis=1) > 1
            if np.any(isCheckCell):
                ac = asum[cell2edge[isCheckCell, :]]
                isNonDelaunayEdge[cell2edge[isCheckCell, :]] = False
                I = np.argmax(ac, axis=1)
                isNonDelaunayEdge[cell2edge[isCheckCell, I]] = True

            if np.any(isNonDelaunayEdge):
                cell = self.ds.cell
                pnext = np.array([1, 2, 0])
                idx = edge2cell[isNonDelaunayEdge, 2]
                p0 = cell[edge2cell[isNonDelaunayEdge, 0], idx]
                p1 = cell[edge2cell[isNonDelaunayEdge, 0], pnext[idx]] 
                idx = edge2cell[isNonDelaunayEdge, 3]
                p2 = cell[edge2cell[isNonDelaunayEdge, 1], idx]
                p3 = cell[edge2cell[isNonDelaunayEdge, 1], pnext[idx]]
                cell[edge2cell[isNonDelaunayEdge, 0], 0] = p1
                cell[edge2cell[isNonDelaunayEdge, 0], 1] = p2
                cell[edge2cell[isNonDelaunayEdge, 0], 2] = p0

                cell[edge2cell[isNonDelaunayEdge, 1], 0] = p3
                cell[edge2cell[isNonDelaunayEdge, 1], 1] = p0
                cell[edge2cell[isNonDelaunayEdge, 1], 2] = p2

                NN = self.number_of_nodes()
                self.ds.reinit(NN, cell)

    def uniform_refine(self, n=1, surface=None, returnim=False):
        if returnim:
            nodeIMatrix = []
            cellIMatrix = []
        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            NE = self.number_of_edges()
            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')
            cell2edge = self.ds.cell_to_edge()
            edge2newNode = np.arange(NN, NN+NE)
            newNode = (node[edge[:,0],:] + node[edge[:,1],:])/2.0

            if returnim:
                A = coo_matrix((np.ones(NN), (range(NN), range(NN))), shape=(NN+NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5*np.ones(NE), (range(NN, NN+NE), edge[:, 0])), shape=(NN+NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5*np.ones(NE), (range(NN, NN+NE), edge[:, 1])), shape=(NN+NE, NN), dtype=self.ftype)
                nodeIMatrix.append(A.tocsr())
                B = eye(NC, dtype=self.ftype)
                B = bmat([[B], [B], [B], [B]])
                cellIMatrix.append(B.tocsr())

            if surface is not None:
                newNode, _ = surface.project(newNode)

            self.node = np.concatenate((node, newNode), axis=0)
            p = np.r_['-1', cell, edge2newNode[cell2edge]]
            cell = np.r_['0', p[:, [0, 5, 4]], p[:, [5, 1, 3]], p[:, [4, 3, 2]], p[:, [3, 4, 5]]]
            NN = self.node.shape[0]
            self.ds.reinit(NN, cell)
        if returnim:
            return nodeIMatrix, cellIMatrix

    def uniform_bisect(self, n=1):
        for i in range(n):
            self.bisect()

    def bisect_options(
            self,
            HB=None,
            IM=None,
            data=None,
            disp=True,
            ):

        options = {
                'HB': HB,
                'IM': IM,
                'data': data,
                'disp': disp
            }
        return options

    def bisect(self, isMarkedCell=None, options={'disp':True}):
        
        if options['disp']:
            print('Bisection begining......')

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NE = self.number_of_edges()

        if options['disp']:
            print('Current number of nodes:', NN)
            print('Current number of edges:', NE)
            print('Current number of cells:', NC)

        if isMarkedCell is None:
            isMarkedCell = np.ones(NC, dtype=np.bool)

        cell = self.entity('cell')
        edge = self.entity('edge')

        cell2edge = self.ds.cell_to_edge()
        cell2cell = self.ds.cell_to_cell()

        isCutEdge = np.zeros((NE,), dtype=np.bool)

        if options['disp']:
            print('The initial number of marked elements:', isMarkedCell.sum())

        markedCell, = np.nonzero(isMarkedCell)
        while len(markedCell)>0:
            isCutEdge[cell2edge[markedCell, 0]]=True
            refineNeighbor = cell2cell[markedCell, 0]
            markedCell = refineNeighbor[~isCutEdge[cell2edge[refineNeighbor,0]]]

        if options['disp']:
            print('The number of markedg edges: ', isCutEdge.sum())

        edge2newNode = np.zeros((NE,), dtype=self.itype)
        edge2newNode[isCutEdge] = np.arange(NN, NN+isCutEdge.sum())

        node = self.node
        newNode =0.5*(node[edge[isCutEdge,0],:] + node[edge[isCutEdge,1],:])
        self.node = np.concatenate((node, newNode), axis=0)
        cell2edge0 = cell2edge[:, 0]

        if 'IM' in options:
            nn = len(newNode)
            IM = coo_matrix((np.ones(NN), (np.arange(NN), np.arange(NN))),
                    shape=(NN+nn, NN), dtype=self.ftype)
            val = np.full(nn, 0.5)
            IM += coo_matrix(
                    (
                        val,
                        (
                            NN+np.arange(nn),
                            edge[isCutEdge, 0]
                        )
                    ), shape=(NN+nn, NN), dtype=self.ftype)
            IM += coo_matrix(
                    (
                        val,
                        (
                            NN+np.arange(nn),
                            edge[isCutEdge, 1]
                        )
                    ), shape=(NN+nn, NN), dtype=self.ftype)
            options['IM'] = IM.tocsr()

        if 'HB' in options:
            options['HB'] = np.arange(NC)

        for k in range(2):
            idx, = np.nonzero(edge2newNode[cell2edge0]>0)
            nc = len(idx)
            if nc == 0:
                break

            if 'HB' in options:
                HB = options['HB']
                options['HB'] = np.concatenate((HB, HB[idx]), axis=0)

            L = idx
            R = np.arange(NC, NC+nc)
            p0 = cell[idx,0]
            p1 = cell[idx,1]
            p2 = cell[idx,2]
            p3 = edge2newNode[cell2edge0[idx]]
            cell = np.concatenate((cell, np.zeros((nc,3), dtype=self.itype)), axis=0)
            cell[L,0] = p3
            cell[L,1] = p0
            cell[L,2] = p1
            cell[R,0] = p3
            cell[R,1] = p2
            cell[R,2] = p0
            if k == 0:
                cell2edge0 = np.zeros((NC+nc,), dtype=self.itype)
                cell2edge0[0:NC] = cell2edge[:,0]
                cell2edge0[L] = cell2edge[idx,2]
                cell2edge0[R] = cell2edge[idx,1]
            NC = NC+nc

        NN = self.node.shape[0]
        self.ds.reinit(NN, cell)

    def label(self, node=None, cell=None, cellidx=None):
        """单元顶点的重新排列，使得cell[:, [1, 2]] 存储了单元的最长边
        Parameter
        ---------

        Return
        ------
        cell ： in-place modify

        """

        rflag = False
        if node is None:
            node = self.entity('node')

        if cell is None:
            cell = self.entity('cell')
            rflag = True

        if cellidx is None:
            cellidx = np.arange(len(cell))

        NC = cellidx.shape[0]
        localEdge = self.ds.localEdge
        totalEdge = cell[cellidx][:, localEdge].reshape(
                -1, localEdge.shape[1])
        NE = totalEdge.shape[0]
        length = np.sum(
                (node[totalEdge[:, 1]] - node[totalEdge[:, 0]])**2,
                axis = -1)
        length += 0.1*np.random.rand(NE)*length
        cellEdgeLength = length.reshape(NC, 3)
        lidx = np.argmax(cellEdgeLength, axis=-1)

        flag = (lidx == 1)
        if  sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [1, 2, 0]]

        flag = (lidx == 2)
        if sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [2, 0, 1]]

        if rflag == True:
            self.ds.construct()

    def adaptive_options(
            self,
            method='mean',
            maxrefine=5,
            maxcoarsen=0,
            theta=1.0,
            HB=None,
            imatrix=False,
            data=None,
            disp=True,
            ):

        options = {
                'method': method,
                'maxrefine': maxrefine,
                'maxcoarsen': maxcoarsen,
                'theta': theta,
                'data': data,
                'HB': HB,
                'imatrix': imatrix,
                'disp': disp
            }
        return options


    def adaptive(self, eta, options):
        theta = options['theta']
        if options['method'] == 'mean':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.mean(eta)))
                )
        elif options['method'] == 'max':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.max(eta)))
                )
        elif options['method'] == 'median':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.median(eta)))
                )
        elif options['method'] == 'min':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.min(eta)))
                )
        else:
            raise ValueError(
                    "I don't know anyting about method %s!".format(options['method']))

        flag = options['numrefine'] > options['maxrefine']
        options['numrefine'][flag] = options['maxrefine']
        flag = options['numrefine'] < -options['maxcoarsen']
        options['numrefine'][flag] = -options['maxcoarsen']

        # refine
        NC = self.number_of_cells()
        print("Number of cells before:", NC)
        isMarkedCell = (options['numrefine'] > 0)
        while sum(isMarkedCell) > 0:
            self.bisect_1(isMarkedCell, options)
            print("Number of cells after refine:", self.number_of_cells())
            isMarkedCell = (options['numrefine'] > 0)

        # coarsen
        if options['maxcoarsen'] > 0:
            isMarkedCell = (options['numrefine'] < 0)
            while sum(isMarkedCell) > 0:
                NN0 = self.number_of_cells()
                self.coarsen_1(isMarkedCell, options)
                NN = self.number_of_cells()
                if NN == NN0:
                    break
                print("Number of cells after coarsen:", self.number_of_cells())
                isMarkedCell = (options['numrefine'] < 0)

    def bisect_1(self, isMarkedCell=None, options={'disp': True}):

        GD = self.geo_dimension()
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NN0 = NN  # 记录下二分加密之前的节点数目

        if isMarkedCell is None:
            # 默认加密所有的单元
            markedCell = np.arange(NC, dtype=self.itype)
        else:
            markedCell, = np.nonzero(isMarkedCell)

        # allocate new memory for node and cell
        node = np.zeros((5*NN, GD), dtype=self.ftype)
        cell = np.zeros((2*NC, 3), dtype=self.itype)

        if ('numrefine' in options) and (options['numrefine'] is not None):
            options['numrefine'] = np.r_[options['numrefine'], np.zeros(NC)]

        node[:NN] = self.entity('node')
        cell[:NC] = self.entity('cell')

        # 用于存储网格节点的代数，初始所有节点都为第 0 代
        generation = np.zeros(NN + 2*NC, dtype=np.uint8)

        # 用于记录被二分的边及其中点编号
        cutEdge = np.zeros((4*NN, 3), dtype=self.itype)

        # 当前的二分边的数目
        nCut = 0
        # 非协调边的标记数组 
        nonConforming = np.ones(4*NN, dtype=np.bool)
        while len(markedCell) != 0:
            # 标记最长边
            self.label(node, cell, markedCell)

            # 获取标记单元的四个顶点编号
            p0 = cell[markedCell, 0]
            p1 = cell[markedCell, 1]
            p2 = cell[markedCell, 2]

            # 找到新的二分边和新的中点 
            nMarked = len(markedCell)
            p3 = np.zeros(nMarked, dtype=self.itype)

            if nCut == 0: # 如果是第一次循环 
                idx = np.arange(nMarked) # cells introduce new cut edges
            else:
                # all non-conforming edges
                ncEdge = np.nonzero(nonConforming[:nCut])
                NE = len(ncEdge)
                I = cutEdge[ncEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[ncEdge][:, [0, 1]].reshape(-1)
                val = np.ones(len(I), dtype=np.bool)
                nv2v = csr_matrix(
                        (val, (I, J)),
                        shape=(NN, NN))
                i, j =  np.nonzero(nv2v[:, p1].multiply(nv2v[:, p2]))
                p3[j] = i
                idx, = np.nonzero(p3 == 0)

            if len(idx) != 0:
                # 把需要二分的边唯一化 
                NE = len(idx)
                cellCutEdge = np.array([p1[idx], p2[idx]])
                cellCutEdge.sort(axis=0)
                s = csr_matrix(
                    (
                        np.ones(NE, dtype=np.bool),
                        (
                            cellCutEdge[0, :],
                            cellCutEdge[1, :]
                        )
                    ), shape=(NN, NN))
                # 获得唯一的边 
                i, j = s.nonzero()
                nNew = len(i)
                newCutEdge = np.arange(nCut, nCut+nNew)
                cutEdge[newCutEdge, 0] = i
                cutEdge[newCutEdge, 1] = j
                cutEdge[newCutEdge, 2] = range(NN, NN+nNew)
                node[NN:NN+nNew, :] = (node[i, :] + node[j, :])/2.0
                nCut += nNew
                NN += nNew

                # 新点和旧点的邻接矩阵 
                I = cutEdge[newCutEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[newCutEdge][:, [0, 1]].reshape(-1)
                val = np.ones(len(I), dtype=np.bool)
                nv2v = csr_matrix(
                        (val, (I, J)),
                        shape=(NN, NN))
                i, j =  np.nonzero(nv2v[:, p1].multiply(nv2v[:, p2]))
                p3[j] = i

            # 如果新点的代数仍然为 0
            idx = (generation[p3] == 0)
            cellGeneration = np.max(
                    generation[cell[markedCell[idx]]],
                    axis=-1)
            # 第几代点 
            generation[p3[idx]] = cellGeneration + 1
            cell[markedCell, 0] = p3
            cell[markedCell, 1] = p0
            cell[markedCell, 2] = p1
            cell[NC:NC+nMarked, 0] = p3
            cell[NC:NC+nMarked, 1] = p2
            cell[NC:NC+nMarked, 2] = p0

            if ('numrefine' in options) and (options['numrefine'] is not None):
                options['numrefine'][markedCell] -= 1
                options['numrefine'][NC:NC+nMarked] = options['numrefine'][markedCell]

            NC = NC + nMarked
            del cellGeneration, p0, p1, p2, p3

            # 找到非协调的单元 
            checkEdge, = np.nonzero(nonConforming[:nCut])
            isCheckNode = np.zeros(NN, dtype=np.bool)
            isCheckNode[cutEdge[checkEdge]] = True
            isCheckCell = np.sum(
                    isCheckNode[cell[:NC]],
                    axis= -1) > 0
            # 找到所有包含检查节点的单元编号 
            checkCell, = np.nonzero(isCheckCell)
            I = np.repeat(checkCell, 3)
            J = cell[checkCell].reshape(-1)
            val = np.ones(len(I), dtype=np.bool)
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN))
            i, j = np.nonzero(
                    cell2node[:, cutEdge[checkEdge, 0]].multiply(
                        cell2node[:, cutEdge[checkEdge, 1]]
                        ))
            markedCell = np.unique(i)
            nonConforming[checkEdge] = False
            nonConforming[checkEdge[j]] = True;

        if ('imatrix' in options) and (options['imatrix'] is True):
            nn = NN - NN0
            IM = coo_matrix(
                    (
                        np.ones(NN0),
                        (
                            np.arange(NN0),
                            np.arange(NN0)
                        )
                    ), shape=(NN, NN), dtype=self.ftype)
            cutEdge = cutEdge[:nn]
            val = np.full((nn, 2), 0.5, dtype=self.ftype)

            g = 2
            markedNode, = np.nonzero(generation == g)

            N = len(markedNode)
            while N != 0:
                nidx = markedNode - NN0
                i = cutEdge[nidx, 0]
                j = cutEdge[nidx, 1]
                ic = np.zeros((N, 2), dtype=self.ftype)
                jc = np.zeros((N, 2), dtype=self.ftype)
                ic[i < NN0, 0] = 1.0
                jc[j < NN0, 1] = 1.0
                ic[i >= NN0, :] = val[i[i >= NN0] - NN0, :]
                jc[j >= NN0, :] = val[j[j >= NN0] - NN0, :]
                val[markedNode - NN0, :] = 0.5*(ic + jc)
                cutEdge[nidx[i >= NN0], 0] = cutEdge[i[i >= NN0] - NN0, 0]
                cutEdge[nidx[j >= NN0], 1] = cutEdge[j[j >= NN0] - NN0, 1]
                g += 1
                markedNode, = np.nonzero(generation == g)
                N = len(markedNode)

            IM += coo_matrix(
                    (
                        val.flat,
                        (
                            cutEdge[:, [2, 2]].flat,
                            cutEdge[:, [0, 1]].flat
                        )
                    ), shape=(NN, NN0), dtype=self.ftype)
            options['imatrix'] = IM.tocsr()

        self.node = node[:NN]
        cell = cell[:NC]
        self.ds.reinit(NN, cell)


    def coarsen_1(self, isMarkedCell=None, options=None):
        pass


    def linear_stiff_matrix(self, c=None):
        """
        Notes
        -----
        线性元的刚度矩阵
        """

        NN = self.number_of_nodes()

        area = self.cell_area()
        gphi = self.grad_lambda()

        if callable(c):
            bc = np.array([1/3, 1/3, 1/3], dtype=self.ftype) 
            if c.coordtype == 'cartesian':
                ps = self.bc_to_point(bc)
                c = c(ps)
            elif c.coordtype == 'barycentric':
                c = c(bc)
        
        if c is not None:
            area *= c

        A = gphi@gphi.swapaxes(-1, -2)
        A *= area[:, None, None]

        cell = self.entity('cell')
        I = np.broadcast_to(cell[:, :, None], shape=A.shape)
        J = np.broadcast_to(cell[:, None, :], shape=A.shape)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(NN, NN))
        return A

    def grad_lambda(self):
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells()
        v0 = node[cell[:, 2], :] - node[cell[:, 1], :]
        v1 = node[cell[:, 0], :] - node[cell[:, 2], :]
        v2 = node[cell[:, 1], :] - node[cell[:, 0], :]
        GD = self.geo_dimension()
        nv = np.cross(v2, -v1)
        Dlambda = np.zeros((NC, 3, GD), dtype=self.ftype)
        if GD == 2:
            length = nv
            W = np.array([[0, 1], [-1, 0]])
            Dlambda[:,0,:] = v0@W/length.reshape((-1, 1))
            Dlambda[:,1,:] = v1@W/length.reshape((-1, 1))
            Dlambda[:,2,:] = v2@W/length.reshape((-1, 1))
        elif GD == 3:
            length = np.sqrt(np.square(nv).sum(axis=1))
            n = nv/length.reshape((-1, 1))
            Dlambda[:,0,:] = np.cross(n, v0)/length.reshape((-1,1))
            Dlambda[:,1,:] = np.cross(n, v1)/length.reshape((-1,1))
            Dlambda[:,2,:] = np.cross(n, v2)/length.reshape((-1,1))
        return Dlambda

    def jacobi_matrix(self, index=np.s_[:]):
        """
        Return
        ------
        J : numpy.ndarray
            `J` is the transpose o  jacobi matrix of each cell.
            The shape of `J` is  `(NC, 2, 2)` or `(NC, 2, 3)`
        """
        node = self.node
        cell = self.ds.cell
        J = node[cell[index, [1, 2]]] - node[cell[index, [0]]]
        return J

    def rot_lambda(self):
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells()
        v0 = node[cell[:, 2], :] - node[cell[:, 1], :]
        v1 = node[cell[:, 0], :] - node[cell[:, 2], :]
        v2 = node[cell[:, 1], :] - node[cell[:, 0], :]
        GD = self.geo_dimension()
        nv = np.cross(v2, -v1)
        Rlambda = np.zeros((NC, 3, GD), dtype=self.ftype)
        if GD == 2:
            length = nv
            Rlambda[:,0,:] = v0/length.reshape((-1, 1))
            Rlambda[:,1,:] = v1/length.reshape((-1, 1))
            Rlambda[:,2,:] = v2/length.reshape((-1, 1))
        elif GD == 3:
            length = np.sqrt(np.square(nv).sum(axis=1))
            Rlambda[:,0,:] = v0/length.reshape((-1, 1))
            Rlambda[:,1,:] = v1/length.reshape((-1, 1))
            Rlambda[:,2,:] = v2/length.reshape((-1, 1))
        return Rlambda

    def cell_area(self, index=np.s_[:]):
        node = self.node
        cell = self.ds.cell
        GD = self.geo_dimension()
        v1 = node[cell[index, 1], :] - node[cell[index, 0], :]
        v2 = node[cell[index, 2], :] - node[cell[index, 0], :]
        nv = np.cross(v2, -v1)
        if GD == 2:
            a = nv/2.0
        elif GD == 3:
            a = np.sqrt(np.square(nv).sum(axis=1))/2.0
        return a

    def bc_to_point(self, bc, etype='cell', index=np.s_[:]):
        """

        Notes
        ----

        etype 是一个多余的参数， bc 中已经包含这单元类型的信息,
        为了向后兼容这里保留下来
        """
        TD = bc.shape[-1] - 1 #
        node = self.node
        entity = self.entity(etype=TD)[index]
        p = np.einsum('...j, ijk->...ik', bc, node[entity])
        return p

    def edge_bc_to_point(self, bcs, index=None):
        node = self.entity('node')
        edge = self.entity('edge')
        index = index if index is not None else np.s_[:]
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge[index]])
        return ps



class TriangleMeshWithInfinityNode:
    def __init__(self, mesh):
        edge = mesh.ds.edge
        bdEdgeIdx = mesh.ds.boundary_edge_index()
        NBE = len(bdEdgeIdx)
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()

        self.itype = mesh.itype
        self.ftype = mesh.ftype

        newCell = np.zeros((NC + NBE, 3), dtype=self.itype)
        newCell[:NC, :] = mesh.ds.cell
        newCell[NC:, 0] = NN
        newCell[NC:, 1:3] = edge[bdEdgeIdx, 1::-1]

        node = mesh.node
        self.node = np.append(node, [[np.nan, np.nan]], axis=0)
        self.ds = TriangleMeshDataStructure(NN+1, newCell)
        self.center = np.append(mesh.entity_barycenter(),
                0.5*(node[edge[bdEdgeIdx, 0], :] + node[edge[bdEdgeIdx, 1], :]), axis=0)
        self.meshtype = 'tri'

    def number_of_nodes(self):
        return self.node.shape[0]

    def number_of_edges(self):
        return self.ds.NE

    def number_of_faces(self):
        return self.ds.NC

    def number_of_cells(self):
        return self.ds.NC

    def is_infinity_cell(self):
        N = self.number_of_nodes()
        cell = self.ds.cell
        return cell[:, 0] == N-1

    def is_boundary_edge(self):
        NE = self.number_of_edges()
        cell2edge = self.ds.cell_to_edge()
        isInfCell = self.is_infinity_cell()
        isBdEdge = np.zeros(NE, dtype=np.bool)
        isBdEdge[cell2edge[isInfCell, 0]] = True
        return isBdEdge

    def is_boundary_node(self):
        N = self.number_of_nodes()
        edge = self.ds.edge
        isBdEdge = self.is_boundary_edge()
        isBdNode = np.zeros(N, dtype=np.bool)
        isBdNode[edge[isBdEdge, :]] = True
        return isBdNode

    def to_polygonmesh(self):
        """

        Notes
        -----
        把一个三角形网格转化为多边形网格。
        """
        isBdNode = self.is_boundary_node()
        NB = isBdNode.sum()

        nodeIdxMap = np.zeros(isBdNode.shape, dtype=self.itype)
        nodeIdxMap[isBdNode] = self.center.shape[0] + np.arange(NB)

        pnode = np.concatenate((self.center, self.node[isBdNode]), axis=0)
        PN = pnode.shape[0]

        node2cell = self.ds.node_to_cell(localidx=True)
        NV = np.asarray((node2cell > 0).sum(axis=1)).reshape(-1)
        NV[isBdNode] += 1
        NV = NV[:-1]
        
        PNC = len(NV)
        pcell = np.zeros(NV.sum(), dtype=self.itype)
        pcellLocation = np.zeros(PNC+1, dtype=self.itype)
        pcellLocation[1:] = np.cumsum(NV)


        isBdEdge = self.is_boundary_edge()
        NC = self.number_of_cells() - isBdEdge.sum()
        cell = self.ds.cell
        currentCellIdx = np.zeros(PNC, dtype=self.itype)
        currentCellIdx[cell[:NC, 0]] = range(NC)
        currentCellIdx[cell[:NC, 1]] = range(NC)
        currentCellIdx[cell[:NC, 2]] = range(NC)
        pcell[pcellLocation[:-1]] = currentCellIdx 

        currentIdx = pcellLocation[:-1]
        N = self.number_of_nodes() - 1
        currentNodeIdx = np.arange(N, dtype=self.itype)
        endIdx = pcellLocation[1:]
        cell2cell = self.ds.cell_to_cell()
        isInfCell = self.is_infinity_cell()
        pnext = np.array([1, 2, 0], dtype=self.itype)
        while True:
            isNotOK = (currentIdx + 1) < endIdx
            currentIdx = currentIdx[isNotOK]
            currentNodeIdx = currentNodeIdx[isNotOK]
            currentCellIdx = pcell[currentIdx]
            endIdx = endIdx[isNotOK]
            if len(currentIdx) == 0:
                break
            localIdx = np.asarray(node2cell[currentNodeIdx, currentCellIdx]) - 1
            cellIdx = np.asarray(cell2cell[currentCellIdx, pnext[localIdx]]).reshape(-1)
            isBdCase = isInfCell[currentCellIdx] & isInfCell[cellIdx]
            if np.any(isBdCase):
                pcell[currentIdx[isBdCase] + 1] = nodeIdxMap[currentNodeIdx[isBdCase]]
                currentIdx[isBdCase] += 1
            pcell[currentIdx + 1] = cellIdx
            currentIdx += 1

        return pnode, pcell, pcellLocation

