import numpy as np
import warnings
from scipy.sparse import coo_matrix, csr_matrix, bmat, eye
from scipy.spatial import KDTree
from .Mesh2d import Mesh2d, Mesh2dDataStructure
from ...quadrature import TriangleQuadrature
from ...quadrature import GaussLegendreQuadrature
from ..triangle_quality import *

class TriangleMeshDataStructure(Mesh2dDataStructure):

    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    localFace = np.array([(1, 2), (2, 0), (0, 1)])
    ccw = np.array([0, 1, 2])

    NVC = 3
    NVE = 2
    NVF = 2

    NEC = 3
    NFC = 3

    localCell = np.array([
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1)])

    def __init__(self, NN, cell):
        super().__init__(NN,cell)

## @defgroup MeshGenerators TriangleMesh Common Region Mesh Generators
## @defgroup MeshQuality
class TriangleMesh(Mesh2d):
    def __init__(self, node, cell):
        """
        @brief TriangleMesh 对象的初始化函数

        """

        assert cell.shape[-1] == 3

        self.node = node
        NN = node.shape[0]
        self.ds = TriangleMeshDataStructure(NN, cell)

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

    def geo_dimension(self):
        return self.node.shape[-1]

    def integrator(self, q, etype='cell'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 2}:
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            return GaussLegendreQuadrature(q)

    def bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把重心坐标积分点变换到实际网格实体上的笛卡尔坐标点
        """
        TD = bc.shape[-1] - 1 # bc.shape == (NQ, TD+1)
        node = self.node
        entity = self.entity(etype=TD)[index]
        p = np.einsum('...j, ijk->...ik', bc, node[entity])
        return p

    def point_to_bc(self, point):
        i_cell = self.location(point)
        node = self.node
        cell = self.entity('cell')
        i_cellmeasure = self.cell_area()[i_cell]
        i_cell = node[cell[i_cell]]
        point = point[:,np.newaxis, :]
        v = i_cell - point
        area0 = 0.5* np.abs(np.cross(v[:, 1, :], v[:, 2, :]))
        area1 = 0.5* np.abs(np.cross(v[:, 0, :], v[:, 2, :]))
        area2 = 0.5* np.abs(np.cross(v[:, 0, :], v[:, 1, :]))
        result = np.zeros((i_cell.shape[0], 3))
        result[:, 0] = area0/i_cellmeasure
        result[:, 1] = area1/i_cellmeasure
        result[:, 2] = area2/i_cellmeasure
        return result

    def edge_bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把重心坐标积分点变换到实际网格边上的笛卡尔坐标点
        """
        node = self.node
        entity = self.entity('edge')[index]
        p = np.einsum('...j, ijk->...ik', bc, node[entity])
        return p

    def cell_bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把重心坐标积分点变换到实际网格单元上的笛卡尔坐标点
        """
        node = self.node
        entity = self.entity('cell')[index]
        p = np.einsum('...j, ijk->...ik', bc, node[entity])
        return p



    def shape_function(self, bc, p=1, etype='cell'):
        """
        @brief
        """
        if p==1:
            return bc

        TD = bc.shape[-1] - 1
        multiIndex = self.multi_index_matrix(p, etype=etype)
        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    def grad_shape_function(self, bc, p=1, index=np.s_[:], variables='x'):
        """
        @note 注意这里调用的实际上不是形状函数的梯度，而是网格空间基函数的梯度
        """
        TD = self.top_dimension()
        multiIndex = self.multi_index_matrix(p)

        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = self.number_of_local_ipoints(p)
        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        if variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = np.einsum('...ij, kjm->...kim', R, Dlambda)
            return gphi #(..., NC, ldof, GD)
        elif variables == 'u':
            return R

    def grad_shape_function_on_edge(self, bc, cindex, lidx, p=1, direction=True):
        """
        @brief 计算单元上所有形函数在边上的积分点处的导函数值

        @brief bc 边上的一组积分点
        @brief cindex 边所在的单元编号
        @brief lidx 边在该单元的局部编号
        @brief direction  True 表示边的方向和单元的逆时针方向一致，False 表示不一致 
        """

        NC = len(cindex)
        nmap = np.array([1, 2, 0])
        pmap = np.array([2, 0, 1])
        shape = (NC, ) + bc.shape[0:-1] + (3, )
        bcs = np.zeros(shape, dtype=self.ftype)  # (NE, 3) or (NE, NQ, 3)
        idx = np.arange(NC)
        if direction:
            bcs[idx, ..., nmap[lidx]] = bc[..., 0]
            bcs[idx, ..., pmap[lidx]] = bc[..., 1]
        else:
            bcs[idx, ..., nmap[lidx]] = bc[..., 1]
            bcs[idx, ..., pmap[lidx]] = bc[..., 0]

        gphi = self.grad_shape_function(bcs, p=p, index=cindex, variables='x')

        return gphi 

    grad_shape_function_on_face = grad_shape_function_on_edge

    def grad_lambda(self, index=np.s_[:]):
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells() if index == np.s_[:] else len(index)
        v0 = node[cell[index, 2]] - node[cell[index, 1]]
        v1 = node[cell[index, 0]] - node[cell[index, 2]]
        v2 = node[cell[index, 1]] - node[cell[index, 0]]
        GD = self.geo_dimension()
        nv = np.cross(v1, v2)
        Dlambda = np.zeros((NC, 3, GD), dtype=self.ftype)
        if GD == 2:
            length = nv
            W = np.array([[0, 1], [-1, 0]])
            Dlambda[:, 0] = v0@W/length[:, None]
            Dlambda[:, 1] = v1@W/length[:, None]
            Dlambda[:, 2] = v2@W/length[:, None]
        elif GD == 3:
            length = np.linalg.norm(nv, axis=-1, keepdims=True)
            n = nv/length
            Dlambda[:, 0] = np.cross(n, v0)/length[:, None]
            Dlambda[:, 1] = np.cross(n, v1)/length[:, None]
            Dlambda[:, 2] = np.cross(n, v2)/length[:, None]
        return Dlambda

    def rot_lambda(self, index=np.s_[:]):
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells() if index == np.s_[:] else len(index)
        v0 = node[cell[index, 2], :] - node[cell[index, 1], :]
        v1 = node[cell[index, 0], :] - node[cell[index, 2], :]
        v2 = node[cell[index, 1], :] - node[cell[index, 0], :]
        GD = self.geo_dimension()
        nv = np.cross(v2, -v1)
        Rlambda = np.zeros((NC, 3, GD), dtype=self.ftype)
        if GD == 2:
            length = nv
        elif GD == 3:
            length = np.sqrt(np.sum(nv**2, axis=-1))

        Rlambda[:,0,:] = v0/length.reshape((-1, 1))
        Rlambda[:,1,:] = v1/length.reshape((-1, 1))
        Rlambda[:,2,:] = v2/length.reshape((-1, 1))

        return Rlambda

    def uniform_refine(self, n=1, surface=None, interface=None, returnim=False):
        """
        @brief 一致加密三角形网格
        """

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

            if interface is not None:
                for key, levelset in interface:
                    isInterfaceEdge = self.edgedata[key]
                    p = newNode[isInterfaceEdge]
                    levelset.project(p)
                    newNode[isInterfaceEdge] = p


            self.node = np.concatenate((node, newNode), axis=0)
            p = np.r_['-1', cell, edge2newNode[cell2edge]]
            cell = np.r_['0', p[:, [0, 5, 4]], p[:, [5, 1, 3]], p[:, [4, 3, 2]], p[:, [3, 4, 5]]]
            self.ds.reinit(NN+NE, cell)

        if returnim:
            return nodeIMatrix, cellIMatrix

    def multi_index_matrix(self, p, etype=2):
        """
        @brief 获取三角形上的 p 次的多重指标矩阵

        @param[in] p positive integer

        @return multiIndex  ndarray with shape (ldof, 3)
        """
        if etype in {'cell', 2}:
            ldof = (p+1)*(p+2)//2
            idx = np.arange(0, ldof)
            idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
            multiIndex = np.zeros((ldof, 3), dtype=np.int_)
            multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
            multiIndex[:,1] = idx0 - multiIndex[:,2]
            multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
            return multiIndex
        elif etype in {'face', 'edge', 1}:
            ldof = p+1
            multiIndex = np.zeros((ldof, 2), dtype=np.int_)
            multiIndex[:, 0] = np.arange(p, -1, -1)
            multiIndex[:, 1] = p - multiIndex[:, 0]
            return multiIndex

    def number_of_local_ipoints(self, p, iptype='cell'):
        if iptype in {'cell', 2}:
            return (p+1)*(p+2)//2
        elif iptype in {'face', 'edge',  1}:
            return self.p + 1
        elif iptype in {'node', 0}:
            return 1

    def number_of_global_ipoints(self, p):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        return NN + (p-1)*NE + (p-2)*(p-1)//2*NC

    def interpolation_points(self, p: int, index=np.s_[:]):
        """
        @brief 获取三角形网格上所有 p 次插值点
        TODO index
        """
        cell = self.entity('cell')
        node = self.entity('node')
        if p == 1:
            return node
        if p > 1:
            NN = self.number_of_nodes()
            GD = self.geo_dimension()

            gdof = self.number_of_global_ipoints(p)
            ipoints = np.zeros((gdof, GD), dtype=self.ftype)
            ipoints[:NN, :] = node

            NE = self.number_of_edges()

            edge = self.entity('edge')

            w = np.zeros((p-1, 2), dtype=np.float64)
            w[:, 0] = np.arange(p-1, 0, -1)/p
            w[:, 1] = w[-1::-1, 0]
            ipoints[NN:NN+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', w,
                    node[edge,:]).reshape(-1, GD)
        if p > 2:
            multiIndex = self.multi_index_matrix(p, 'cell')
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:,0] | isEdgeIPoints[:,1] |
                    isEdgeIPoints[:,2])
            w = multiIndex[isInCellIPoints, :]/p
            ipoints[NN+(p-1)*NE:, :] = np.einsum('ij, kj...->ki...', w,
                    node[cell, :]).reshape(-1, GD)
        return ipoints


    def edge_to_ipoint(self, p, index=np.s_[:]):
        """
        @brief 获取网格边与插值点的对应关系
        """
        if isinstance(index, slice) and index == slice(None):
            NE = self.number_of_edges()
            index = np.arange(NE)
        elif isinstance(index, np.ndarray) and (index.dtype == np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        elif isinstance(index, list) and (type(index[0]) is np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        else:
            NE = len(index)

        NN = self.number_of_nodes()

        edge = self.entity('edge', index=index)
        edge2ipoints = np.zeros((NE, p+1), dtype=self.itype)
        edge2ipoints[:, [0, -1]] = edge
        if p > 1:
            idx = NN + np.arange(p-1)
            edge2ipoints[:, 1:-1] =  (p-1)*index[:, None] + idx 
        return edge2ipoints

    face_to_ipoint = edge_to_ipoint

    def cell_to_ipoint(self, p, index=np.s_[:]):
        """
        """
        cell = self.entity('cell')
        if p==1:
            return cell[index] 

        mi = self.multi_index_matrix(p)
        idx0, = np.nonzero(mi[:, 0] == 0)
        idx1, = np.nonzero(mi[:, 1] == 0)
        idx2, = np.nonzero(mi[:, 2] == 0)

        edge2cell = self.ds.edge_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells() 

        e2p = self.edge_to_ipoint(p)
        ldof = self.number_of_local_ipoints(p)
        c2p = np.zeros((NC, ldof), dtype=self.itype)

        flag = edge2cell[:, 2] == 0
        c2p[edge2cell[flag, 0][:, None], idx0] = e2p[flag]

        flag = edge2cell[:, 2] == 1
        c2p[edge2cell[flag, 0][:, None], idx1[-1::-1]] = e2p[flag]

        flag = edge2cell[:, 2] == 2
        c2p[edge2cell[flag, 0][:, None], idx2] = e2p[flag]


        iflag = edge2cell[:, 0] != edge2cell[:, 1]

        flag = iflag & (edge2cell[:, 3] == 0)
        c2p[edge2cell[flag, 1][:, None], idx0[-1::-1]] = e2p[flag]

        flag = iflag & (edge2cell[:, 3] == 1)
        c2p[edge2cell[flag, 1][:, None], idx1] = e2p[flag]

        flag = iflag & (edge2cell[:, 3] == 2)
        c2p[edge2cell[flag, 1][:, None], idx2[-1::-1]] = e2p[flag]

        cdof = (p-1)*(p-2)//2
        flag = np.sum(mi > 0, axis=1) == 3
        c2p[:, flag] = NN + NE*(p-1) + np.arange(NC*cdof).reshape(NC, cdof)
        return c2p[index]

    def _cell_to_ipoint(self, p, index=np.s_[:]):
        """
        @brief 获取网格中的三角形单元与插值点的对应关系
        @todo 只获取一部分单元的插值点全局编号
        """
        cell = self.entity('cell', index=index)
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        ldof = self.number_of_local_ipoints(p)

        if p == 1:
            cell2ipoint = cell

        if p > 1:
            cell2ipoint = np.zeros((NC, ldof), dtype=self.itype)

            isEdgeIPoint = self.multi_index_matrix(p, 'cell') == 0
            edge2ipoint = self.edge_to_ipoint(p)


            cell2edgeSign = self.ds.cell_to_edge_sign()
            cell2edge = self.ds.cell_to_edge()

            cell2ipoint[np.ix_(cell2edgeSign[:, 0], isEdgeIPoint[:, 0])] = \
                    edge2ipoint[cell2edge[cell2edgeSign[:, 0], [0]], :]
            cell2ipoint[np.ix_(~cell2edgeSign[:, 0], isEdgeIPoint[:,0])] = \
                    edge2ipoint[cell2edge[~cell2edgeSign[:, 0], [0]], -1::-1]

            cell2ipoint[np.ix_(cell2edgeSign[:, 1], isEdgeIPoint[:, 1])] = \
                    edge2ipoint[cell2edge[cell2edgeSign[:, 1], [1]], -1::-1]
            cell2ipoint[np.ix_(~cell2edgeSign[:, 1], isEdgeIPoint[:,1])] = \
                    edge2ipoint[cell2edge[~cell2edgeSign[:, 1], [1]], :]

            cell2ipoint[np.ix_(cell2edgeSign[:, 2], isEdgeIPoint[:, 2])] = \
                    edge2ipoint[cell2edge[cell2edgeSign[:, 2], [2]], :]
            cell2ipoint[np.ix_(~cell2edgeSign[:, 2], isEdgeIPoint[:,2])] = \
                    edge2ipoint[cell2edge[~cell2edgeSign[:, 2], [2]], -1::-1]
        if p > 2:
            base = NN + (p-1)*NE
            isInCellIPoint = ~(isEdgeIPoint[:,0] | isEdgeIPoint[:,1] | isEdgeIPoint[:,2])
            idof = ldof - 3*p
            cell2ipoint[:, isInCellIPoint] = base + np.arange(NC*idof).reshape(NC, idof)

        return cell2ipoint


    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_TRIANGLE = 5
            return VTK_TRIANGLE
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
        """
        @brief 把网格转化为 vtk 的数据格式
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

    def number_of_corner_nodes(self):
        return self.ds.NN

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

        isNotOK = np.ones(NP, dtype=np.bool_)
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

    def circumcenter(self, index=np.s_[:], returnradius=False):
        """
        @brief 计算三角形外接圆的圆心和半径
        """
        node = self.node
        cell = self.ds.cell
        GD = self.geo_dimension()

        v0 = node[cell[index, 2], :] - node[cell[index, 1], :]
        v1 = node[cell[index, 0], :] - node[cell[index, 2], :]
        v2 = node[cell[index, 1], :] - node[cell[index, 0], :]
        nv = np.cross(v2, -v1)
        if GD == 2:
            area = nv/2.0
            x2 = np.sum(node**2, axis=1, keepdims=True)
            w0 = x2[cell[index, 2]] + x2[cell[index, 1]]
            w1 = x2[cell[index, 0]] + x2[cell[index, 2]]
            w2 = x2[cell[index, 1]] + x2[cell[index, 0]]
            W = np.array([[0, -1],[1, 0]], dtype=self.ftype)
            fe0 = w0*v0@W
            fe1 = w1*v1@W
            fe2 = w2*v2@W
            c = 0.25*(fe0 + fe1 + fe2)/area.reshape(-1,1)
            R = np.sqrt(np.sum((c-node[cell[index, 0], :])**2,axis=1))
        elif GD == 3:
            length = np.sqrt(np.sum(nv**2, axis=1))
            n = nv/length.reshape((-1, 1))
            l02 = np.sum(v1**2, axis=1, keepdims=True)
            l01 = np.sum(v2**2, axis=1, keepdims=True)
            d = 0.5*(l02*np.cross(n, v2) + l01*np.cross(-v1, n))/length.reshape(-1, 1)
            c = node[cell[index, 0]] + d
            R = np.sqrt(np.sum(d**2, axis=1))

        if returnradius:
            return c, R
        else:
            return c

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

    def show_angle(self, axes, angle=None):
        """
        @brief 显示网格角度的分布直方图
        """
        if angle is None:
            angle = self.angle() 
        hist, bins = np.histogram(angle.flatten('F')*180/np.pi, bins=50, range=(0, 180))
        center = (bins[:-1] + bins[1:])/2
        axes.bar(center, hist, align='center', width=180/50.0)
        axes.set_xlim(0, 180)
        mina = np.min(angle.flatten('F')*180/np.pi)
        maxa = np.max(angle.flatten('F')*180/np.pi)
        meana = np.mean(angle.flatten('F')*180/np.pi)
        axes.annotate('Min angle: {:.4}'.format(mina), xy=(0.41, 0.5),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        axes.annotate('Max angle: {:.4}'.format(maxa), xy=(0.41, 0.45),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        axes.annotate('Average angle: {:.4}'.format(meana), xy=(0.41, 0.40),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        return mina, maxa, meana
    
    def cell_quality(self,measure='radius_ratio'):
        if measure=='radius_ratio':
            return radius_ratio(self) 

    def show_quality(self, axes, qtype=None, quality=None):
        """
        @brief 显示网格质量分布的分布直方图
        """
        if quality is None:
            quality = self.cell_quality() 
        minq = np.min(quality)
        maxq = np.max(quality)
        meanq = np.mean(quality)
        hist, bins = np.histogram(quality, bins=50, range=(0, 1))
        center = (bins[:-1] + bins[1:]) / 2
        axes.bar(center, hist, align='center', width=0.02)
        axes.set_xlim(0, 1)
        axes.annotate('Min quality: {:.6}'.format(minq), xy=(0,0),xytext=(0.1, 0.5),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        axes.annotate('Max quality: {:.6}'.format(maxq), xy=(0,0),xytext=(0.1, 0.45),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        axes.annotate('Average quality: {:.6}'.format(meanq), xy=(0,0),xytext=(0.1, 0.40),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        return minq, maxq, meanq


    def edge_swap(self):
        while True:
            # Construct necessary data structure
            edge2cell = self.ds.edge_to_cell()
            cell2edge = self.ds.cell_to_edge()

            # Find non-Delaunay edges
            angle = self.angle()
            asum = np.sum(angle[edge2cell[:, 0:2], edge2cell[:, 2:4]], axis=1)
            isNonDelaunayEdge = (asum > np.pi) & (edge2cell[:,0] != edge2cell[:,1])

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

    def odt_iterate(self):
        node = self.node.copy()
        cell = self.ds.cell
        alpha = 1

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        
        isBdNode = self.ds.boundary_node_flag()
        isBdCell = self.ds.boundary_cell_flag()
        isFreeNode = ~isBdNode

        cm = self.entity_measure("cell") # 单元面积
        cb = self.entity_barycenter('cell') # 单元重心
        cc = self.circumcenter() # 单元外心
        cc[isBdCell] = cb[isBdCell]
        cc = cm[...,None]*cc

        newNode = np.zeros((NN,2),dtype=np.float64)
        patch_area = np.zeros(NN,dtype=np.float64)
        
        np.add.at(newNode,cell,np.broadcast_to(cc[:,None],(NC,3,2)))
        np.add.at(patch_area,cell,np.broadcast_to(cm[:,None],(NC,3))) 
        
        newNode[isBdNode] = node[isBdNode]
        newNode[isFreeNode] = newNode[isFreeNode]/patch_area[...,None][isFreeNode]
        self.node = newNode
        while np.sum(cm<=0)>0:
            self.node[isFreeNode] = (1-alpha/2)*node[isFreeNode]+alpha*newNode[isFreeNode]
        self.edge_swap()

    def cpt_iterate(self):
        node = self.node.copy()
        cell = self.ds.cell
        alpha = 1

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        
        isBdNode = self.ds.boundary_node_flag()
        isBdCell = self.ds.boundary_cell_flag()
        isFreeNode = ~isBdNode

        cm = self.entity_measure("cell") # 单元面积
        cb = self.entity_barycenter('cell') # 单元重心
        cb = cm[...,None]*cb

        newNode = np.zeros((NN,2),dtype=np.float64)
        patch_area = np.zeros(NN,dtype=np.float64)
        
        np.add.at(newNode,cell,np.broadcast_to(cb[:,None],(NC,3,2)))
        np.add.at(patch_area,cell,np.broadcast_to(cm[:,None],(NC,3))) 
        
        newNode[isBdNode] = node[isBdNode]
        newNode[isFreeNode] = newNode[isFreeNode]/patch_area[...,None][isFreeNode]
        self.node = newNode
        while np.sum(cm<=0)>0:
            self.node[isFreeNode] = (1-alpha/2)*node[isFreeNode]+alpha*newNode[isFreeNode]
        self.edge_swap()

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
            isMarkedCell = np.ones(NC, dtype=np.bool_)

        cell = self.entity('cell')
        edge = self.entity('edge')

        cell2edge = self.ds.cell_to_edge()
        cell2cell = self.ds.cell_to_cell()

        isCutEdge = np.zeros((NE,), dtype=np.bool_)

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

        if 'data' in options:
            pass

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

            if ('data' in options) and (options['data'] is not None):
                for key, value in options['data'].items():
                    if len(value.shape) == 1: # 分片常数
                        value = np.r_[value, value[idx]]
                        options[key] = value
                    else:
                        ldof = value.shape[-1]
                        p = int((np.sqrt(1+8*ldof)-3)//2)
                        bc = self.multi_index_matrix(p, etype='cell')/p

                        bcl = np.zeros_like(bc)
                        bcl[:, 0] = bc[:, 1]
                        bcl[:, 1] = 1/2*bc[:, 0] + bc[:,2]
                        bcl[:, 2] = 1/2*bc[:, 0]

                        bcr = np.zeros_like(bc)
                        bcr[:, 0] = bc[:, 2]
                        bcr[:, 1] = 1/2*bc[:, 0]
                        bcr[:, 2] = 1/2*bc[:, 0] + bc[:, 1]

                        value = np.r_['0', value, np.zeros((nc, ldof), dtype=self.ftype)]

                        phi = self.shape_function(bcr, p=p)
                        value[NC:, :] = np.einsum('cj,kj->ck', value[idx], phi)

                        phi = self.shape_function(bcl, p=p)
                        value[idx, :] = np.einsum('cj,kj->ck', value[idx], phi)

                        options['data'][key] = value



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

    def coarsen(self, isMarkedCell=None, options={}):
        """
        @brief

        https://lyc102.github.io/ifem/afem/coarsen/
        """

        if isMarkedCell is None:
            return


        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        cell = self.entity('cell')
        node = self.entity('node')

        valence = np.zeros(NN, dtype=self.itype)
        np.add.at(valence, cell,1)

        valenceNew = np.zeros(NN, dtype=self.itype)
        np.add.at(valenceNew, cell[isMarkedCell][:, 0], 1)

        isIGoodNode = (valence == valenceNew) & (valence == 4)
        isBGoodNode = (valence == valenceNew) & (valence == 2)

        node2cell = self.ds.node_to_cell()

        I, J = node2cell[isIGoodNode, :].nonzero()
        nodeStar = J.reshape(-1, 4)

        ix = (cell[nodeStar[:, 0], 2] == cell[nodeStar[:, 3], 1])
        iy = (cell[nodeStar[:, 1], 1] == cell[nodeStar[:, 2], 2])
        nodeStar[ ix & (~iy), :] = nodeStar[ix & (~iy), :][:, [0, 2, 1, 3]]
        nodeStar[ (~ix) & iy, :] = nodeStar[(~ix) & iy, :][:, [0, 3, 1, 2]]

        t0 = nodeStar[:, 0]
        t1 = nodeStar[:, 1]
        t2 = nodeStar[:, 2]
        t3 = nodeStar[:, 3]

        p1 = cell[t0, 2]
        p2 = cell[t1, 1]
        p3 = cell[t0, 1]
        p4 = cell[t2, 1]

        cell[t0, 0] = p3
        cell[t0, 1] = p1
        cell[t0, 2] = p2
        cell[t1, 0] = -1

        cell[t2, 0] = p4
        cell[t2, 1] = p2
        cell[t2, 2] = p1
        cell[t3, 0] = -1

        I, J = node2cell[isBGoodNode, :].nonzero()
        nodeStar = J.reshape(-1, 2)
        idx = (cell[nodeStar[:, 0], 2] == cell[nodeStar[:, 1], 1])
        nodeStar[idx, :] = nodeStar[idx, :][:, [1,0]]

        t4 = nodeStar[:, 0]
        t5 = nodeStar[:, 1]
        p0 = cell[t4, 0]
        p1 = cell[t4, 2]
        p2 = cell[t5, 1]
        p3 = cell[t4, 1]
        cell[t4, 0] = p3
        cell[t4, 1] = p1
        cell[t4, 2] = p2
        cell[t5, 0] = -1

        isKeepCell = cell[:, 0] > -1
        if ('data' in options) and (options['data'] is not None):
            # value.shape == (NC, (p+1)*(p+2)//2)
            lidx = np.r_[t0, t2, t4]
            ridx = np.r_[t1, t3, t5]
            for key, value in options['data'].items():
                ldof = value.shape[1]
                p = int((np.sqrt(8*ldof+1) - 3)/2)
                bc = self.multi_index_matrix(p=p)/p
                bcl = np.zeros_like(bc)
                bcl[:, 0] = 2*bc[:, 2]
                bcl[:, 1] = bc[:, 0]
                bcl[:, 2] = bc[:, 1] - bc[:, 2]

                bcr = np.zeros_like(bc)
                bcr[:, 0] = 2*bc[:, 1]
                bcr[:, 1] = bc[:, 2] - bc[:, 1]
                bcr[:, 2] = bc[:, 0]

                phi = self.shape_function(bcl, p=p) # (NQ, ldof)
                value[lidx, :] = np.einsum('ci, qi->cq', value[lidx, :], phi)

                phi = self.shape_function(bcr, p=p) # (NQ, ldof)
                value[lidx, :]+= np.einsum('ci, qi->cq', value[ridx, :], phi)
                value[lidx] /= 2
                options['data'][key] = value[isKeepCell]

        cell = cell[isKeepCell]
        isGoodNode = (isIGoodNode | isBGoodNode)

        idxMap = np.zeros(NN, dtype=self.itype)
        self.node = node[~isGoodNode]

        NN = self.node.shape[0]
        idxMap[~isGoodNode] = range(NN)
        cell = idxMap[cell]

        self.ds = TriangleMeshDataStructure(NN, cell)


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
            tol=1e-6, # 目标误差
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
                'tol': tol,
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
        elif options['method'] == 'target':
            NT = self.number_of_cells()
            e = options['tol']/np.sqrt(NT)
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*e)
                ))
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
        cell = np.zeros((3*NC, 3), dtype=self.itype)

        if ('numrefine' in options) and (options['numrefine'] is not None):
            options['numrefine'] = np.r_[options['numrefine'], np.zeros(2*NC)]

        node[:NN] = self.entity('node')
        cell[:NC] = self.entity('cell')

        # 用于存储网格节点的代数，初始所有节点都为第 0 代
        generation = np.zeros(NN + 2*NC, dtype=np.uint8)

        # 用于记录被二分的边及其中点编号
        cutEdge = np.zeros((4*NN, 3), dtype=self.itype)

        # 当前的二分边的数目
        nCut = 0
        # 非协调边的标记数组
        nonConforming = np.ones(4*NN, dtype=np.bool_)
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
                val = np.ones(len(I), dtype=np.bool_)
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
                        np.ones(NE, dtype=np.bool_),
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
                val = np.ones(len(I), dtype=np.bool_)
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
            isCheckNode = np.zeros(NN, dtype=np.bool_)
            isCheckNode[cutEdge[checkEdge]] = True
            isCheckCell = np.sum(
                    isCheckNode[cell[:NC]],
                    axis= -1) > 0
            # 找到所有包含检查节点的单元编号
            checkCell, = np.nonzero(isCheckCell)
            I = np.repeat(checkCell, 3)
            J = cell[checkCell].reshape(-1)
            val = np.ones(len(I), dtype=np.bool_)
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


    def linear_stiff_matrix(self, c=None):
        """
        Notes
        -----
        线性元的刚度矩阵
        """
        warnings.warn("The linear_stiff_matrix() is deprecated and will be removed in a future version. "
                      "Please use the interpolate() instead.", DeprecationWarning)

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


    def jacobian_matrix(self, index=np.s_[:]):
        """
        @brief 获得三角形单元对应的 Jacobian 矩阵
        """
        NC = self.number_of_cells()
        GD = self.geo_dimension()

        node = self.entity('node')
        cell = self.entity('cell')

        J = np.zeros((NC, GD, 2), dtype=self.ftype)

        J[..., 0] = node[cell[:, 1]] - node[cell[:, 0]]
        J[..., 1] = node[cell[:, 2]] - node[cell[:, 0]]

        return J


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


    def mark_interface_cell(self, phi):
        """
        @brief 标记穿过界面的单元
        """

        if callable(phi):
            node = self.entity('node')
            phi = phi(node)

        cell = self.entity('cell')

        s0 = np.sign(phi[cell[:, 0]])
        s1 = np.sign(phi[cell[:, 1]])
        s2 = np.sign(phi[cell[:, 2]])

        eta0 = np.abs(s0 + s1 + s2)
        eta1 = np.abs(s0) + np.abs(s1) + np.abs(s2)

        isInterfaceCell = ((eta0 == 1) & (eta1 == 3)) | ((eta0 == 0) & (eta1 == 2))

        return isInterfaceCell

    def mark_interface_cell_with_curvature(self, phi, hmax=None):
        """
        @brief 标记曲率大的单元
        """

        if callable(phi):
            node = self.entity('node')
            phi = phi(node)

        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        node = self.entity('node')
        edge = self.entity('edge')
        cell = self.entity('cell')
        edge2cell = self.ds.edge_to_cell()

        isInterfaceCell = self.mark_interface_cell(phi)
        isInterfaceNode = np.zeros(NN, dtype=np.bool_)
        # 界面单元的顶点称为界面节点
        isInterfaceNode[cell[isInterfaceCell]] = True

        # case: Fig 2(a) in p6
        # 一个网格点周围的单元全为界面单元，则周围这些单元以该网格节点为顶点的角度
        # 之和为 360
        angle = np.zeros(NN, dtype=self.itype)
        np.add.at(angle, cell[isInterfaceCell, :], np.array([90, 45, 45]))
        is360 = (angle == 360)
        print('is360:', is360.sum())

        # case: Fig 2(b) in p6
        isInteriorEdge = ~self.ds.boundary_edge_flag()
        isInterfaceBEdge = (isInterfaceCell[edge2cell[:, 0:2]].sum(axis=1) == 1)
        valence = np.zeros(NN, dtype=self.itype)
        if np.any(isInterfaceBEdge):
            np.add.at(valence, edge[isInterfaceBEdge], 1)
        isLinkNode = (valence > 2) & (phi != 0)
        print("isLinkNode:", isLinkNode.sum())


        # case：离散曲率
        isInteriorNode = phi < 0
        tangle = np.zeros(NN, dtype=self.itype) # 记录界面转角
        isIn = isInterfaceNode & isInteriorNode & (~is360) & (~isLinkNode)
        if np.any(isIn):
            tangle[isIn] = 180 - (360 - angle[isIn]);

        isOut = isInterfaceNode & (~isInteriorNode) & (~is360) & (~isLinkNode)
        if np.any(isOut):
            tangle[isOut] = (360 - angle[isOut]) - 180

        isInterfaceEdge = isInterfaceCell[edge2cell[:, 0]] | isInterfaceCell[edge2cell[:, 1]]

        ta = tangle[edge[isInterfaceEdge][:, 1::-1]]
        np.add.at(tangle, edge[isInterfaceEdge], ta)


        flag = np.abs(tangle) > 170
        print("转角:", flag.sum())

        isBigCurveNode = is360 | isLinkNode | (np.abs(tangle) > 170)

        isBigCurveCell = (isBigCurveNode[cell].sum(axis=1) > 0) & isInterfaceCell

        v = node[cell[:, 2]] - node[cell[:, 1]]
        l = np.sqrt(np.sum(v**2, axis=1))
        if hmax is None:
            lmax = np.max(l)
            isBigSizeCell = (l > lmax*lmax) & isInterfaceCell
        else:
            isBigSizeCell = (l > hmax) & isInterfaceCell


        return isBigCurveCell | isBigSizeCell



    def mark_interface_cell_with_type(self, phi, interface):
        """
        @brief 等腰直角三角形，可以分为两类
            - Type A：两条直角边和坐标轴平行
            - Type B: 最长边和坐标轴平行
        """

        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        node = self.entity('node')
        cell = self.entity('cell')
        cell2cell = self.ds.cell_to_cell()

        v = node[cell[:, 2]] - node[cell[:, 1]]
        cellType = (np.abs(v[:, 0]) > 0.0) & (np.abs(v[:, 1]) > 0.0) # TODO: 0.0-->eps
        isInterfaceCell = self.mark_interface_cell(phi)

        isMark = np.zeros(NC, dtype=np.bool_)
        isMark[isInterfaceCell] = True
        isMark[cell2cell[isInterfaceCell, 0]] = True

        n0 = cell2cell[:, 0]
        isTransitionTypeBCell = isInterfaceCell & isInterfaceCell[cell2cell[:,
            0]] & (cell2cell[n0, 0] != range(NC)) & (~cellType)

        # Case 1:
        n1 = cell2cell[:, 1]
        flag = isTransitionTypeBCell & isInterfaceCell[n1] & cellType[n1]
        if np.any(flag):
            idx, = np.nonzero(flag)
            p0 = node[cell[idx, 0]]
            p1 = node[cell[idx, 1]]
            p2 = node[cell[idx, 2]]
            p3 = node[cell[n1[idx], 0]]

            m03 = (p0 + p3)/2
            m23 = (p2 + p3)/2
            m12 = (p1 + p2)/2
            m0 = (m12 + p2)/2
            m1 = (m12 + p1)/2
            isNotTB = ((phi[cell[idx, 2]]*interface(m0) > 0.0) &
                    (interface(m03)*interface(m23) < 0.0)) | (interface(m0)*interface(m1) < 0.0)

            isTransitionTypeBCell[idx[isNotTB]] = False

        # Case 2:
        n2 = cell2cell[:, 2]
        flag = isTransitionTypeBCell & isInterfaceCell[n2] & cellType[n2]
        if np.any(flag):
            idx, = np.nonzero(flag)
            p0 = node[cell[idx, 0]]
            p1 = node[cell[idx, 1]]
            p2 = node[cell[idx, 2]]
            p3 = node[cell[n2[idx], 0]]
            m03 = (p0 + p3)/2
            m13 = (p1 + p3)/2
            m12 = (p1 + p2)/2
            m0 = (m12 + p2)/2
            m1 = (m12 + p1)/2
            isNotTB = ((phi[cell[idx, 1]]*interface(m1) > 0.0) &
                    (interface(m03)*interface(m13) < 0.0)) | (interface(m0)*interface(m1) < 0.0)

            isTransitionTypeBCell[idx[isNotTB]] = False

        # Case 3:
        n1 = cell2cell[:, 1]
        flag = isTransitionTypeBCell & isInterfaceCell[n1] & ~cellType[n1]
        if np.any(flag):
            idx, = np.nonzero(flag)
            p1 = node[cell[flag, 1]]
            p2 = node[cell[flag, 2]]
            m12 = (p1 + p2)/2
            m0 = (m12 + p2)/2
            m1 = (m12 + p2)/2
            isNotTB = interface(m0)*interface(m1) < 0.0
            isTransitionTypeBCell[idx[isNotTB]] = False

        # Case 4:
        n2 = cell2cell[:, 2]
        flag = isTransitionTypeBCell & isInterfaceCell[n2] & (~cellType[n2])
        if np.any(flag):
            idx, = np.nonzero(flag)
            p1 = node[cell[idx, 1]]
            p2 = node[cell[idx, 2]]
            m12 = (p1 + p2)/2
            m0 = (m12 + p2)/2
            m1 = (m12 + p1)/2
            isNotTB = interface(m0)*interface(m1) < 0.0
            isTransitionTypeBCell[idx[isNotTB]] = False


        isTypeBCell = isMark & (~cellType) & (~isTransitionTypeBCell)
        return isTypeBCell, cellType


    def bisect_interface_cell_with_curvature(self, interface, hmax):
        """
        """
        NN = self.number_of_nodes()
        node= self.entity('node')

        phi = interface(node)

        if np.all(phi < 0):
            raise ValueError('初始网格在界面围成区域的内部，需要更换一个可以覆盖界面的网格')

        # Step 1: 一致二分法加密网格
        while np.all(phi>0):
            self.uniform_bisect()
            node = self.entity('node')
            phi = np.append(phi, interface(node[NN:]))
            NN = self.number_of_nodes()

        # Step 2: 估计离散曲率

        isBigCurveCell = self.mark_interface_cell_with_curvature(phi, hmax=hmax)

        k = 0
        while np.any(isBigCurveCell) & (k < 100):
            k += 1
            self.bisect(isBigCurveCell)
            node = self.entity('node')
            phi = np.append(phi, interface(node[NN:]))
            NN = self.number_of_nodes()
            isBigCurveCell = self.mark_interface_cell_with_curvature(phi, hmax=hmax)


        isTypeBCell, cellType = self.mark_interface_cell_with_type(phi, interface)

        k = 0
        while np.any(isTypeBCell) & (k < 100):
            k += 1
            self.bisect(isTypeBCell)
            node = self.entity('node')
            phi = np.append(phi, interface(node[NN:]))
            NN = self.number_of_nodes()
            isTypeBCell, cellType = self.mark_interface_cell_with_type(phi, interface)

    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_square(cls, nx=10, ny=10, threshold=None):
        """
        Generate a triangle mesh for a unit square.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TriangleMesh instance
        """
        return cls.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny, threshold=threshold)

    ## @ingroup MeshGenerators
    @classmethod
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10, threshold=None):
        """
        Generate a triangle mesh for a box domain .

        @param box
        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TriangleMesh instance
        """
        NN = (nx+1)*(ny+1)
        NC = nx*ny
        node = np.zeros((NN,2))
        X, Y = np.mgrid[
                box[0]:box[1]:(nx+1)*1j,
                box[2]:box[3]:(ny+1)*1j]
        node[:, 0] = X.flat
        node[:, 1] = Y.flat

        idx = np.arange(NN).reshape(nx+1, ny+1)
        cell = np.zeros((2*NC, 3), dtype=np.int_)
        cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
        cell[:NC, 1] = idx[1:,1:].flatten(order='F')
        cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 2] = idx[1:, 1:].flatten(order='F')

        if threshold is not None:
            bc = np.sum(node[cell, :], axis=1)/cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = np.zeros(NN, dtype=np.bool_)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = np.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_one_triangle(cls, meshtype='iso'):
        if meshtype == 'equ':
            node = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2]], dtype=np.float64)
        elif meshtype == 'iso':
            node = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]], dtype=np.float64)
        cell = np.array([[0, 1, 2]], dtype=np.int_)
        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_square_domain_with_fracture(cls):
        node = np.array([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]], dtype=np.float64)

        cell = np.array([
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5]], dtype=np.int_)

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_domain_distmesh(cls, domain, hmin, maxit=100):
        from .distmesh import DistMesh2d
        mesher = DistMesher(domain, hmin)
        mesh = mesher.meshing(maxit=maxit)
        return mesh

    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_circle_gmsh(cls, h):
        """
        Generate a triangular mesh for a unit circle by gmsh.

        @param h Parameter controlling mesh density
        @return TriangleMesh instance
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add("UnitCircle")

        # 创建单位圆
        gmsh.model.occ.addDisk(0.0,0.0,0.0,1,1,1)

        # 同步几何模型
        gmsh.model.occ.synchronize()

        # 设置网格尺寸
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)

        # 生成网格
        gmsh.model.mesh.generate(2)

        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = node_coords.reshape((-1,3))[:,:2]
        
        # 节点编号映射
        nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

        # 获取单元信息
        cell_type = 2 # 三角形单元的类型编号为 2
        cell_tags,cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)

        # 节点编号映射到单元
        evid = np.array([nodetags_map[j] for j in cell_connectivity])
        cell = evid.reshape((cell_tags.shape[-1],-1))

        gmsh.finalize()

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of cells: {cell.shape[0]}")

        return cls(node, cell)


    ## @ingroup MeshGenerators
    @classmethod
    def from_polygon_gmsh(cls, vertices, h):
        """
        Generate a triangle mesh for a polygonal region by gmsh.

        @param vertices List of tuples representing vertices of the polygon
        @param h Parameter controlling mesh density
        @return TriangleMesh instance
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add("Polygon")

        # 创建多边形
        lc = h  # 设置网格大小
        polygon_points = []
        for i, vertex in enumerate(vertices):
            point = gmsh.model.geo.addPoint(vertex[0], vertex[1], 0, lc)
            polygon_points.append(point)

        # 添加线段和循环
        lines = []
        for i in range(len(polygon_points)):
            line = gmsh.model.geo.addLine(polygon_points[i], polygon_points[(i+1) % len(polygon_points)])
            lines.append(line)
        curve_loop = gmsh.model.geo.addCurveLoop(lines)

        # 创建平面表面
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        # 同步几何模型
        gmsh.model.geo.synchronize()

        # 添加物理组
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Polygon")

        # 生成网格
        gmsh.model.mesh.generate(2)

        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)[:, 0:2].copy()

        # 获取三角形单元信息
        cell_type = 2  # 三角形单元的类型编号为 2
        cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)
        cell = np.array(cell_connectivity, dtype=np.int_).reshape(-1, 3) - 1

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of cells: {cell.shape[0]}")

        gmsh.finalize()

        NN = len(node)
        isValidNode = np.zeros(NN, dtype=np.bool_)
        isValidNode[cell] = True
        node = node[isValidNode]
        idxMap = np.zeros(NN, dtype=cell.dtype)
        idxMap[isValidNode] = range(isValidNode.sum())
        cell = idxMap[cell]

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_torus_surface(cls, R, r, nu, nv):
        """
        @brief Generate a structured triangular mesh on a torus surface.

        @param R  The major radius of the torus (distance from the center of the torus to the center of the tube).
        @param r  The minor radius of the torus (radius of the tube).
        @param nu The number of discrete segments in the u-direction.
        @param nv The number of discrete segments in the v-direction.

        @return  the triangular mesh.

        @details This function generates a structured triangular mesh on a torus surface with major radius R,
                 minor radius r, and nu and nv discrete segments in the u and v directions, respectively.
                 The output consists of a tuple containing the nodes and cells of the mesh. The nodes are
                 represented as an Nx3 array of 3D coordinates, and the cells are represented as an Mx3
                 array of node indices for each triangle.
        @todo 检查生成曲面单元的法向是否指向外部
        """
        NN = nu*nv
        NC = nu*nv
        node = np.zeros((NN, 3), dtype=np.float64)

        U, V = np.mgrid[0:2*np.pi:nu*1j, 0:2*np.pi:nv*1j]
        X = (R + r * np.cos(V)) * np.cos(U)
        Y = (R + r * np.cos(V)) * np.sin(U)
        Z = r * np.sin(V)
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        node[:, 2] = Z.flatten()

        idx = np.zeros((nu+1, nv+1), np.int_)
        idx[0:-1, 0:-1] = np.arange(NN).reshape(nu, nv)
        idx[-1, :] = idx[0, :]
        idx[:, -1] = idx[:, 0]
        cell = np.zeros((2*NC, 3), dtype=np.int_)
        cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
        cell[:NC, 1] = idx[1:,1:].flatten(order='F')
        cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 2] = idx[1:, 1:].flatten(order='F')

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_sphere_surface(cls,refine=2):
        """
        @brief  Generate a triangular mesh on a unit sphere surface.
        @return the triangular mesh.
        """
        t = (np.sqrt(5) - 1)/2
        node = np.array([
            [ 0, 1, t],[ 0, 1,-t],[ 1, t, 0],[ 1,-t, 0],
            [ 0,-1,-t],[ 0,-1, t],[ t, 0, 1],[-t, 0, 1],
            [ t, 0,-1],[-t, 0,-1],[-1, t, 0],[-1,-t, 0]], dtype=np.float64)
        cell = np.array([
            [6, 2, 0],[3, 2, 6],[5, 3, 6],[5, 6, 7],
            [6, 0, 7],[3, 8, 2],[2, 8, 1],[2, 1, 0],
            [0, 1,10],[1, 9,10],[8, 9, 1],[4, 8, 3],
            [4, 3, 5],[4, 5,11],[7,10,11],[0,10, 7],
            [4,11, 9],[8, 4, 9],[5, 7,11],[10,9,11]], dtype=np.int_)
        mesh = cls(node,cell)
        mesh.uniform_refine(refine)
        node = mesh.node
        cell = mesh.entity('cell')
        d = np.sqrt(node[:,0]**2 + node[:,1]**2 + node[:,2]**2)-1
        l = np.sqrt(np.sum(node**2,axis=1))
        n = node/l[...,np.newaxis]
        node = node - d[...,np.newaxis]*n
        return cls(node,cell)


class TriangleMeshWithInfinityNode:
    def __init__(self, mesh, bc=True):
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

        if bc:
            self.center = np.append(mesh.entity_barycenter(),
                    0.5*(node[edge[bdEdgeIdx, 0], :] + node[edge[bdEdgeIdx, 1], :]), axis=0)
        else:
            self.center = np.append(mesh.circumcenter(),
                    0.5*(node[edge[bdEdgeIdx, 0], :] + node[edge[bdEdgeIdx, 1], :]), axis=0)

        self.meshtype = 'tri'

    def number_of_nodes(self):
        return self.node.shape[0]

    def number_of_edges(self):
        return self.ds.NE

    def number_of_faces(self):
        return self.ds.NE

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
        isBdEdge = np.zeros(NE, dtype=np.bool_)
        isBdEdge[cell2edge[isInfCell, 0]] = True
        return isBdEdge

    def is_boundary_node(self):
        N = self.number_of_nodes()
        edge = self.ds.edge
        isBdEdge = self.is_boundary_edge()
        isBdNode = np.zeros(N, dtype=np.bool_)
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

        node2cell = self.ds.node_to_cell(return_localidx=True)
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
