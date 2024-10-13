from typing import Sequence
import numpy as np
import warnings
from scipy.sparse import coo_matrix, csr_matrix, bmat, eye
from scipy.spatial import KDTree

from .triangle_quality import *

from .mesh_base import Mesh, Plotable
from .mesh_data_structure import Mesh2dDataStructure


class TriangleMeshDataStructure(Mesh2dDataStructure):
    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    localFace = np.array([(1, 2), (2, 0), (0, 1)])
    ccw = np.array([0, 1, 2])

    localCell = np.array([
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1)])


## @defgroup MeshGenerators TriangleMesh Common Region Mesh Generators
## @defgroup MeshQuality
class TriangleMesh(Mesh, Plotable):
    def __init__(self, node, cell):
        """
        @brief TriangleMesh 对象的初始化函数

        """

        assert cell.shape[-1] == 3

        self.node = node
        NN = node.shape[0]
        self.ds = TriangleMeshDataStructure(NN, cell)

        self.meshtype = 'tri'
        self.type = "TRI"

        self.itype = cell.dtype
        self.ftype = node.dtype
        self.p = 1  # 平面三角形

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.meshdata = {}

        self.edge_bc_to_point = self.bc_to_point
        self.cell_bc_to_point = self.bc_to_point
        self.face_to_ipoint = self.edge_to_ipoint

        self.shape_function = self._shape_function
        self.cell_shape_function = self._shape_function
        self.face_shape_function = self._shape_function
        self.edge_shape_function = self._shape_function

    def ref_cell_measure(self):
        return 0.5

    def ref_face_measure(self):
        return 1.0

    def integrator(self, q, etype='cell'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 2}:
            from ..quadrature import TriangleQuadrature
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            from ..quadrature import GaussLegendreQuadrature
            return GaussLegendreQuadrature(q)

    def entity_measure(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return 0
        else:
            raise ValueError(f"Invalid entity type '{etype}'.")

    def grad_shape_function(self, bc, p=1, index=np.s_[:], variables='x'):
        """
        @note 注意这里调用的实际上不是形状函数的梯度，而是网格空间基函数的梯度
        """
        R = self._grad_shape_function(bc, p)
        if variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = np.einsum('...ij, kjm-> ...kim', R, Dlambda, optimize=True)
            return gphi  # (NQ, NC, ldof, GD)
        elif variables == 'u':
            return R  # (NQ, ldof, TD+1)

    cell_grad_shape_function = grad_shape_function

    def grad_shape_function_on_edge(self, bc, cindex, lidx, p=1, direction=True):
        """
        @brief 计算单元上所有形函数在边上的积分点处的导函数值

        @param bc 边上的一组积分点
        @param cindex 边所在的单元编号
        @param lidx 边在该单元的局部编号
        @param direction  True 表示边的方向和单元的逆时针方向一致，False 表示不一致
        """

        NC = len(cindex)
        nmap = np.array([1, 2, 0])
        pmap = np.array([2, 0, 1])
        shape = (NC,) + bc.shape[0:-1] + (3,)
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

    def number_of_local_ipoints(self, p, iptype='cell'):
        """
        @brief
        """
        if iptype in {'cell', 2}:
            return (p + 1) * (p + 2) // 2
        elif iptype in {'face', 'edge', 1}:  # 包括两个顶点
            return p + 1
        elif iptype in {'node', 0}:
            return 1

    def number_of_global_ipoints(self, p):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        return NN + (p - 1) * NE + (p - 2) * (p - 1) // 2 * NC

    def prolongation_matrix(self, p0: int, p1: int):
        """
        @brief 生成从 p0 元到 p1 元的延拓矩阵，假定 0 < p0 < p1
        """

        assert 0 < p0 < p1

        TD = self.top_dimension()
        gdof0 = self.number_of_global_ipoints(p0)
        gdof1 = self.number_of_global_ipoints(p1)

        # 1. 网格节点上的插值点
        NN = self.number_of_nodes()
        I = range(NN)
        J = range(NN)
        V = np.ones(NN, dtype=self.ftype)
        P = coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 2. 网格边内部的插值点
        NE = self.number_of_edges()
        # p1 元在边上插值点对应的重心坐标
        bcs = self.multi_index_matrix(p1, TD - 1) / p1
        # p0 元基函数在 p1 元对应的边内部插值点处的函数值
        phi = self.edge_shape_function(bcs[1:-1], p=p0)  # (ldof1 - 2, ldof0)

        e2p1 = self.edge_to_ipoint(p1)[:, 1:-1]
        e2p0 = self.edge_to_ipoint(p0)
        shape = (NE,) + phi.shape

        I = np.broadcast_to(e2p1[:, :, None], shape=shape).flat
        J = np.broadcast_to(e2p0[:, None, :], shape=shape).flat
        V = np.broadcast_to(phi[None, :, :], shape=shape).flat

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 3. 单元内部的插值点
        if p1 > 2:
            NC = self.number_of_cells()
            # p1 元在单元上对应插值点的重心坐标
            bcs = self.multi_index_matrix(p1, TD) / p1
            flag = np.sum(bcs > 0, axis=1) == 3
            # p0 元基函数在 p1 元对应的单元内部插值点处的函数值
            phi = self.shape_function(bcs[flag, :], p=p0)
            c2p1 = self.cell_to_ipoint(p1)[:, flag]
            c2p0 = self.cell_to_ipoint(p0)

            shape = (NC,) + phi.shape

            I = np.broadcast_to(c2p1[:, :, None], shape=shape).flat
            J = np.broadcast_to(c2p0[:, None, :], shape=shape).flat
            V = np.broadcast_to(phi[None, :, :], shape=shape).flat

            P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        return P.tocsr()

    def interpolation_points(self, p: int, index=np.s_[:]):
        """
        @brief 获取三角形网格上所有 p 次插值点
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

            w = np.zeros((p - 1, 2), dtype=np.float64)
            w[:, 0] = np.arange(p - 1, 0, -1) / p
            w[:, 1] = w[-1::-1, 0]
            ipoints[NN:NN + (p - 1) * NE, :] = np.einsum('ij, ...jm->...im', w,
                                                         node[edge, :]).reshape(-1, GD)
        if p > 2:
            TD = self.top_dimension()
            multiIndex = self.multi_index_matrix(p, TD)
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:, 0] | isEdgeIPoints[:, 1] |
                                isEdgeIPoints[:, 2])
            w = multiIndex[isInCellIPoints, :] / p
            ipoints[NN + (p - 1) * NE:, :] = np.einsum('ij, kj...->ki...', w,
                                                       node[cell, :]).reshape(-1, GD)
        return ipoints  # (gdof, GD)

    def cell_to_ipoint(self, p, index=np.s_[:]):
        """
        """
        cell = self.entity('cell')
        if p == 0:
            return np.arange(len(cell)).reshape((-1, 1))[index]
        if p == 1:
            return cell[index]

        mi = self.multi_index_matrix(p, 2)
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

        cdof = (p - 1) * (p - 2) // 2
        flag = np.sum(mi > 0, axis=1) == 3
        c2p[:, flag] = NN + NE * (p - 1) + np.arange(NC * cdof).reshape(NC, cdof)
        return c2p[index]

    def edge_frame(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上的局部标架
        """
        assert self.geo_dimension() == 2
        t = self.edge_unit_tangent(index=index)
        w = np.array([(0, -1), (1, 0)])
        n = t @ w
        return n, t

    def edge_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_tangent(index=index)
        w = np.array([(0, -1), (1, 0)])
        return v @ w

    def edge_unit_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_unit_tangent(index=index)
        w = np.array([(0, -1), (1, 0)])
        return v @ w

    face_normal = edge_normal
    face_unit_normal = edge_unit_normal

    def sphere_surface_unit_normal(self, index=np.s_[:]):
        """
        @brief 计算单位球面三角形网格中每个面上的单位法线
        """
        assert self.geo_dimension() == 3
        node = self.entity('node')
        cell = self.entity('cell')

        v0 = node[cell[index, 2]] - node[cell[index, 1]]
        v1 = node[cell[index, 0]] - node[cell[index, 2]]
        v2 = node[cell[index, 1]] - node[cell[index, 0]]

        nv = np.cross(v1, v2)
        length = np.linalg.norm(nv, axis=-1, keepdims=True)

        n = nv / length
        return n

    def grad_lambda(self, index=np.s_[:]):
        """
        @brief Calculate the gradient of the barycenter coordinates in each cell.

        @param index: int, NDArray or slice.

        @return: An array with shape (NC, 3, GD).
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        NC = cell.shape[0]
        v0 = node[cell[..., 2]] - node[cell[..., 1]]
        v1 = node[cell[..., 0]] - node[cell[..., 2]]
        v2 = node[cell[..., 1]] - node[cell[..., 0]]
        GD = self.geo_dimension()
        nv = np.cross(v1, v2)
        Dlambda = np.zeros((NC, 3, GD), dtype=self.ftype)

        if GD == 2:
            length = nv
            W = np.array([[0, 1], [-1, 0]], dtype=self.ftype)
            Dlambda[:, 0] = v0 @ W / length[:, None]
            Dlambda[:, 1] = v1 @ W / length[:, None]
            Dlambda[:, 2] = v2 @ W / length[:, None]
        elif GD == 3:
            length = np.linalg.norm(nv, axis=-1, keepdims=True)
            n = nv / length
            Dlambda[:, 0] = np.cross(n, v0) / length
            Dlambda[:, 1] = np.cross(n, v1) / length
            Dlambda[:, 2] = np.cross(n, v2) / length
        return Dlambda

    def rot_lambda(self, index=np.s_[:]):
        """
        @brief

        @param index: int, NDArray or slice.

        @return: An array with shape (NC, 3, GD).
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        NC = cell.shape[0]
        v0 = node[cell[..., 2]] - node[cell[..., 1]]
        v1 = node[cell[..., 0]] - node[cell[..., 2]]
        v2 = node[cell[..., 1]] - node[cell[..., 0]]
        GD = self.geo_dimension()
        nv = np.cross(v2, -v1)
        Rlambda = np.zeros((NC, 3, GD), dtype=self.ftype)
        if GD == 2:
            length = nv
        elif GD == 3:
            length = np.linalg.norm(nv, axis=-1)

        Rlambda[:, 0, :] = v0 / length.reshape((-1, 1))
        Rlambda[:, 1, :] = v1 / length.reshape((-1, 1))
        Rlambda[:, 2, :] = v2 / length.reshape((-1, 1))

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
            edge2newNode = np.arange(NN, NN + NE)
            newNode = (node[edge[:, 0], :] + node[edge[:, 1], :]) / 2.0

            if returnim:
                A = coo_matrix((np.ones(NN), (range(NN), range(NN))), shape=(NN + NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5 * np.ones(NE), (range(NN, NN + NE), edge[:, 0])), shape=(NN + NE, NN),
                                dtype=self.ftype)
                A += coo_matrix((0.5 * np.ones(NE), (range(NN, NN + NE), edge[:, 1])), shape=(NN + NE, NN),
                                dtype=self.ftype)
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
            self.ds.reinit(NN + NE, cell)

        if returnim:
            return nodeIMatrix, cellIMatrix

    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_TRIANGLE = 5
            return VTK_TRIANGLE
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, fname=None, etype='cell', index=np.s_[:]):
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
        p0 = point[segment[:, 0]]  # 线段起点
        p1 = point[segment[:, 1]]  # 线段终点
        v = p0 - p1

        start = location[segment[:, 0]]  # 出发单元
        end = location[segment[:, 1]]  # 终止单元

        isNotOK = np.ones(len(segment), dtype=np.bool_)
        jdx = 3
        while isNotOK.any():
            idx = start[isNotOK]  # 当前单元

            pp0 = p0[isNotOK]
            pp1 = p1[isNotOK]
            vv = v[isNotOK]

            a = np.zeros((len(idx), 3), dtype=self.ftype)
            v0 = node[cell[idx, 0]] - pp1  # 所在单元的三个顶点
            v1 = node[cell[idx, 1]] - pp1
            v2 = node[cell[idx, 2]] - pp1
            a[:, 0] = np.cross(v0, vv)
            a[:, 1] = np.cross(v1, vv)
            a[:, 2] = np.cross(v2, vv)

            b = np.zeros((len(idx), 3), dtype=self.ftype)
            b[:, 0] = np.cross(v1, v2)
            b[:, 1] = np.cross(v2, v0)
            b[:, 2] = np.cross(v0, v1)

            isOK = np.sum(b >= 0, axis=-1) == 3
            idx0, = np.nonzero(isNotOK)

            for i in range(3):
                flag = np.abs(a[:, i]) < 1e-12
                isCrossedNode[cell[idx[flag], i]] = True

            lidx = np.zeros(len(idx), dtype=np.int_)
            for i in range(3):
                j = nx[i]
                k = pr[i]
                flag0 = (a[:, j] <= 0) & (a[:, k] >= 0) & (jdx != i)
                lidx[flag0] = i

            # 移动到下一个单元
            tmp = start[idx0[~isOK]]
            start[idx0[~isOK]] = cell2cell[idx[~isOK], lidx[~isOK]]
            isNotOK[idx0[isOK]] = False
            _, jdx = np.where((cell2cell[start[isNotOK]].T == tmp).T)

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
        给定一组点 p , 找到这些点所在的单元

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
        start = start[loc]  # 设置一个初始单元位置

        isNotOK = np.ones(NP, dtype=np.bool_)
        while np.any(isNotOK):
            idx = start[isNotOK]
            pp = points[isNotOK]

            v0 = node[cell[idx, 0]] - pp  # 所在单元的三个顶点
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
            area = nv / 2.0
            x2 = np.sum(node ** 2, axis=1, keepdims=True)
            w0 = x2[cell[index, 2]] + x2[cell[index, 1]]
            w1 = x2[cell[index, 0]] + x2[cell[index, 2]]
            w2 = x2[cell[index, 1]] + x2[cell[index, 0]]
            W = np.array([[0, -1], [1, 0]], dtype=self.ftype)
            fe0 = w0 * v0 @ W
            fe1 = w1 * v1 @ W
            fe2 = w2 * v2 @ W
            c = 0.25 * (fe0 + fe1 + fe2) / area.reshape(-1, 1)
            R = np.sqrt(np.sum((c - node[cell[index, 0], :]) ** 2, axis=1))
        elif GD == 3:
            length = np.sqrt(np.sum(nv ** 2, axis=1))
            n = nv / length.reshape((-1, 1))
            l02 = np.sum(v1 ** 2, axis=1, keepdims=True)
            l01 = np.sum(v2 ** 2, axis=1, keepdims=True)
            d = 0.5 * (l02 * np.cross(n, v2) + l01 * np.cross(-v1, n)) / length.reshape(-1, 1)
            c = node[cell[index, 0]] + d
            R = np.sqrt(np.sum(d ** 2, axis=1))

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
        for i, (j, k) in zip(range(3), localEdge):
            v0 = node[cell[:, j]] - node[cell[:, i]]
            v1 = node[cell[:, k]] - node[cell[:, i]]
            angle[:, i] = np.arccos(
                np.sum(v0 * v1, axis=1) / np.sqrt(np.sum(v0 ** 2, axis=1) * np.sum(v1 ** 2, axis=1)))
        return angle

    def show_angle(self, axes, angle=None):
        """
        @brief 显示网格角度的分布直方图
        """
        if angle is None:
            angle = self.angle()
        hist, bins = np.histogram(angle.flatten('F') * 180 / np.pi, bins=50, range=(0, 180))
        center = (bins[:-1] + bins[1:]) / 2
        axes.bar(center, hist, align='center', width=180 / 50.0)
        axes.set_xlim(0, 180)
        mina = np.min(angle.flatten('F') * 180 / np.pi)
        maxa = np.max(angle.flatten('F') * 180 / np.pi)
        meana = np.mean(angle.flatten('F') * 180 / np.pi)
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

    def cell_quality(self, measure='radius_ratio'):
        if measure == 'radius_ratio':
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
        axes.annotate('Min quality: {:.6}'.format(minq), xy=(0, 0), xytext=(0.1, 0.5),
                      textcoords='axes fraction',
                      horizontalalignment='left', verticalalignment='top')
        axes.annotate('Max quality: {:.6}'.format(maxq), xy=(0, 0), xytext=(0.1, 0.45),
                      textcoords='axes fraction',
                      horizontalalignment='left', verticalalignment='top')
        axes.annotate('Average quality: {:.6}'.format(meanq), xy=(0, 0), xytext=(0.1, 0.40),
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
            isNonDelaunayEdge = (asum > np.pi) & (edge2cell[:, 0] != edge2cell[:, 1])

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

        cm = self.entity_measure("cell")  # 单元面积
        cb = self.entity_barycenter('cell')  # 单元重心
        cc = self.circumcenter()  # 单元外心
        cc[isBdCell] = cb[isBdCell]
        cc = cm[..., None] * cc

        newNode = np.zeros((NN, 2), dtype=np.float64)
        patch_area = np.zeros(NN, dtype=np.float64)

        np.add.at(newNode, cell, np.broadcast_to(cc[:, None], (NC, 3, 2)))
        np.add.at(patch_area, cell, np.broadcast_to(cm[:, None], (NC, 3)))

        newNode[isBdNode] = node[isBdNode]
        newNode[isFreeNode] = newNode[isFreeNode] / patch_area[..., None][isFreeNode]
        self.node = newNode
        while np.sum(cm <= 0) > 0:
            self.node[isFreeNode] = (1 - alpha / 2) * node[isFreeNode] + alpha * newNode[isFreeNode]
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

        cm = self.entity_measure("cell")  # 单元面积
        cb = self.entity_barycenter('cell')  # 单元重心
        cb = cm[..., None] * cb

        newNode = np.zeros((NN, 2), dtype=np.float64)
        patch_area = np.zeros(NN, dtype=np.float64)

        np.add.at(newNode, cell, np.broadcast_to(cb[:, None], (NC, 3, 2)))
        np.add.at(patch_area, cell, np.broadcast_to(cm[:, None], (NC, 3)))

        newNode[isBdNode] = node[isBdNode]
        newNode[isFreeNode] = newNode[isFreeNode] / patch_area[..., None][isFreeNode]
        self.node = newNode
        while np.sum(cm <= 0) > 0:
            self.node[isFreeNode] = (1 - alpha / 2) * node[isFreeNode] + alpha * newNode[isFreeNode]
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

    def bisect(self, isMarkedCell=None, options={'disp': True}):

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
        cell2ipoint = self.cell_to_ipoint(self.p)
        isCutEdge = np.zeros((NE,), dtype=np.bool_)

        if options['disp']:
            print('The initial number of marked elements:', isMarkedCell.sum())

        markedCell, = np.nonzero(isMarkedCell)
        while len(markedCell) > 0:
            isCutEdge[cell2edge[markedCell, 0]] = True
            refineNeighbor = cell2cell[markedCell, 0]
            markedCell = refineNeighbor[~isCutEdge[cell2edge[refineNeighbor, 0]]]

        if options['disp']:
            print('The number of markedg edges: ', isCutEdge.sum())

        edge2newNode = np.zeros((NE,), dtype=self.itype)
        edge2newNode[isCutEdge] = np.arange(NN, NN + isCutEdge.sum())

        node = self.node
        newNode = 0.5 * (node[edge[isCutEdge, 0], :] + node[edge[isCutEdge, 1], :])
        self.node = np.concatenate((node, newNode), axis=0)
        cell2edge0 = cell2edge[:, 0]

        if 'data' in options:
            pass

        if 'IM' in options:
            nn = len(newNode)
            IM = coo_matrix((np.ones(NN), (np.arange(NN), np.arange(NN))),
                            shape=(NN + nn, NN), dtype=self.ftype)
            val = np.full(nn, 0.5)
            IM += coo_matrix(
                (
                    val,
                    (
                        NN + np.arange(nn),
                        edge[isCutEdge, 0]
                    )
                ), shape=(NN + nn, NN), dtype=self.ftype)
            IM += coo_matrix(
                (
                    val,
                    (
                        NN + np.arange(nn),
                        edge[isCutEdge, 1]
                    )
                ), shape=(NN + nn, NN), dtype=self.ftype)
            options['IM'] = IM.tocsr()

        if 'HB' in options:
            options['HB'] = np.arange(NC)

        for k in range(2):
            idx, = np.nonzero(edge2newNode[cell2edge0] > 0)
            nc = len(idx)
            if nc == 0:
                break

            if 'HB' in options:
                HB = options['HB']
                options['HB'] = np.concatenate((HB, HB[idx]), axis=0)

            L = idx
            R = np.arange(NC, NC + nc)
            if ('data' in options) and (options['data'] is not None):
                for key, value in options['data'].items():
                    if value.shape == (NC,):  # 分片常数
                        value = np.r_[value[:], value[idx]]
                        options['data'][key] = value
                    elif value.shape == (NN + k * nn,):
                        if k == 0:
                            value = np.r_['0', value, np.zeros((nn,), dtype=self.ftype)]
                            value[NN:] = 0.5 * (value[edge[isCutEdge, 0]] + value[edge[isCutEdge, 1]])
                            options['data'][key] = value
                    else:
                        ldof = value.shape[-1]
                        p = int((np.sqrt(1 + 8 * ldof) - 3) // 2)
                        bc = self.multi_index_matrix(p, etype=2) / p

                        bcl = np.zeros_like(bc)
                        bcl[:, 0] = bc[:, 1]
                        bcl[:, 1] = 1 / 2 * bc[:, 0] + bc[:, 2]
                        bcl[:, 2] = 1 / 2 * bc[:, 0]

                        bcr = np.zeros_like(bc)
                        bcr[:, 0] = bc[:, 2]
                        bcr[:, 1] = 1 / 2 * bc[:, 0]
                        bcr[:, 2] = 1 / 2 * bc[:, 0] + bc[:, 1]
                        value = np.r_['0', value, np.zeros((nc, ldof), dtype=self.ftype)]

                        phi = self.shape_function(bcr, p=p)
                        value[NC:, :] = np.einsum('cj,kj->ck', value[idx], phi)

                        phi = self.shape_function(bcl, p=p)
                        value[idx, :] = np.einsum('cj,kj->ck', value[idx], phi)

                        options['data'][key] = value

            p0 = cell[idx, 0]
            p1 = cell[idx, 1]
            p2 = cell[idx, 2]
            p3 = edge2newNode[cell2edge0[idx]]
            cell = np.concatenate((cell, np.zeros((nc, 3), dtype=self.itype)), axis=0)
            cell[L, 0] = p3
            cell[L, 1] = p0
            cell[L, 2] = p1
            cell[R, 0] = p3
            cell[R, 1] = p2
            cell[R, 2] = p0
            if k == 0:
                cell2edge0 = np.zeros((NC + nc,), dtype=self.itype)
                cell2edge0[0:NC] = cell2edge[:, 0]
                cell2edge0[L] = cell2edge[idx, 2]
                cell2edge0[R] = cell2edge[idx, 1]
            NC = NC + nc

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
        np.add.at(valence, cell, 1)

        valenceNew = np.zeros(NN, dtype=self.itype)
        np.add.at(valenceNew, cell[isMarkedCell][:, 0], 1)

        isIGoodNode = (valence == valenceNew) & (valence == 4)
        isBGoodNode = (valence == valenceNew) & (valence == 2)

        node2cell = self.ds.node_to_cell()

        I, J = node2cell[isIGoodNode, :].nonzero()
        nodeStar = J.reshape(-1, 4)

        ix = (cell[nodeStar[:, 0], 2] == cell[nodeStar[:, 3], 1])
        iy = (cell[nodeStar[:, 1], 1] == cell[nodeStar[:, 2], 2])
        nodeStar[ix & (~iy), :] = nodeStar[ix & (~iy), :][:, [0, 2, 1, 3]]
        nodeStar[(~ix) & iy, :] = nodeStar[(~ix) & iy, :][:, [0, 3, 1, 2]]

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
        nodeStar[idx, :] = nodeStar[idx, :][:, [1, 0]]

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
                p = int((np.sqrt(8 * ldof + 1) - 3) / 2)
                bc = self.multi_index_matrix(p=p, etype=2) / p
                bcl = np.zeros_like(bc)
                bcl[:, 0] = 2 * bc[:, 2]
                bcl[:, 1] = bc[:, 0]
                bcl[:, 2] = bc[:, 1] - bc[:, 2]

                bcr = np.zeros_like(bc)
                bcr[:, 0] = 2 * bc[:, 1]
                bcr[:, 1] = bc[:, 2] - bc[:, 1]
                bcr[:, 2] = bc[:, 0]

                phi = self.shape_function(bcl, p=p)  # (NQ, ldof)
                value[lidx, :] = np.einsum('ci, qi->cq', value[lidx, :], phi)

                phi = self.shape_function(bcr, p=p)  # (NQ, ldof)
                value[lidx, :] += np.einsum('ci, qi->cq', value[ridx, :], phi)
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
            (node[totalEdge[:, 1]] - node[totalEdge[:, 0]]) ** 2,
            axis=-1)
        length += 0.1 * np.random.rand(NE) * length
        cellEdgeLength = length.reshape(NC, 3)
        lidx = np.argmax(cellEdgeLength, axis=-1)

        flag = (lidx == 1)
        if sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [1, 2, 0]]

        flag = (lidx == 2)
        if sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [2, 0, 1]]

        if rflag == True:
            self.ds.construct()
    
    def delete_degree_4(self):
        node = self.entity('node')
        cell = self.entity('cell')
        edge = self.entity('edge')
        
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        isBdNode = self.ds.boundary_node_flag()
        isFreeNode = ~isBdNode

        node2node = self.ds.node_to_node()
        node2cell = self.ds.node_to_cell()

        degree = np.array(np.sum(node2cell > 0, axis=1)).reshape(-1)
        degree4flag = isFreeNode & (degree == 4)
        nodemap = np.arange(NN)
        degree4node = nodemap[degree4flag] # 得到度为4的节点编号
        
        degree4 = node2node[degree4node]
        nonzero_indices = degree4.nonzero()
        d4node = np.vstack(nonzero_indices).T # 度为4节点的相邻节点编号
        d4node[:,0] = np.repeat(d4node[::4,1],4)
        connect = node2node[d4node[:,0],d4node[:,1]]
        connect = connect.reshape(-1,4) # 找到度为4节点相邻节点的对点

        cdegree = degree[d4node[:,1]]
        cdegree = cdegree.reshape(-1,4) # 度为4相邻节点的度数
        cnode = d4node[:,1].reshape(-1,4) # 度为4相邻节点的编号

        mindegree = np.argmin(cdegree,axis=1) # 找到相邻节点中度最小的节点
        mindegreeflag = np.array(connect[np.arange(len(mindegree)),mindegree]).reshape(-1)

        degree4cell = node2cell[degree4node] # 找到度为4节点所在的单元编号
        nonzero_indices = degree4cell.nonzero()
        d4cell = np.vstack(nonzero_indices).T

        maskcell = np.ones(NC,dtype=np.bool_)
        maskcell[d4cell[:,1]] = False
        newcell0 = cell[maskcell] # 去除度为4节点所在的单元
        
        newcell1 = np.zeros((cdegree.shape[0],3),dtype=np.int_)
        newcell2 = np.zeros((cdegree.shape[0],3),dtype=np.int_)

        # 确定度为4节点相邻节点中度最小节点的对点
        true_idx = np.argwhere(connect)
        false_idx = np.argwhere(connect==False)
        true_idx = cnode[true_idx[:,0],true_idx[:,1]]
        false_idx = cnode[false_idx[:,0],false_idx[:,1]]
        true_idx = true_idx.reshape(-1,2)
        false_idx = false_idx.reshape(-1,2)
        
        # 连接度最小的节点和其对点，生成两个三角形
        newcell1[mindegreeflag,:2] = true_idx[mindegreeflag]
        newcell2[mindegreeflag,:2] = true_idx[mindegreeflag]
        newcell1[~mindegreeflag,:2] = false_idx[~mindegreeflag]
        newcell2[~mindegreeflag,:2] = false_idx[~mindegreeflag]
        newcell1[mindegreeflag,2] = false_idx[mindegreeflag,0]
        newcell2[mindegreeflag,2] = false_idx[mindegreeflag,1]
        newcell1[~mindegreeflag,2] = true_idx[~mindegreeflag,0]
        newcell2[~mindegreeflag,2] = true_idx[~mindegreeflag,1]
        newcell = np.r_[newcell1,newcell2]
        
        # 调整单元节点编号顺序
        newcell = np.r_[newcell1,newcell2]
        v1 = node[newcell[:,1]]-node[newcell[:,0]]
        v2 = node[newcell[:,2]]-node[newcell[:,0]]
        flag = np.cross(v1,v2)
        newcell[flag<0,0],newcell[flag<0,1] = newcell[flag<0,1],newcell[flag<0,0]
        
        newcell = np.r_[newcell0,newcell]# 新的单元

        masknode = np.ones(NN,dtype=np.bool_)
        masknode[degree4node] = False
        newnode = node[masknode] # 删除度为4的节点

        node_map = np.full(NN,-1)
        node_map[masknode] = np.arange(len(newnode))
        newcell = node_map[newcell] # 根据新的节点编号调整单元节点编号

        self.node = newnode
        NN = newnode.shape[0]
        self.ds.reinit(NN,newcell)
    
    def degree_edgeswap(self):
        node = self.entity('node')
        cell = self.entity('cell')
        edge = self.entity('edge')
         
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
       
        isBdNode = self.ds.boundary_node_flag()
        isFreeNode = ~isBdNode

        node2node = self.ds.node_to_node()
        node2cell = self.ds.node_to_cell()
        edge2cell = self.ds.edge_to_cell()

        degree = np.array(np.sum(node2cell > 0, axis=1)).reshape(-1)
        edge_degree = degree[edge]
        
        # 找到两个端点的度都为7的边
        markedge = (edge_degree[:,0]==7) & (edge_degree[:,1]==7) 
        markcell_index = edge2cell[markedge][:,:2]# 边所在的两个单元
        
        markcell = cell[markcell_index]
        markcell_degree = degree[markcell]# 单元节点的度
        # 计算两个单元节点度的和
        markcell_degreesum = np.sum(np.sum(markcell_degree,axis=1),axis=1)
        
        # 两个单元节点度的和为38(四个节点度为7,7,5,5)或39(四个节点度为7,7,5,6)的单元
        swapcellflag = (markcell_degreesum==38) #| (markcell_degreesum==39)
        swapcell = markcell[swapcellflag]# 交换边的单元
        swapedge = edge[markedge]
        swapedge = swapedge[swapcellflag]#要交换的边
        swapcelldegree = markcell_degree[swapcellflag]
        swapcelldegreeflag = np.ones_like(swapcelldegree,dtype=np.bool_)
        swapcelldegreeflag[swapcelldegree==7] = False
        swapnode = swapcell[swapcelldegreeflag].reshape(-1,2)# 新边的节点
        
        # 删除单元 
        maskcell = np.ones(NC,dtype=np.bool_)
        deletecell = markcell_index[swapcellflag].reshape(-1)
        maskcell[deletecell] = False
        newcell0 = cell[maskcell]

        newcell1 = np.zeros((len(swapnode),3),dtype=np.int_)
        newcell2 = np.zeros((len(swapnode),3),dtype=np.int_)

        newcell1[:,:2] = swapnode
        newcell2[:,:2] = swapnode
        newcell1[:,2] = swapedge[:,0]
        newcell2[:,2] = swapedge[:,1]
        
        # 调整单元节点编号顺序
        newcell = np.r_[newcell1,newcell2]
        v1 = node[newcell[:,1]]-node[newcell[:,0]]
        v2 = node[newcell[:,2]]-node[newcell[:,0]]
        flag = np.cross(v1,v2)
        newcell[flag<0,0],newcell[flag<0,1] = newcell[flag<0,1],newcell[flag<0,0]

        newcell = np.r_[newcell0,newcell]# 新的单元
        self.ds.reinit(NN,newcell)

    def insert_node_optimize(self,angle_limit=88):
        node = self.entity('node')
        cell = self.entity('cell')
        edge = self.entity('edge')
        area = self.entity_measure('cell')
         
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        cell_index = np.arange(NC)

        node2node = self.ds.node_to_node()
        node2cell = self.ds.node_to_cell()
        edge2cell = self.ds.edge_to_cell()
        cell2edge = self.ds.cell_to_edge()
        degree = np.array(np.sum(node2node, axis=1)).reshape(-1)
        
        angle = self.angle()
        max_angle = np.max(angle,axis=1)
        angles = max_angle*(180/np.pi)
        markangle = (angles>angle_limit)

        markcell = cell[markangle]
        markcell_index = cell_index[markangle]

        rows, cols = markcell.shape
        mask = np.ones(rows, dtype=np.bool8)
        row_sets = [set(row) for row in markcell]

        for i in range(rows):
            if mask[i]:
                for j in range(i+1,rows):
                    if mask[j] and not row_sets[i].isdisjoint(row_sets[j]):
                        mask[j] = False
        markcell1 = markcell[mask]
        markcell1_index = markcell_index[mask]
        
        deletecell = np.zeros((len(markcell1_index),3),dtype = np.int64)
        
        deletecell[:,0] = markcell1_index
        markcell1_degree = degree[markcell1]
        markcell1_angle = angle[markcell1_index]
        angle_max_positions = np.argmax(markcell1_angle,axis=1)
        markcell_edgeindex = cell2edge[markcell1_index]
        markedge1 = markcell_edgeindex[np.arange(len(angle_max_positions)),angle_max_positions]
        markcell2_index = edge2cell[markedge1,:2]
        
        _,mask_index = np.where(markcell2_index==markcell1_index[:,None])
        mask_index = 1-mask_index
        markcell2_index = markcell2_index[np.arange(len(mask_index)),mask_index]
        deletecell[:,1] = markcell2_index
        
        markedge1_degree = degree[edge[markedge1]]
        
        edgedegree_min_positions = np.argmin(markedge1_degree,axis=1)
        aux_node = edge[markedge1][np.arange(len(edgedegree_min_positions)),edgedegree_min_positions]
        markcell2 = cell[markcell2_index]
        mask_edge2 = (markcell2 == aux_node[:,None])
        markedge2 = cell2edge[markcell2_index][mask_edge2]
        
        markcell3_index = edge2cell[markedge2,:2]
        _,mask_index = np.where(markcell3_index==markcell2_index[:,None])
        mask_index = 1-mask_index
        markcell3_index = markcell3_index[np.arange(len(mask_index)),mask_index]

        judge = (markedge1_degree[:,0]==markedge1_degree[:,1])
        aux_edge = markedge1[judge]
        aux_cell_index = markcell2_index[judge]
        aux_cell = cell[aux_cell_index]
        mask = (aux_cell == edge[aux_edge,0,None]) | (aux_cell == edge[aux_edge,1,None])
        aux_edge = cell2edge[aux_cell_index][mask].reshape(-1,2)
        global_index = edge2cell[aux_edge,:2]
        local_index = edge2cell[aux_edge,2:]
        mask = ~(global_index == aux_cell_index[:,None,None])
        global_index = global_index[mask].reshape(-1,2)
        local_index = local_index[mask].reshape(-1,2)
        
        rows,cols = local_index.shape
        node_index = cell[global_index][np.arange(rows)[:,None],np.arange(cols),local_index]
        node_degree = degree[node_index]
        np.argmin(node_degree,axis=1)
        markcell4 = global_index[np.arange(len(global_index)),np.argmin(node_degree,axis=1)]
        markedge3 = aux_edge[np.arange(len(global_index)),np.argmin(node_degree,axis=1)]

        judge2 = (node_degree[:,0]!=node_degree[:,1])

        if np.sum(judge2)>0:
            cell_area = area[global_index[judge2]]
            markcell4[judge2] = global_index[judge2][np.arange(len(cell_area)),np.argmax(cell_area,axis=1)]
            markedge3[judge2] = aux_edge[judge2][np.arange(len(cell_area)),np.argmax(cell_area,axis=1)]
        markcell3_index[judge] = markcell4
        markedge2[judge] = markedge3
        deletecell[:,2] = markcell3_index
        rows, cols = deletecell.shape
        mask = np.ones(rows, dtype=np.bool8)
        row_sets = [set(row) for row in deletecell]

        for i in range(rows):
            if mask[i]:
                for j in range(i+1,rows):
                    if mask[j] and not row_sets[i].isdisjoint(row_sets[j]):
                        mask[j] = False
        deletecell = deletecell[mask]
        vertices = cell[deletecell[:,::2]].reshape(-1,6)
        vertices = np.array([row[np.unique(row, return_index=True)[1]] for row in vertices])
        insert_node = np.sum(node[vertices],axis=1)/5

        mask_cell_flag = np.ones(NC,dtype=np.bool8)
        mask_cell_flag[deletecell.reshape(-1)] = False
        newcell0 = cell[mask_cell_flag]

        polygon_edge = cell2edge[deletecell].reshape(-1,9)
        polygon_edge_flag = (polygon_edge==markedge1[:,None]) | (polygon_edge==markedge2[:,None])
        polygon_edge = polygon_edge[~polygon_edge_flag].reshape(-1,5)
        newcell1 = np.zeros((len(insert_node),3),dtype=np.int64)
        newcell2 = np.zeros((len(insert_node),3),dtype=np.int64)
        newcell1[:,0] = np.arange(NN,NN+len(insert_node))
        newcell2[:,0] = np.arange(NN,NN+len(insert_node))
        newcell1[:,1] = edge[polygon_edge[:,0],0]
        newcell1[:,2] = edge[polygon_edge[:,0],1]
               
        for i in range(1,5):
            newcell2[:,1] = edge[polygon_edge[:,i],0]
            newcell2[:,2] = edge[polygon_edge[:,i],1]
            newcell1 = np.r_[newcell1,newcell2]
        
        node = np.r_[node,insert_node]
        v1 = node[newcell1[:,1]]-node[newcell1[:,0]]
        v2 = node[newcell1[:,2]]-node[newcell1[:,0]]
        flag = np.cross(v1,v2)
        newcell1[flag<0,0],newcell1[flag<0,1] = newcell1[flag<0,1],newcell1[flag<0,0]
        newcell = np.r_[newcell0,newcell1]
        self.node = node
        NN = self.node.shape[0]
        self.ds.reinit(NN,newcell)

    @staticmethod
    def adaptive_options(
            method='mean',
            maxrefine=5,
            maxcoarsen=0,
            theta=1.0,
            tol=1e-6,  # 目标误差
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
                np.log2(eta / (theta * np.mean(eta)))
            )
        elif options['method'] == 'max':
            options['numrefine'] = np.around(
                np.log2(eta / (theta * np.max(eta)))
            )
        elif options['method'] == 'median':
            options['numrefine'] = np.around(
                np.log2(eta / (theta * np.median(eta)))
            )
        elif options['method'] == 'min':
            options['numrefine'] = np.around(
                np.log2(eta / (theta * np.min(eta)))
            )
        elif options['method'] == 'target':
            NT = self.number_of_cells()
            e = options['tol'] / np.sqrt(NT)
            options['numrefine'] = np.around(
                np.log2(eta / (theta * e)
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
        node = np.zeros((5 * NN, GD), dtype=self.ftype)
        cell = np.zeros((3 * NC, 3), dtype=self.itype)

        if ('numrefine' in options) and (options['numrefine'] is not None):
            options['numrefine'] = np.r_[options['numrefine'], np.zeros(2 * NC)]

        node[:NN] = self.entity('node')
        cell[:NC] = self.entity('cell')

        # 用于存储网格节点的代数，初始所有节点都为第 0 代
        generation = np.zeros(NN + 2 * NC, dtype=np.uint8)

        # 用于记录被二分的边及其中点编号
        cutEdge = np.zeros((4 * NN, 3), dtype=self.itype)

        # 当前的二分边的数目
        nCut = 0
        # 非协调边的标记数组
        nonConforming = np.ones(4 * NN, dtype=np.bool_)
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

            if nCut == 0:  # 如果是第一次循环
                idx = np.arange(nMarked)  # cells introduce new cut edges
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
                i, j = np.nonzero(nv2v[:, p1].multiply(nv2v[:, p2]))
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
                newCutEdge = np.arange(nCut, nCut + nNew)
                cutEdge[newCutEdge, 0] = i
                cutEdge[newCutEdge, 1] = j
                cutEdge[newCutEdge, 2] = range(NN, NN + nNew)
                node[NN:NN + nNew, :] = (node[i, :] + node[j, :]) / 2.0
                nCut += nNew
                NN += nNew

                # 新点和旧点的邻接矩阵
                I = cutEdge[newCutEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[newCutEdge][:, [0, 1]].reshape(-1)
                val = np.ones(len(I), dtype=np.bool_)
                nv2v = csr_matrix(
                    (val, (I, J)),
                    shape=(NN, NN))
                i, j = np.nonzero(nv2v[:, p1].multiply(nv2v[:, p2]))
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
            cell[NC:NC + nMarked, 0] = p3
            cell[NC:NC + nMarked, 1] = p2
            cell[NC:NC + nMarked, 2] = p0

            if ('numrefine' in options) and (options['numrefine'] is not None):
                options['numrefine'][markedCell] -= 1
                options['numrefine'][NC:NC + nMarked] = options['numrefine'][markedCell]

            NC = NC + nMarked
            del cellGeneration, p0, p1, p2, p3

            # 找到非协调的单元
            checkEdge, = np.nonzero(nonConforming[:nCut])
            isCheckNode = np.zeros(NN, dtype=np.bool_)
            isCheckNode[cutEdge[checkEdge]] = True
            isCheckCell = np.sum(
                isCheckNode[cell[:NC]],
                axis=-1) > 0
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
                val[markedNode - NN0, :] = 0.5 * (ic + jc)
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
            a = nv / 2.0
        elif GD == 3:
            a = np.sqrt(np.square(nv).sum(axis=1)) / 2.0
        return a

    def point_to_bc(self, point):
        """
        @brief 找到定点 point 所在的单元，并计算其重心坐标 
        """
        index = self.location(point)
        node = self.node
        cell = self.entity('cell')
        cm = self.cell_area()[index]
        point = point[:, np.newaxis, :]
        v = node[cell[index]] - point
        a0 = 0.5 * np.abs(np.cross(v[:, 1, :], v[:, 2, :]))
        a1 = 0.5 * np.abs(np.cross(v[:, 0, :], v[:, 2, :]))
        a2 = 0.5 * np.abs(np.cross(v[:, 0, :], v[:, 1, :]))
        result = np.zeros((index.shape[0], 3))
        result[:, 0] = a0 / cm
        result[:, 1] = a1 / cm
        result[:, 2] = a2 / cm
        return index, result

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
        tangle = np.zeros(NN, dtype=self.itype)  # 记录界面转角
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
        l = np.sqrt(np.sum(v ** 2, axis=1))
        if hmax is None:
            lmax = np.max(l)
            isBigSizeCell = (l > lmax * lmax) & isInterfaceCell
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
        cellType = (np.abs(v[:, 0]) > 0.0) & (np.abs(v[:, 1]) > 0.0)  # TODO: 0.0-->eps
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

            m03 = (p0 + p3) / 2
            m23 = (p2 + p3) / 2
            m12 = (p1 + p2) / 2
            m0 = (m12 + p2) / 2
            m1 = (m12 + p1) / 2
            isNotTB = ((phi[cell[idx, 2]] * interface(m0) > 0.0) &
                       (interface(m03) * interface(m23) < 0.0)) | (interface(m0) * interface(m1) < 0.0)

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
            m03 = (p0 + p3) / 2
            m13 = (p1 + p3) / 2
            m12 = (p1 + p2) / 2
            m0 = (m12 + p2) / 2
            m1 = (m12 + p1) / 2
            isNotTB = ((phi[cell[idx, 1]] * interface(m1) > 0.0) &
                       (interface(m03) * interface(m13) < 0.0)) | (interface(m0) * interface(m1) < 0.0)

            isTransitionTypeBCell[idx[isNotTB]] = False

        # Case 3:
        n1 = cell2cell[:, 1]
        flag = isTransitionTypeBCell & isInterfaceCell[n1] & ~cellType[n1]
        if np.any(flag):
            idx, = np.nonzero(flag)
            p1 = node[cell[flag, 1]]
            p2 = node[cell[flag, 2]]
            m12 = (p1 + p2) / 2
            m0 = (m12 + p2) / 2
            m1 = (m12 + p2) / 2
            isNotTB = interface(m0) * interface(m1) < 0.0
            isTransitionTypeBCell[idx[isNotTB]] = False

        # Case 4:
        n2 = cell2cell[:, 2]
        flag = isTransitionTypeBCell & isInterfaceCell[n2] & (~cellType[n2])
        if np.any(flag):
            idx, = np.nonzero(flag)
            p1 = node[cell[idx, 1]]
            p2 = node[cell[idx, 2]]
            m12 = (p1 + p2) / 2
            m0 = (m12 + p2) / 2
            m1 = (m12 + p1) / 2
            isNotTB = interface(m0) * interface(m1) < 0.0
            isTransitionTypeBCell[idx[isNotTB]] = False

        isTypeBCell = isMark & (~cellType) & (~isTransitionTypeBCell)
        return isTypeBCell, cellType

    def bisect_interface_cell_with_curvature(self, interface, hmax):
        """
        """
        NN = self.number_of_nodes()
        node = self.entity('node')

        phi = interface(node)

        if np.all(phi < 0):
            raise ValueError('初始网格在界面围成区域的内部，需要更换一个可以覆盖界面的网格')

        # Step 1: 一致二分法加密网格
        while np.all(phi > 0):
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

    def show_function(self, plot, uh, cmap=None):
        """
        """
        from types import ModuleType
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        from mpl_toolkits.mplot3d import Axes3D
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = plot.axes(projection='3d')
        else:
            axes = plot

        node = self.entity('node')
        cell = self.entity('cell')
        cax = axes.plot_trisurf(
            node[:, 0], node[:, 1],
            uh, triangles=cell, cmap=cmap, lw=0.0)
        axes.figure.colorbar(cax, ax=axes)
        return axes

    @classmethod
    def show_multi_index(cls, p=1):
        """
        """
        pass

    @classmethod
    def show_lattice(cls, p=1, showmultiindex=False):
        """
        @brief 展示三角形上的单纯形格点
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        if showmultiindex:
            n = 3
        else:
            n = 2

        mesh = cls.from_one_triangle('equ')  # 返回只有一个单位等边三角形的网格
        node = mesh.entity('node')
        ips = mesh.interpolation_points(p)
        c2p = mesh.cell_to_ipoint(p)
        ips = ips[c2p].reshape(-1, 2)

        fig = plt.figure()
        axes = fig.add_subplot(1, n, 1)
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True, fontcolor='k')

        axes = fig.add_subplot(1, n, 2)
        mesh.add_plot(axes)
        mesh.find_node(axes, node=ips, showindex=True)
        triangulation = mtri.Triangulation(ips[:, 0], ips[:, 1])
        axes.triplot(triangulation, color='black', linestyle='dashed')

        if showmultiindex:
            axes = fig.add_subplot(1, n, 3)
            mesh.add_plot(axes)
            mesh.find_node(axes, node=ips)
            mi = mesh.multi_index_matrix(p, 2)
            for i, idx in enumerate(mi):
                s = str(idx).replace('[', '(')
                s = s.replace(']', ')')
                s = s.replace(' ', ',')
                axes.text(ips[i, 0], ips[i, 1], s,
                          multialignment='center',
                          fontsize=12,
                          color='r')
        plt.show()

    @classmethod
    def show_shape_function(cls, p=1, funtype='L'):
        """
        @brief 可视化展示三角形单元上的 p 次基函数
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        mesh = cls.from_one_triangle('equ')  # 返回只有一个单位等边三角形的网格
        TD = mesh.top_dimension()
        ldof = mesh.number_of_local_ipoints(p)

        if p % 2 == 0:
            m = (p + 2) // 2
            n = p + 1
        else:
            m = (p + 1) // 2
            n = p + 2

        node = mesh.entity('node')
        ips = mesh.interpolation_points(p)
        c2p = mesh.cell_to_ipoint(p)
        ips = ips[c2p].reshape(-1, 2)
        bcs = mesh.multi_index_matrix(10 * p, TD) / 10 / p
        ps = mesh.bc_to_point(bcs).reshape(len(bcs), -1)
        if funtype == 'L':
            phi = mesh.shape_function(bcs, p)
        elif funtype == 'B':
            phi = mesh._bernstein_shape_function(bcs, p)
        fig = plt.figure()
        for i in range(ldof):
            axes = fig.add_subplot(m, n, i + 1, projection='3d')
            axes.plot_trisurf(node[:, 0], node[:, 1], np.zeros(3),
                              color='#99BBF6', alpha=0.5)

            for j in range(3):
                axes.scatter(node[j, 0], node[j, 1], 0.0, color='k')
                axes.text(node[j, 0], node[j, 1], 0.0, f'$x_{j}$', color='k')

            axes.scatter(ips[i, 0], ips[i, 1], 1.0, color='r')
            axes.text(ips[i, 0], ips[i, 1], 1 + 0.02, f'$p_{i}$', color='r')

            axes.plot([ips[i, 0], ips[i, 0]], [ips[i, 1], ips[i, 1]], [0.0,
                                                                       1.0], 'r--')

            axes.plot_trisurf(ps[:, 0], ps[:, 1], phi[:, i], cmap='viridis',
                              linewidths=0)
            if p == 1:
                axes.set_title(f'$\phi_{{{i}}}=\lambda_{{{i}}}$')
            else:
                axes.set_title(f'$\phi_{{{i}}}$')
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel('Z')
        plt.show()

    @classmethod
    def show_global_basis_function(cls, p=3):
        """
        @brief 展示通过单元基函数的拼接+零扩展的方法获取整体基函数的过程
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        mesh = cls.from_unit_sq(nx=3, ny=3)
        node = mesh.entity('node')

    @classmethod
    def show_grad_shape_function(cls, p, funtype='L'):
        """
        """
        import matplotlib.pyplot as plt

        mesh = cls.from_one_triangle('equ')  # 返回只有一个单位等边三角形的网格
        TD = mesh.top_dimension()
        ldof = mesh.number_of_local_ipoints(p)

        if p % 2 == 0:
            m = (p + 2) // 2
            n = p + 1
        else:
            m = (p + 1) // 2
            n = p + 2

        node = mesh.entity('node')
        ips = mesh.interpolation_points(p)
        c2p = mesh.cell_to_ipoint(p)
        ips = ips[c2p].reshape(-1, 2)
        bcs = mesh.multi_index_matrix(p, TD) / p
        ps = mesh.bc_to_point(bcs).reshape(len(bcs), -1)
        if funtype == 'L':
            gphi = mesh.grad_shape_function(bcs, p)  # (NQ, NC, ldof, GD)
        elif funtype == 'B':
            R = mesh._grad_bernstein_shape_function(bcs, p)
            Dlambda = mesh.grad_lambda()
            gphi = np.einsum('...ij, kjm->...kim', R, Dlambda)

        fig = plt.figure()
        for i in range(ldof):
            axes = fig.add_subplot(m, n, i + 1)
            mesh.add_plot(axes)
            mesh.find_node(axes, node=ips, showindex=True)
            axes.quiver(ps[:, 0], ps[:, 1], gphi[:, 0, i, 0], gphi[:, 0, i, 1],
                        units='xy')
            axes.set_title(f'$\\nabla\\phi_{{{i}}}$')
        plt.show()

    @classmethod
    def from_obj_file(cls, fname):
        """
        @brief 从一个 obj 文件中读取一个三角形网格面
        """
        nodes = []
        cells = []
        with open(fname, 'r') as file:
            for line in file:
                components = line.strip(' \n').split(' ')
                if components[0] == 'v':
                    nodes.append([float(components[1]), float(components[2]), float(components[3])])
                elif components[0] == 'f':
                    # Subtract 1 from each index to account for 1-based indexing in OBJ files
                    cells.append([int(i) - 1 for i in components[1:]])

        return cls(np.array(nodes), np.array(cells))

    ## @ingroup MeshGenerators
    @classmethod
    def from_quadrangle_mesh(cls, mesh):
        """
        @brief 把一个四边形网格中每个单元分成两个三角形
        """
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        localCell = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int_)
        cell = cell[:, localCell].reshape(-1, 3)
        tmesh = cls(node, cell)
        return tmesh

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
        NN = (nx + 1) * (ny + 1)
        NC = nx * ny
        node = np.zeros((NN, 2))
        X, Y = np.mgrid[
               box[0]:box[1]:(nx + 1) * 1j,
               box[2]:box[3]:(ny + 1) * 1j]
        node[:, 0] = X.flat
        node[:, 1] = Y.flat

        idx = np.arange(NN).reshape(nx + 1, ny + 1)
        cell = np.zeros((2 * NC, 3), dtype=np.int_)
        cell[:NC, 0] = idx[1:, 0:-1].flatten(order='F')
        cell[:NC, 1] = idx[1:, 1:].flatten(order='F')
        cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 2] = idx[1:, 1:].flatten(order='F')

        if threshold is not None:
            bc = np.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = np.zeros(NN, dtype=np.bool_)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = np.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]

        return cls(node, cell)

    @classmethod
    def from_meshio(cls, file, show=False):
        import meshio
        data = meshio.read(file)
        node = data.points
        cell = data.cells_dict['triangle']
        #print(data.cells_dict)
        mesh = cls(node, cell)
        if show:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            mesh.add_plot(ax)
            plt.show()
        return mesh

        ## @ingroup MeshGenerators

    @classmethod
    def from_one_triangle(cls, meshtype='iso'):
        if meshtype == 'equ':
            node = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3) / 2]], dtype=np.float64)
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
    def from_domain_distmesh(cls, domain, maxit=100, output=False):
        from .distmesher_2d import DistMesher2d
        mesher = DistMesher2d(domain, domain.hmin, output=output)
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
        gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1, 1, 1)

        # 同步几何模型
        gmsh.model.occ.synchronize()

        # 设置网格尺寸
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

        # 生成网格
        gmsh.model.mesh.generate(2)

        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = node_coords.reshape((-1, 3))[:, :2]

        # 节点编号映射
        nodetags_map = dict({j: i for i, j in enumerate(node_tags)})

        # 获取单元信息
        cell_type = 2  # 三角形单元的类型编号为 2
        cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)

        # 节点编号映射到单元
        evid = np.array([nodetags_map[j] for j in cell_connectivity])
        cell = evid.reshape((cell_tags.shape[-1], -1))

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
            line = gmsh.model.geo.addLine(polygon_points[i], polygon_points[(i + 1) % len(polygon_points)])
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
    def from_fuel_rod_gmsh(cls,R1,R2,L,w,h,meshtype='normal'):
        """
        Generate a trangle mesh for a fuel-rod region by gmsh

        @param R1 The radius of semicircles
        @param R2 The radius of quarter circles
        @param L The length of straight segments
        @param w The thickness of caldding
        @param h Parameter controlling mesh density
        @param meshtype Choose whether to add mesh refinement at the boundary
        @return TriangleMesh instance
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add("fuel_rod_2D")

        # 内部单元大小
        Lc1 = h
        # 包壳单元大小
        Lc2 = h/2.5

        factory = gmsh.model.geo
        # 外圈点
        factory.addPoint( -R1 -R2 -L, 0 , 0 , Lc2 , 1 )#圆心1
        factory.addPoint( -R1 -R2 -L, -R1 , 0 , Lc2 , 2)
        factory.addPoint( -R1 -R2 , -R1 , 0 , Lc2 , 3)
        factory.addPoint( -R1 -R2 , -R1 -R2 , 0 , Lc2 , 4)#圆心2
        factory.addPoint( -R1 , -R1 -R2 , 0 , Lc2 , 5)
        factory.addPoint( -R1 , -R1 -R2 -L , 0 , Lc2 , 6)
        factory.addPoint( 0 , -R1 -R2 -L , 0 , Lc2 , 7)#圆心3
        factory.addPoint( R1 , -R1 -R2 -L , 0 , Lc2 , 8)
        factory.addPoint( R1 , -R1 -R2 , 0 , Lc2 , 9)
        factory.addPoint( R1 +R2 , -R1 -R2 , 0, Lc2 , 10)#圆心4
        factory.addPoint( R1 +R2 , -R1 , 0 , Lc2 , 11) 
        factory.addPoint( R1 +R2 +L , -R1 , 0 , Lc2 , 12)
        factory.addPoint( R1 +R2 +L , 0 , 0 , Lc2 , 13)#圆心5
        factory.addPoint( R1 +R2 +L , R1 , 0 , Lc2 , 14)
        factory.addPoint( R1 +R2 , R1 , 0 , Lc2 , 15)
        factory.addPoint( R1 +R2 , R1 +R2 , 0 , Lc2 , 16)#圆心6
        factory.addPoint( R1 , R1 +R2 , 0 , Lc2 , 17)
        factory.addPoint( R1 , R1 +R2 +L , 0 , Lc2 , 18)
        factory.addPoint( 0 , R1 +R2 +L , 0 , Lc2 , 19)#圆心7
        factory.addPoint( -R1 , R1 +R2 +L , 0 , Lc2 , 20)
        factory.addPoint( -R1 , R1 +R2 , 0 , Lc2 , 21)
        factory.addPoint( -R1 -R2 , R1 +R2 , 0 , Lc2 , 22)#圆心8
        factory.addPoint( -R1 -R2 , R1 , 0 , Lc2 , 23)
        factory.addPoint( -R1 -R2 -L , R1 , 0 , Lc2 , 24)

        # 外圈线
        line_list_out = []
        for i in range(8):
            if i == 0:
                factory.addCircleArc(24 , 3*i+1 , 3*i+2, 2*i+1)
                factory.addLine( 3*i+2 , 3*i+3 , 2*(i+1) )
            else:
                factory.addCircleArc(3*i , 3*i+1 , 3*i+2 , 2*i+1)
                factory.addLine( 3*i+2 , 3*i+3 , 2*(i+1) )
            # 填充线环中的线
            line_list_out.append(2*i+1)
            line_list_out.append(2*(i+1))
        # 生成外圈线环
        factory.addCurveLoop(line_list_out,17)

        # 内圈点
        factory.addPoint( -R1 -R2 -L, -R1 +w , 0 , Lc1 , 25)
        factory.addPoint( -R1 -R2 , -R1 +w , 0 , Lc1 , 26)
        factory.addPoint( -R1 +w , -R1 -R2 , 0 , Lc1 , 27)
        factory.addPoint( -R1 +w , -R1 -R2 -L , 0 , Lc1 , 28)
        factory.addPoint( R1 -w , -R1 -R2 -L , 0 , Lc1 , 29)
        factory.addPoint( R1 -w , -R1 -R2 , 0 , Lc1 , 30)
        factory.addPoint( R1 +R2 , -R1 +w , 0 , Lc1 , 31) 
        factory.addPoint( R1 +R2 +L , -R1 +w , 0 , Lc1 , 32)
        factory.addPoint( R1 +R2 +L , R1 -w , 0 , Lc1 , 33)
        factory.addPoint( R1 +R2 , R1 -w , 0 , Lc1 , 34)
        factory.addPoint( R1 -w , R1 +R2 , 0 , Lc1 , 35)
        factory.addPoint( R1 -w , R1 +R2 +L , 0 , Lc1 , 36)
        factory.addPoint( -R1 +w , R1 +R2 +L , 0 , Lc1 , 37)
        factory.addPoint( -R1 +w , R1 +R2 , 0 , Lc1 , 38)
        factory.addPoint( -R1 -R2 , R1 -w, 0 , Lc1 , 39)
        factory.addPoint( -R1 -R2 -L , R1 -w, 0 , Lc1 , 40)

        # 内圈线
        line_list_in = []
        for j in range(8):
            if j == 0:
                factory.addCircleArc(40 , 3*j+1 , 25+2*j , 18+2*j)
                factory.addLine(25+2*j , 26+2*j , 19+2*j)
            else:
                factory.addCircleArc(24+2*j , 3*j+1 , 25+2*j, 18+2*j)
                factory.addLine(25+2*j , 26+2*j , 19+2*j)
            line_list_in.append(18+2*j)
            line_list_in.append(19+2*j)
        # 生成内圈线环  
        factory.addCurveLoop(line_list_in,34)

        # 内圈面
        factory.addPlaneSurface([34],35)
        # 包壳截面
        factory.addPlaneSurface([17, 34],36)

        factory.synchronize()

        if meshtype == 'refine':
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmmf = gmsh.model.mesh.field
            gmmf.add("Distance",1)
            gmmf.setNumbers(1, "CurvesList",line_list_in)
            gmmf.setNumber(1,"Sampling",1000)
            gmmf.add("Threshold",2)
            gmmf.setNumber(2, "InField", 1)
            gmmf.setNumber(2, "SizeMin", Lc1/5)
            gmmf.setNumber(2, "SizeMax", Lc1)
            gmmf.setNumber(2, "DistMin", w)
            gmmf.setNumber(2, "DistMax", 2*w)
            gmmf.setAsBackgroundMesh(2)

        # 生成网格
        gmsh.model.mesh.generate(2)

        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)[:, 0:2].copy()

        # 获取三角形单元信息
        cell_type = 2  # 三角形单元的类型编号为 2
        cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)
        cell = np.array(cell_connectivity, dtype=np.int_).reshape(-1, 3) - 1


        gmsh.finalize()
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of cells: {cell.shape[0]}")

        NN = len(node)
        isValidNode = np.zeros(NN, dtype=np.bool_)
        isValidNode[cell] = True
        node = node[isValidNode]
        idxMap = np.zeros(NN, dtype=cell.dtype)
        idxMap[isValidNode] = range(isValidNode.sum())
        cell = idxMap[cell]

        return cls(node,cell)
    
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
        NN = nu * nv
        NC = nu * nv
        node = np.zeros((NN, 3), dtype=np.float64)

        U, V = np.mgrid[0:2 * np.pi:nu * 1j, 0:2 * np.pi:nv * 1j]
        X = (R + r * np.cos(V)) * np.cos(U)
        Y = (R + r * np.cos(V)) * np.sin(U)
        Z = r * np.sin(V)
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        node[:, 2] = Z.flatten()

        idx = np.zeros((nu + 1, nv + 1), np.int_)
        idx[0:-1, 0:-1] = np.arange(NN).reshape(nu, nv)
        idx[-1, :] = idx[0, :]
        idx[:, -1] = idx[:, 0]
        cell = np.zeros((2 * NC, 3), dtype=np.int_)
        cell[:NC, 0] = idx[1:, 0:-1].flatten(order='F')
        cell[:NC, 1] = idx[1:, 1:].flatten(order='F')
        cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 2] = idx[1:, 1:].flatten(order='F')

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_sphere_surface(cls, refine=0):
        """
        @brief  Generate a triangular mesh on a unit sphere surface.
        @return the triangular mesh.
        """
        t = (np.sqrt(5) - 1) / 2
        node = np.array([
            [0, 1, t], [0, 1, -t], [1, t, 0], [1, -t, 0],
            [0, -1, -t], [0, -1, t], [t, 0, 1], [-t, 0, 1],
            [t, 0, -1], [-t, 0, -1], [-1, t, 0], [-1, -t, 0]], dtype=np.float64)
        cell = np.array([
            [6, 2, 0], [3, 2, 6], [5, 3, 6], [5, 6, 7],
            [6, 0, 7], [3, 8, 2], [2, 8, 1], [2, 1, 0],
            [0, 1, 10], [1, 9, 10], [8, 9, 1], [4, 8, 3],
            [4, 3, 5], [4, 5, 11], [7, 10, 11], [0, 10, 7],
            [4, 11, 9], [8, 4, 9], [5, 7, 11], [10, 9, 11]], dtype=np.int_)
        mesh = cls(node, cell)
        mesh.uniform_refine(refine)
        node = mesh.node
        cell = mesh.entity('cell')
        d = np.sqrt(node[:, 0] ** 2 + node[:, 1] ** 2 + node[:, 2] ** 2) - 1
        l = np.sqrt(np.sum(node ** 2, axis=1))
        n = node / l[..., np.newaxis]
        node = node - d[..., np.newaxis] * n
        return cls(node, cell)

    @classmethod
    def from_box_with_interface(cls, interface, box, nx=10, ny=10):
        """
        """
        from scipy.spatial import Delaunay
        from .uniform_mesh_2d import UniformMesh2d

    @classmethod
    def interfacemesh_generator(cls, box: Sequence[float], nx: int, ny: int, phi):
        """
        @brief Generate a triangle mesh fitting the interface.

        @param box:
        @param nx, ny:
        @param phi:

        @return: TriangleMesh.
        """
        from scipy.spatial import Delaunay
        from fealpy.mesh.uniform_mesh_2d import UniformMesh2d

        mesh = UniformMesh2d((0, nx, 0, ny), ((box[1] - box[0]) / nx, (box[3] - box[2]) / ny), (box[0], box[2]))

        interfaceNode, isInterfaceNode, isInterfaceCell, ncut, naux = mesh.find_interface_node(phi)

        N = mesh.number_of_nodes()
        cell = mesh.entity('cell')[:, [0, 2, 3, 1]]
        node = mesh.entity('node')

        dt = Delaunay(interfaceNode)
        tri = dt.simplices
        NI = np.sum(isInterfaceNode)
        isUnnecessaryCell = (np.sum(tri < NI, axis=1) == 3)
        tri = tri[~isUnnecessaryCell, :]

        interfaceNodeIdx = np.zeros(interfaceNode.shape[0], dtype=np.int_)
        interfaceNodeIdx[:NI], = np.nonzero(isInterfaceNode)
        interfaceNodeIdx[NI:NI + ncut] = N + np.arange(ncut)
        interfaceNodeIdx[NI + ncut:] = N + ncut + np.arange(naux)
        tri = interfaceNodeIdx[tri]

        NS = np.sum(~isInterfaceCell)
        NT = tri.shape[0]
        pnode = np.concatenate((node, interfaceNode[NI:]), axis=0)
        pcell = np.zeros((NS * 2 + NT, 3), dtype=np.int_)
        temp = cell[~isInterfaceCell, :]
        pcell[0:NS, :] = temp[:, [1, 2, 0]]
        pcell[NS:2 * NS, :] = temp[:, [3, 0, 2]]
        pcell[2 * NS:, :] = tri

        pmesh = cls(pnode, pcell)

        return pmesh

    @classmethod
    def from_ellipsoid_surface(cls, ntheta=10, nphi=10,
                               radius=(1, 1, 1),
                               theta=(np.pi / 4, 3 * np.pi / 4),
                               phi=None,
                               returnuv=False
                               ):
        """
        @brief 给定椭球面的三个轴半径 radius=(a, b, c)，以及天顶角 theta 的范围,
        生成相应带状区域的三角形网格

        x = a \sin\theta \cos\phi
        y = b \sin\theta \sin\phi
        z = c \cos\theta

        @param[in] ntheta \theta 方向的剖分段数
        @param[in] nphi \phi 方向的剖分段数 
        """

        a, b, c = radius
        if phi is None:  # 默认为一封闭的带状区域
            NN = (ntheta + 1) * nphi
        else:  # 否则为四边形区域
            NN = (ntheta + 1) * (nphi + 1)

        NC = ntheta * nphi

        if phi is None:
            U, V = np.mgrid[
                   theta[0]:theta[1]:(ntheta + 1) * 1j,
                   0:2 * np.pi:(nphi + 1) * 1j]
            U = U[:, 0:-1]  # 去掉最后一列
            V = V[:, 0:-1]  # 去年最后一列
        else:
            U, V = np.mgrid[
                   theta[0]:theta[1]:(ntheta + 1) * 1j,
                   phi[0]:phi[1]:(nphi + 1) * 1j]

        node = np.zeros((NN, 3), dtype=np.float64)
        X = a * np.sin(U) * np.cos(V)
        Y = b * np.sin(U) * np.sin(V)
        Z = c * np.cos(U)
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        node[:, 2] = Z.flatten()

        idx = np.zeros((ntheta + 1, nphi + 1), np.int_)
        if phi is None:
            idx[:, 0:-1] = np.arange(NN).reshape(ntheta + 1, nphi)
            idx[:, -1] = idx[:, 0]
        else:
            idx = np.arange(NN).reshape(ntheta + 1, nphi + 1)
        cell = np.zeros((2 * NC, 3), dtype=np.int_)
        cell[0::2, 0] = idx[1:, 0:-1].flatten(order='F')
        cell[0::2, 1] = idx[1:, 1:].flatten(order='F')
        cell[0::2, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[1::2, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[1::2, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[1::2, 2] = idx[1:, 1:].flatten(order='F')

        if returnuv:
            return cls(node, cell), U.flatten(), V.flatten()
        else:
            return cls(node, cell)

    @classmethod
    def from_cylinder_surface(cls, nphi=10, nz=10,
                              radius=(1.0, 1.0),
                              height=2.0
                              ):

        a, b = radius
        h = height
        NN = nphi * (nz + 1)
        NC = nphi * nz

        U, V = np.mgrid[
               0:h:(nz + 1) * 1j,
               0:2 * np.pi:(nphi + 1) * 1j]
        U = U[:, 0:-1]
        V = V[:, 0:-1]

        node = np.zeros((NN, 3), dtype=np.float64)
        X = a * np.cos(V)
        Y = b * np.sin(V)
        Z = U
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        node[:, 2] = Z.flatten()

        idx = np.zeros((ntheta + 1, nphi + 1), np.int_)
        idx[:, 0:-1] = np.arange(NN).reshape(ntheta + 1, nphi)
        idx[:, -1] = idx[:, 0]
        cell = np.zeros((2 * NC, 3), dtype=np.int_)
        cell[0::2, 0] = idx[1:, 0:-1].flatten(order='F')
        cell[0::2, 1] = idx[1:, 1:].flatten(order='F')
        cell[0::2, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[1::2, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[1::2, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[1::2, 2] = idx[1:, 1:].flatten(order='F')

        return cls(node, cell)

    @classmethod
    def from_section_ellipsoid(
            cls,
            size=(17.5, 3.47, 3),
            center_height=6,
            scale_ratio=(1, 1, 1),
            density=0.1,
            top_section=np.pi / 2,
            return_edge=False):
        """
        构造包围一个长方体的椭球面网格，该椭球面被两个平面截取，并将底部填充
        @param size: 长方体的长宽高
        @param center_height: 椭球面球心的高度
        @param scale_ratio: 椭球面三个半轴的缩放比例
        @param density: 网格密度
        @param top_section: 顶部截面对应天顶角
        @param return_edge: 是否需要返回分区交界边上的点
        @return: 截断的椭球面网格
        """
        l, w, h = size
        # 构造椭球面
        a = l * scale_ratio[0]
        b = w * scale_ratio[1]
        c = h * scale_ratio[2]

        t = np.sqrt(c ** 2 - center_height ** 2) / c
        theta = (top_section, np.arccos(-center_height / c))
        phi1 = np.array([[np.arctan(w / l), -np.arctan(w / l)],
                         [-np.arctan(w / l), -np.pi / 2],
                         [-np.pi / 2, -np.pi + np.arctan(w / l)],
                         [-np.pi + np.arctan(w / l), -np.pi - np.arctan(w / l)],
                         [-np.pi - np.arctan(w / l), -1.5 * np.pi],
                         [-1.5 * np.pi, -2 * np.pi + np.arctan(w / l)]])
        temp_cof = a / b * t * 1.236
        phi2 = np.array([[temp_cof * np.arctan(w / l), -temp_cof * np.arctan(w / l)],
                         [-temp_cof * np.arctan(w / l), -np.pi / 2],
                         [-np.pi / 2, -np.pi + temp_cof * np.arctan(w / l)],
                         [-np.pi + temp_cof * np.arctan(w / l), -np.pi - temp_cof * np.arctan(w / l)],
                         [-np.pi - temp_cof * np.arctan(w / l), -1.5 * np.pi],
                         [-1.5 * np.pi, -2 * np.pi + temp_cof * np.arctan(w / l)]])
        nphi = int(1 / density * a / b)
        ntheta = int(1 / density * c)

        assert (a * t / 2 ** (1 / 2)) > (l / 2), "椭球面球心高度超出范围"
        assert (b * t / 2 ** (1 / 2)) > w / 2, "椭球面球心高度超出范围"
        dphi1 = np.linspace(phi1[:, 0], phi1[:, 1], nphi + 1, axis=1)
        dphi2 = np.linspace(phi2[:, 0], phi2[:, 1], nphi + 1, axis=1)
        rectangle_node = [[l / 2, l / 2 * np.tan(dphi1[0])],
                          [-w / 2 / np.tan(dphi1[1]), -w / 2],
                          [-w / 2 / np.tan(dphi1[2]), -w / 2],
                          [-l / 2, -l / 2 * np.tan(dphi1[3])],
                          [w / 2 / np.tan(dphi1[4]), w / 2],
                          [w / 2 / np.tan(dphi1[5]), w / 2]]

        # 根据输入长宽高自适应段数，待完善
        nthetas = [ntheta, ntheta]
        nphis = [nphi, nphi, nphi, nphi, nphi, nphi]

        NN = (sum(nthetas) + 1) * sum(nphis)
        node = np.zeros((NN, 3)).reshape((sum(nthetas) + 1, sum(nphis), -1))

        for i in range(6):
            NN1 = (nthetas[0] + 1) * (nphis[i] + 1)
            line1 = np.zeros((nphis[i] + 1, 3))
            line2 = np.zeros((nphis[i] + 1, 3))
            line1[:, 0] = rectangle_node[i][0]
            line1[:, 1] = rectangle_node[i][1]
            line1[:, 2] = -center_height
            line2[:, 0] = a * t * np.cos(dphi2[i])
            line2[:, 1] = b * t * np.sin(dphi2[i])
            line2[:, 2] = -center_height
            node2 = np.linspace(line2, line1, nthetas[1] + 1)

            U, V = np.mgrid[
                   theta[0]:theta[1]:(ntheta + 1) * 1j,
                   phi2[i, 0]:phi2[i, 1]:(nphi + 1) * 1j]

            node1 = np.zeros((NN1, 3), dtype=np.float64).reshape((nthetas[0] + 1, nphis[i] + 1, 3))
            X = a * np.sin(U) * np.cos(V)
            Y = b * np.sin(U) * np.sin(V)
            Z = c * np.cos(U)
            node1[..., 0] = X
            node1[..., 1] = Y
            node1[..., 2] = Z

            node[0:nthetas[0] + 1, sum(nphis[0:i]):(sum(nphis[0:i]) + nphis[i]), :] = node1[:, 0:-1, :]
            node[nthetas[0] + 1:, sum(nphis[0:i]):(sum(nphis[0:i]) + nphis[i]), :] = node2[1:, 0:-1, :]
        # 处理分区交界边
        if return_edge:
            edge_node = np.zeros((6, sum(nthetas) + 1, 3))
            for i in range(6):
                edge_node[i, ...] = node[:, sum(nphis[0:(i + 1) % 6]), :]

        # 处理中间区域，构造节点
        central_node = np.zeros((nphis[1] + nphis[2] + 1, nphis[0] + 1, 3))
        central_node[0, ...] = node[sum(nthetas), :nphis[0] + 1]
        central_node[-1, ...] = node[sum(nthetas), sum(nphis[0:3]):sum(nphis[0:4]) + 1][::-1, ...]
        central_node[:, -1, :] = node[sum(nthetas), sum(nphis[0:1]):sum(nphis[0:3]) + 1]
        central_node[1:, 0, :] = node[sum(nthetas), sum(nphis[0:4]):sum(nphis[0:6]) + 1][::-1, ...]

        central_node[..., 0] = np.linspace(central_node[:, 0, 0], central_node[:, -1, 0], nphis[0] + 1, axis=0).T
        central_node[..., 1] = np.linspace(central_node[0, :, 1], central_node[-1, :, 1], nphis[1] + nphis[2] + 1,
                                           axis=0)
        central_node[..., 2] = -center_height

        node = node.reshape((-1, 3))

        idx = np.zeros((sum(nthetas) + 1, sum(nphis) + 1), np.int_)
        idx[:, 0:-1] = np.arange(NN).reshape((sum(nthetas) + 1, sum(nphis)))
        idx[:, -1] = idx[:, 0]
        NC = sum(nthetas) * sum(nphis)
        cell = np.zeros((2 * NC, 3), dtype=np.int_)

        cell[0::2, 0] = idx[1:, 0:-1].flatten(order='F')
        cell[0::2, 1] = idx[1:, 1:].flatten(order='F')
        cell[0::2, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[1::2, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[1::2, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[1::2, 2] = idx[1:, 1:].flatten(order='F')

        # 处理中心区域，构造单元
        central_idx = np.zeros((nphis[1] + nphis[2] + 1, nphis[0] + 1), dtype=np.int_)
        central_idx[0, ...] = idx[sum(nthetas), :nphis[0] + 1]
        central_idx[-1, ...] = idx[sum(nthetas), sum(nphis[0:3]):sum(nphis[0:4]) + 1][::-1, ...]
        central_idx[:, -1] = idx[sum(nthetas), sum(nphis[0:1]):sum(nphis[0:3]) + 1]
        central_idx[1:, 0] = idx[sum(nthetas), sum(nphis[0:4]):sum(nphis[0:6])][::-1, ...]
        central_idx[1:-1, 1:-1] = np.arange(NN, NN + (nphis[0] - 1) * (nphis[1] + nphis[2] - 1)).reshape(
            (nphis[1] + nphis[2] - 1, nphis[0] - 1))
        central_cell = np.zeros((2 * (nphis[1] + nphis[2]) * nphis[0], 3), dtype=np.int_)
        central_cell[0::2, 0] = central_idx[1:, 0:-1].flatten(order='F')
        central_cell[0::2, 1] = central_idx[1:, 1:].flatten(order='F')
        central_cell[0::2, 2] = central_idx[0:-1, 0:-1].flatten(order='F')
        central_cell[1::2, 0] = central_idx[0:-1, 1:].flatten(order='F')
        central_cell[1::2, 1] = central_idx[0:-1, 0:-1].flatten(order='F')
        central_cell[1::2, 2] = central_idx[1:, 1:].flatten(order='F')

        # 组装底面与侧面
        node = np.concatenate((node, central_node[1:-1, 1:-1].reshape(-1, 3)), axis=0)
        cell = np.concatenate((cell, central_cell), axis=0, dtype=np.int_)

        mesh = cls(node, cell)
        # 标记单元
        domain = np.zeros(2 * sum(nthetas) * sum(nphis), dtype=np.int_).reshape((2 * sum(nthetas), sum(nphis)))
        for i in range(len(nphis)):
            for j in range(len(nthetas)):
                domain[2 * sum(nthetas[0:j]):2 * sum(nthetas[0:j + 1]), sum(nphis[0:i]):sum(nphis[0:i + 1])] = ((
                                                                                                                            i + 5) % 6 + 1) * 10 + j + 1
        central_domain = np.zeros(2 * nphis[0] * (nphis[1] + nphis[2]), dtype=np.int_)
        domain = np.concatenate((domain.flatten(order='F'), central_domain))
        mesh.celldata['domain'] = domain

        if return_edge:
            return mesh, edge_node
        else:
            return mesh
    
    @classmethod
    def from_half_sphere_surface_with_cutting(cls, theta=np.pi/3, h=0.3):
        import gmsh
        theta = np.pi/2-theta
        gmsh.initialize()
        gmsh.model.occ.addSphere(0,0,0,1.0,tag=1,angle1=-theta,angle2=theta,angle3=np.pi)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
        gmsh.model.mesh.generate(2)

        ntags, vxyz, _ = gmsh.model.mesh.getNodes()
        node = vxyz.reshape((-1,3))
        vmap = dict({j:i for i,j in enumerate(ntags)})
        tets_tags,evtags = gmsh.model.mesh.getElementsByType(2)
        evid = np.array([vmap[j] for j in evtags])
        cell = evid.reshape((tets_tags.shape[-1],-1))
        gmsh.finalize()

        W1 = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        W2 = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        node = node@W1@W2

        mesh1 = cls(node,cell)
        bc = mesh1.entity_barycenter("cell")

        remove_flag = np.zeros(mesh1.number_of_cells(),dtype=np.bool_)
        remove_flag[bc[:,2]<1e-6]=True
        remove_flag[bc[:,1]>(np.sin(theta)-1e-6)]=True
        remove_flag[bc[:,1]<(-np.sin(theta)+1e-6)]=True
        cell = cell[~remove_flag]
        return cls(node,cell)

    def streamline_callculator(self, vector_field, start_cell, start_point):
        """
        @brief 给定网格上的向量场，根据初始点计算流线.

        @param vector_field: 网格上的向量场
        @param start_cell: 初始点所在的单元索引
        @param start_point: 初始点

        @return: 流线上的一系列点坐标
        """
        cells = self.entity('cell')
        nodes = self.entity('node')
        cell2edge = self.ds.cell_to_edge()
        edge2cell = self.ds.edge_to_cell()
        bd_cell_index = self.ds.boundary_cell_index()

        point_list = []

        for i in range(len(start_point)):
            start_point_weight = np.linalg.solve(
                np.concatenate((nodes[cells[start_cell[i]]].T, np.ones((1, 3))), axis=0),
                np.concatenate((start_point[i], np.ones(1))))
            # 计算流线
            current_cell = start_cell[i]
            current_point = start_point_weight

            point_list.append([])
            point_list[i].append(current_point @ nodes[cells[current_cell]].tolist())

            while True:
                A = np.concatenate((nodes[cells[current_cell]].T, np.ones((1, 3))), axis=0)
                W = np.concatenate((vector_field[cells[current_cell]].T, np.zeros((1, 3))), axis=0)
                B = np.linalg.inv(A) @ W

                # 计算下一个点的参数
                C = np.zeros((3, 3))
                C[:, 0] = np.einsum('ik, kj, j -> i', B, B, current_point) / 2
                C[:, 1] = np.einsum('ij, j -> i', B, current_point)
                C[:, 2] = current_point

                # 获取参数 h 的最小实根
                temp_hc = np.apply_along_axis(np.roots, axis=1, arr=C).flatten()
                real = np.real(temp_hc)
                img = np.imag(temp_hc)
                hc = np.min(real[(real > 1e-13) & (np.abs(img) < 1e-13)])

                current_point = (np.eye(3) + hc * B + hc ** 2 / 2 * B @ B) @ current_point

                local_edge_index = np.where(current_point < 1e-13)[0][0]
                adj_info = edge2cell[cell2edge[current_cell][local_edge_index]]

                # 确定下一个单元编号，获取所在边局部索引
                if adj_info[0] != current_cell:
                    current_cell, out_edge_index = adj_info[0], adj_info[2]
                else:
                    current_cell, out_edge_index = adj_info[1], adj_info[3]

                # 相邻单元间重心坐标转换
                in_node_index1 = (local_edge_index + 1) % 3
                in_node_index2 = (local_edge_index + 2) % 3

                out_node_index1 = (out_edge_index + 1) % 3
                out_node_index2 = (out_edge_index + 2) % 3

                temp_weight = current_point[in_node_index1]
                current_point[out_node_index1] = current_point[in_node_index2]
                current_point[out_node_index2] = temp_weight
                current_point[out_edge_index] = 0

                point_list[i].append(current_point @ nodes[cells[current_cell]].tolist())

                if current_cell in bd_cell_index:
                    break

        return point_list


TriangleMesh.set_ploter('2d')


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
        self.ds = TriangleMeshDataStructure(NN + 1, newCell)

        if bc:
            self.center = np.append(mesh.entity_barycenter('cell'),
                                    0.5 * (node[edge[bdEdgeIdx, 0], :] + node[edge[bdEdgeIdx, 1], :]), axis=0)
        else:
            self.center = np.append(mesh.circumcenter(),
                                    0.5 * (node[edge[bdEdgeIdx, 0], :] + node[edge[bdEdgeIdx, 1], :]), axis=0)

        self.meshtype = 'tri'

    def is_infinity_cell(self):
        N = self.ds.number_of_nodes()
        cell = self.ds.cell
        return cell[:, 0] == N - 1

    def is_boundary_edge(self):
        NE = self.ds.number_of_edges()
        cell2edge = self.ds.cell_to_edge()
        isInfCell = self.is_infinity_cell()
        isBdEdge = np.zeros(NE, dtype=np.bool_)
        isBdEdge[cell2edge[isInfCell, 0]] = True
        return isBdEdge

    def is_boundary_node(self):
        N = self.ds.number_of_nodes()
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

        node2cell = self.ds.node_to_cell(return_local=True)
        NV = np.asarray((node2cell > 0).sum(axis=1)).reshape(-1)
        NV[isBdNode] += 1
        NV = NV[:-1]

        PNC = len(NV)
        pcell = np.zeros(NV.sum(), dtype=self.itype)
        pcellLocation = np.zeros(PNC + 1, dtype=self.itype)
        pcellLocation[1:] = np.cumsum(NV)

        isBdEdge = self.is_boundary_edge()
        NC = self.ds.number_of_cells() - isBdEdge.sum()
        cell = self.ds.cell
        currentCellIdx = np.zeros(PNC, dtype=self.itype)
        currentCellIdx[cell[:NC, 0]] = range(NC)
        currentCellIdx[cell[:NC, 1]] = range(NC)
        currentCellIdx[cell[:NC, 2]] = range(NC)
        pcell[pcellLocation[:-1]] = currentCellIdx

        currentIdx = pcellLocation[:-1]
        N = self.ds.number_of_nodes() - 1
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
