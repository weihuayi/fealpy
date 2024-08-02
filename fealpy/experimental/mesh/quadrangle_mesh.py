from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .. import logger
from .utils import estr2dim

from .mesh_base import TensorMesh


class QuadrangleMesh(TensorMesh):
    def __init__(self, node, cell):
        """
        """
        super().__init__(TD=2)
        kwargs = {'dtype': cell.dtype}
        self.node = node
        self.cell = cell
        self.localEdge = bm.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.localFace = bm.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.ccw = bm.tensor([0, 1, 2, 3], **kwargs)

        self.localCell = None

        self.construct()

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}

        self.edge_bc_to_point = self.bc_to_point
        self.face_bc_to_point = self.bc_to_point
        self.cell_bc_to_point = self.bc_to_point

        self.cell_grad_shape_function = self.grad_shape_function
        self.cell_shape_function = self.shape_function

        self.face_normal = self.edge_normal
        self.face_unit_normal = self.edge_unit_normal

    def ref_cell_measure(self):
        return 1.0

    def ref_face_measure(self):
        return 1.0

    def cell_area(self, index: Index = _S) -> TensorLike:
        """
        @brief 根据散度定理计算多边形的面积
        @note 请注意下面的计算方式不方便实现部分单元面积的计算
        """
        GD = self.GD
        if GD == 2:
            NC = self.number_of_cells()
            node = self.entity('node')
            edge = self.entity('edge')
            edge2cell = self.edge2cell

            t = self.edge_tangent()
            val = t[:, 1] * node[edge[:, 0], 0] - t[:, 0] * node[edge[:, 0], 1]

            a = bm.zeros(NC, dtype=self.ftype)
            bm.add.at(a, edge2cell[:, 0], val)

            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            bm.add.at(a, edge2cell[isInEdge, 1], -val[isInEdge])

            a /= 2.0

            return a[index]
        elif GD == 3:
            node = self.entity('node')
            cell = self.entity('cell')[index]

            v0 = node[cell[:, 1]] - node[cell[:, 0]]
            v1 = node[cell[:, 2]] - node[cell[:, 0]]
            v2 = node[cell[:, 3]] - node[cell[:, 0]]

            s1 = 0.5 * bm.linalg.norm(bm.cross(v0, v1), axis=-1)
            s2 = 0.5 * bm.linalg.norm(bm.cross(v1, v2), axis=-1)
            s = s1 + s2
            return s

    def entity_measure(self, etype: Union[int, str] = 'cell', index: Index = _S) -> TensorLike:
        node = self.node

        if isinstance(etype, str):
            etype = estr2dim(self, etype)

        if etype == 0:
            return bm.tensor([0, ], dtype=self.ftype)
        elif etype == 1:
            edge = self.entity(1, index)
            return bm.edge_length(edge, node)
        elif etype == 2:
            return self.cell_area(index=index)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

    def quadrature_formula(self, q, etype: Union[int, str] = 'cell'):
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        qf = GaussLegendreQuadrature(q)
        if etype == 2:
            return TensorProductQuadrature((qf, qf))
        elif etype == 1:
            return qf
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def grad_shape_function(self, bcs: Tuple[TensorLike], p: int = 1, *, index: Index = _S,
                            variables: str = 'u', mi: Optional[TensorLike] = None) -> TensorLike:
        assert isinstance(bcs, tuple)
        TD = len(bcs)
        Dlambda = bm.array([-1, 1], dtype=self.ftype)
        phi = bm.simplex_shape_function(bcs[0], p=p)
        R = bm.simplex_grad_shape_function(bcs[0], p=p)
        dphi = bm.einsum('...ij, j->...i', R, Dlambda)  # (..., ldof)

        n = phi.shape[0] ** TD
        ldof = phi.shape[-1] ** TD
        shape = (n, ldof, TD)
        gphi = bm.zeros(shape, dtype=self.ftype)

        gphi0 = bm.einsum('im, jn->ijmn', dphi, phi).reshape(-1, ldof, 1)
        gphi1 = bm.einsum('im, jn->ijmn', phi, dphi).reshape(-1, ldof, 1)
        gphi = bm.concatenate((gphi0, gphi1), axis=-1)
        if variables == 'x':
            J = self.jacobi_matrix(bcs, index=index)
            G = self.first_fundamental_form(J)
            G = bm.linalg.inv(G)
            gphi = bm.einsum('qikm, qimn, qln->qilk', J, G, gphi)
        return gphi

    def jacobi_matrix(self, bc, index: Index = _S) -> TensorLike:
        """
        @brief 计算参考单元 (xi, eta) 到实际 Lagrange 四边形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        gphi = self.grad_shape_function(bc, p=1, variables='u', index=index)
        J = bm.einsum('cim, ...in->...cmn', node[cell[:, [0, 3, 1, 2]]], gphi)
        return J

    def first_fundamental_form(self, J) -> TensorLike:
        """
        @brief 由 Jacobi 矩阵计算第一基本形式。
        """
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        data = [[0 for i in range(TD)] for j in range(TD)]
        for i in range(TD):
            data[i][i] = bm.einsum('...d, ...d->...', J[..., i], J[..., i])
            for j in range(i + 1, TD):
                data[i][j] = bm.einsum('...d, ...d->...', J[..., i], J[..., j])
                data[j][i] = data[i][j]
        data = [val.reshape(val.shape + (1,)) for data_ in data for val in data_]
        G = bm.concatenate(data, axis=-1).reshape(shape)
        return G

    def edge_unit_tangent(self, index: Index = _S) -> TensorLike:
        return self.edge_tangent(index=index, unit=True)

    def edge_unit_normal(self, index: Index = _S) -> TensorLike:
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.GD == 2
        v = self.edge_unit_tangent(index=index)
        w = bm.tensor([(0, -1), (1, 0)])
        return v @ w

    def edge_frame(self, index: Index = _S):
        """
        @brief 计算二维网格中每条边上的局部标架
        """
        assert self.GD == 2
        # TODO: index 出错
        t = self.edge_unit_tangent(index=index)
        w = bm.tensor([(0, -1), (1, 0)])
        n = t @ w
        return n, t

    def interpolation_points(self, p, index: Index = _S):
        """
        @brief 获取四边形网格上所有 p 次插值点
        """
        cell = self.entity('cell')
        node = self.entity('node')
        if p == 1:
            return node

        NN = self.number_of_nodes()
        GD = self.geo_dimension()

        gdof = self.number_of_global_ipoints(p)

        NE = self.number_of_edges()

        edge = self.entity('edge')

        multiIndex = self.multi_index_matrix(p, 1)
        w = multiIndex[1:-1, :] / p
        ipoints0 = bm.einsum('ij, ...jm->...im', w, node[edge, :]).reshape(-1, GD)

        w = bm.einsum('im, jn->ijmn', w, w).reshape(-1, 4)
        ipoints1 = bm.einsum('ij, kj...->ki...', w, node[cell[:, [0, 3, 1, 2]]]).reshape(-1, GD)

        ipoints = bm.concatenate((node, ipoints0, ipoints1), axis=0)
        return ipoints

    def number_of_corner_nodes(self):
        return self.number_of_nodes()

    def cell_to_ipoint(self, p, index: Index = _S):
        """
        @brief 获取单元上的双 p 次插值点
        """

        cell = self.entity('cell')

        if p == 0:
            return bm.arange(len(cell)).reshape((-1, 1))[index]

        if p == 1:
            return cell[index, [0, 3, 1, 2]]  # 先排 y 方向，再排 x 方向

        edge2cell = self.edge2cell
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        if bm.backend_name in ["numpy", "pytorch"]:
            cell2ipoint = bm.zeros((NC, (p + 1) * (p + 1)), dtype=self.itype)
            c2p = cell2ipoint.reshape((NC, p + 1, p + 1))

            e2p = self.edge_to_ipoint(p)
            flag = edge2cell[:, 2] == 0
            c2p[edge2cell[flag, 0], :, 0] = e2p[flag]
            flag = edge2cell[:, 2] == 1
            c2p[edge2cell[flag, 0], -1, :] = e2p[flag]
            flag = edge2cell[:, 2] == 2
            c2p[edge2cell[flag, 0], :, -1] = e2p[flag, -1::-1]
            flag = edge2cell[:, 2] == 3
            c2p[edge2cell[flag, 0], 0, :] = e2p[flag, -1::-1]

            iflag = edge2cell[:, 0] != edge2cell[:, 1]
            flag = iflag & (edge2cell[:, 3] == 0)
            c2p[edge2cell[flag, 1], :, 0] = e2p[flag, -1::-1]
            flag = iflag & (edge2cell[:, 3] == 1)
            c2p[edge2cell[flag, 1], -1, :] = e2p[flag, -1::-1]
            flag = iflag & (edge2cell[:, 3] == 2)
            c2p[edge2cell[flag, 1], :, -1] = e2p[flag]
            flag = iflag & (edge2cell[:, 3] == 3)
            c2p[edge2cell[flag, 1], 0, :] = e2p[flag]

            c2p[:, 1:-1, 1:-1] = NN + NE * (p - 1) + bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p - 1, p - 1)
        elif bm.backend_name == "jax":
            raise NotImplementedError
        else:
            raise ValueError("Unsupported backend")
        return cell2ipoint[index]

    def prolongation_matrix(self, p0: int, p1: int):
        """
        @brief 生成从 p0 元到 p1 元的延拓矩阵，假定 0 < p0 < p1
        """
        raise NotImplementedError

    def jacobi_at_corner(self):
        NC = self.number_of_cells()
        node = self.entity('node')
        cell = self.entity('cell')
        localEdge = self.localEdge
        iprev = [3, 0, 1, 2]
        jacobis = []
        for i, j in localEdge:
            k = iprev[i]
            v0 = node[cell[:, j], :] - node[cell[:, i], :]
            v1 = node[cell[:, k], :] - node[cell[:, i], :]
            jacobis.append((v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]).reshape(-1, 1))
        jacobi = bm.concatenate(jacobis, axis=-1)
        return jacobi

    def angle(self):
        NC = self.number_of_cells()
        node = self.node
        cell = self.cell
        localEdge = self.localEdge
        iprev = [3, 0, 1, 2]
        angles = []
        for i, j in localEdge:
            k = iprev[i]
            v0 = node[cell[:, j], :] - node[cell[:, i], :]
            v1 = node[cell[:, k], :] - node[cell[:, i], :]
            angles.append(bm.arccos(
                bm.sum(v0 * v1, axis=1)
                / bm.sqrt(bm.sum(v0 ** 2, axis=1)
                          * bm.sum(v1 ** 2, axis=1))).reshape(-1, 1))
        angle = bm.concatenate(angles, axis=-1)
        return angle

    def cell_quality(self):
        jacobi = self.jacobi_at_corner()
        return jacobi.sum(axis=1)/4

    @classmethod
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10, threshold=None):
        """
        Generate a quadrilateral mesh for a rectangular domain.

        :param box: list of four float values representing the x- and y-coordinates of the lower left and upper right corners of the domain (default: [0, 1, 0, 1])
        :param nx: number of cells along the x-axis (default: 10)
        :param ny: number of cells along the y-axis (default: 10)
        :param threshold: optional function to filter cells based on their barycenter coordinates (default: None)
        :return: QuadrangleMesh instance
        """
        NN = (nx + 1) * (ny + 1)
        NC = nx * ny
        node = bm.zeros((NN, 2))
        x = bm.linspace(box[0], box[1], nx + 1)
        y = bm.linspace(box[2], box[3], ny + 1)
        X, Y = bm.meshgrid(x, y, indexing='ij')
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

        idx = bm.arange(NN).reshape(nx + 1, ny + 1)
        cell = bm.concatenate((idx[0:-1, 0:-1].T.reshape(-1, 1),
                               idx[1:, 0:-1].T.reshape(-1, 1),
                               idx[1:, 1:].T.reshape(-1, 1),
                               idx[0:-1, 1:].T.reshape(-1, 1),), axis=1)

        if threshold is not None:
            bc = bm.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = bm.zeros(NN, dtype=bm.bool_)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]

        return cls(node, cell)
