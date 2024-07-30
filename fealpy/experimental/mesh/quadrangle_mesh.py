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

    def ref_cell_measure(self):
        return 1.0

    def ref_face_measure(self):
        return 1.0

    def cell_area(self, index: Index=_S) -> TensorLike:
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

            s1 = 0.5*bm.linalg.norm(bm.cross(v0, v1), axis=-1)
            s2 = 0.5*bm.linalg.norm(bm.cross(v1, v2), axis=-1)
            s = s1 + s2
            return s
    def entity_measure(self, etype: Union[int, str]='cell', index: Index=_S) -> TensorLike:
        node = self.node

        if isinstance(etype, str):
            etype = estr2dim(self, etype)

        if etype == 0:
            return bm.tensor([0,], dtype=self.ftype)
        elif etype == 1:
            edge = self.entity(1, index)
            return bm.edge_length(edge, node)
        elif etype == 2:
            return self.cell_area(index=index)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

    def quadrature_formula(self, q, etype:Union[int, str]='cell'):
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

    def grad_shape_function(self, bcs: Tuple[TensorLike], p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        assert isinstance(bcs, tuple)
        TD = len(bcs)
        Dlambda = bm.array([-1, 1], dtype=self.ftype)
        phi = bm.simplex_shape_function(bcs[0], p=p)
        R = bm.simplex_grad_shape_function(bcs[0], p=p)
        dphi = bm.einsum('...ij, j->...i', R, Dlambda) # (..., ldof)

        n = phi.shape[0]**TD
        ldof = phi.shape[-1]**TD
        shape = (n, ldof, TD)
        gphi = bm.zeros(shape, dtype=self.ftype)

        gphi0 = bm.einsum('im, jn->ijmn', dphi, phi).reshape(-1, ldof, 1)
        gphi1 = bm.einsum('im, jn->ijmn', phi, dphi).reshape(-1, ldof, 1)
        gphi = bm.concatenate((gphi0, gphi1), axis=-1)
        if variable == 'x':
            J = self.jacobi_matrix(bcs, index=index)
            G = self.first_fundamental_form(J)
            # TODO: numpy 后端未实现 linalg
            G = bm.linalg.inv(G)
            gphi = bm.einsum('qikm, qimn, qln->qilk', J, G, gphi)
        return gphi

    # TODO: test
    def jacobi_matrix(self, bc, index: Index=_S):
        """
        @brief 计算参考单元 (xi, eta) 到实际 Lagrange 四边形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        gphi = self.grad_shape_function(bc, p=1, variable='u', index=index)
        J = bm.einsum( 'cim, ...in->...cmn', node[cell[:, [0, 3, 1, 2]]], gphi)
        return J

    # TODO: test
    def first_fundamental_form(self, J):
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

    def edge_unit_tangent(self, index:Index=_S):
        v = self.edge_tangent(index=index)
        length = bm.sqrt((v**2).sum(axis=1)).reshape(-1, 1)
        return v/length

    def edge_unit_normal(self, index:Index=_S):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.GD == 2
        v = self.edge_unit_tangent(index=index)
        w = bm.tensor([(0,-1),(1,0)])
        return v@w

    def edge_frame(self, index: Index=_S):
        """
        @brief 计算二维网格中每条边上的局部标架
        """
        assert self.GD == 2
        t = self.edge_unit_tangent(index=index)
        w = bm.tensor([(0,-1),(1,0)])
        n = t@w
        return n, t



