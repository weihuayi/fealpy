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
        return self.edge_normal(index=index, unit=True)

    def edge_frame(self, index: Index = _S):
        """
        @brief 计算二维网格中每条边上的局部标架
        """
        assert self.GD == 2
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

    def reorder_cell(self, idx):
        raise NotImplementedError
        # NC = self.number_of_cells()
        # NN = self.number_of_nodes()
        # cell = self.cell
        # # localCell 似乎之前未初始化
        # cell = cell[bm.arange(NC).reshape(-1, 1), self.localCell[idx]]
        # self.ds.reinit(NN, cell)

    def uniform_refine(self, n=1):
        """
        @brief 一致加密四边形网格
        """
        for i in range(n):
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            # Find the cutted edge
            cell2edge = self.cell2edge
            edgeCenter = self.entity_barycenter('edge')
            cellCenter = self.entity_barycenter('cell')

            edge2center = bm.arange(NN, NN + NE)

            cell = self.cell
            cp = [cell[:, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[:, i]].reshape(-1, 1) for i in range(4)]
            cc = bm.arange(NN + NE, NN + NE + NC).reshape(-1, 1)

            if bm.backend_name in ["numpy", "pytorch"]:
                cell = bm.zeros((4 * NC, 4), dtype=bm.int_)
                cell[0::4, :] = bm.concatenate([cp[0], ep[0], cc, ep[3]], axis=1)
                cell[1::4, :] = bm.concatenate([ep[0], cp[1], ep[1], cc], axis=1)
                cell[2::4, :] = bm.concatenate([cc, ep[1], cp[2], ep[2]], axis=1)
                cell[3::4, :] = bm.concatenate([ep[3], cc, ep[2], cp[3]], axis=1)
            elif bm.backend_name == "jax":
                # TODO: 考虑拼接次数太多导致的效率问题
                cell_blocks = []
                for i in range(NC):
                    block = bm.concatenate([
                        bm.array([cp[0][i], ep[0][i], cc[i], ep[3][i]]).reshape(1, -1),
                        bm.array([ep[0][i], cp[1][i], ep[1][i], cc[i]]).reshape(1, -1),
                        bm.array([cc[i], ep[1][i], cp[2][i], ep[2][i]]).reshape(1, -1),
                        bm.array([ep[3][i], cc[i], ep[2][i], cp[3][i]]).reshape(1, -1)
                    ], axis=0)
                    cell_blocks.append(block)

                # 将所有单元块沿行方向拼接
                cell = bm.concatenate(cell_blocks, axis=0)
            else:
                raise ValueError("Unsupported backend")

            self.node = bm.concatenate([self.node, edgeCenter, cellCenter], axis=0)
            self.cell = cell
            self.construct()

    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_Quad = 9
            return VTK_Quad
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, etype='cell', index: Index = _S, fname=None):
        """
        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        raise NotImplementedError

        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.GD
        if GD == 2:
            node = bm.concatenate((node, bm.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        NV = cell.shape[-1]

        cell = bm.concatenate([NV, cell], axis=1)

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)

    def show_function(self, plot, uh, cmap=None):
        """
        TODO: no test
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

        node = self.node
        cax = axes.plot_trisurf(
                node[:, 0], node[:, 1],
                uh, cmap=cmap, lw=0.0)
        axes.figure.colorbar(cax, ax=axes)
        return axes

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
            if bm.backend_name in ["numpy", "pytorch"]:
                bc = bm.sum(node[cell, :], axis=1) / cell.shape[1]
                isDelCell = threshold(bc)
                cell = cell[~isDelCell]
                isValidNode = bm.zeros(NN, dtype=bm.bool_)
                isValidNode[cell] = True
                node = node[isValidNode]
                idxMap = bm.zeros(NN, dtype=cell.dtype)
                idxMap[isValidNode] = range(isValidNode.sum())
                cell = idxMap[cell]
            elif bm.backend_name == "jax":
                bc = bm.sum(node[cell, :], axis=1) / cell.shape[1]
                isDelCell = threshold(bc)
                cell = cell[~isDelCell]
                isValidNode = bm.zeros(NN, dtype=bm.bool_)
                isValidNode = isValidNode.at[cell].set(True)
                node = node[isValidNode]
                idxMap = bm.zeros(NN, dtype=cell.dtype)
                idxMap.at[isValidNode].set(bm.arange(isValidNode.sum()))
                cell = idxMap[cell]
            else:
                raise ValueError("Unsupported backend")

        return cls(node, cell)
