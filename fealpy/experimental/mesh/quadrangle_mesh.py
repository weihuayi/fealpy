from typing import Union, Optional, Sequence, Tuple, Any, Callable

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .. import logger
from .utils import estr2dim

from .mesh_base import TensorMesh
from .plot import Plotable


class QuadrangleMesh(TensorMesh, Plotable):
    def __init__(self, node, cell):
        """
        """
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)
        kwargs = bm.context(cell)
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
    """
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
    """

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

    def first_fundamental_form(self, J:TensorLike) -> TensorLike:
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

    def interpolation_points(self, p:int, index: Index = _S):
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

    def cell_to_ipoint(self, p:int, index: Index = _S):
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

        cell2ipoint = bm.zeros((NC, (p + 1) * (p + 1)), dtype=self.itype)
        c2p = cell2ipoint.reshape((NC, p + 1, p + 1))
        e2p = self.edge_to_ipoint(p)
        if bm.backend_name in ["numpy", "pytorch"]:
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
            flag = edge2cell[:, 2] == 0
            # c2p[edge2cell[flag, 0], :, 0] = e2p[flag]
            c2p.at[edge2cell[flag, 0], :, 0].set(e2p[flag])
            flag = edge2cell[:, 2] == 1
            # c2p[edge2cell[flag, 0], -1, :] = e2p[flag]
            c2p.at[edge2cell[flag, 0], -1, :].set(e2p[flag])
            flag = edge2cell[:, 2] == 2
            # c2p[edge2cell[flag, 0], :, -1] = e2p[flag, -1::-1]
            c2p.at[edge2cell[flag, 0], :, -1].set(e2p[flag, -1::-1])
            flag = edge2cell[:, 2] == 3
            # c2p[edge2cell[flag, 0], 0, :] = e2p[flag, -1::-1]
            c2p.at[edge2cell[flag, 0], 0, :].set(e2p[flag, -1::-1])

            iflag = edge2cell[:, 0] != edge2cell[:, 1]
            flag = iflag & (edge2cell[:, 3] == 0)
            # c2p[edge2cell[flag, 1], :, 0] = e2p[flag, -1::-1]
            c2p.at[edge2cell[flag, 1], :, 0].set(e2p[flag, -1::-1])
            flag = iflag & (edge2cell[:, 3] == 1)
            # c2p[edge2cell[flag, 1], -1, :] = e2p[flag, -1::-1]
            c2p.at[edge2cell[flag, 1], -1, :].set(e2p[flag, -1::-1])
            flag = iflag & (edge2cell[:, 3] == 2)
            # c2p[edge2cell[flag, 1], :, -1] = e2p[flag]
            c2p.at[edge2cell[flag, 1], :, -1].set(e2p[flag])
            flag = iflag & (edge2cell[:, 3] == 3)
            # c2p[edge2cell[flag, 1], 0, :] = e2p[flag]
            c2p.at[edge2cell[flag, 1], 0, :].set(e2p[flag])

            c2p[:, 1:-1, 1:-1] = NN + NE * (p - 1) + bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p - 1, p - 1)
        else:
            raise ValueError("Unsupported backend")
        return cell2ipoint[index]

    def prolongation_matrix(self, p0: int, p1: int):
        """
        @brief 生成从 p0 元到 p1 元的延拓矩阵，假定 0 < p0 < p1
        """
        raise NotImplementedError

    def jacobi_at_corner(self) -> TensorLike:
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

    def angle(self) -> TensorLike:
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

    def cell_quality(self)  -> TensorLike:
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

    def uniform_refine(self, n:int=1) -> 'QuadrangleMesh':
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

            cell = bm.zeros((4 * NC, 4), dtype=bm.int64)
            if bm.backend_name in ["numpy", "pytorch"]:
                cell[0::4, :] = bm.concatenate([cp[0], ep[0], cc, ep[3]], axis=1)
                cell[1::4, :] = bm.concatenate([ep[0], cp[1], ep[1], cc], axis=1)
                cell[2::4, :] = bm.concatenate([cc, ep[1], cp[2], ep[2]], axis=1)
                cell[3::4, :] = bm.concatenate([ep[3], cc, ep[2], cp[3]], axis=1)
            elif bm.backend_name == "jax":
                row_indices = bm.arange(4 * NC).reshape(NC, 4)
                # 将单元块的行索引和列索引拼接在一起
                cell = cell.at[row_indices[:, 0]].set(bm.concatenate([cp[0], ep[0], cc, ep[3]], axis=-1))
                cell = cell.at[row_indices[:, 1]].set(bm.concatenate([ep[0], cp[1], ep[1], cc], axis=-1))
                cell = cell.at[row_indices[:, 2]].set(bm.concatenate([cc, ep[1], cp[2], ep[2]], axis=-1))
                cell = cell.at[row_indices[:, 3]].set(bm.concatenate([ep[3], cc, ep[2], cp[3]], axis=-1))

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
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10, threshold:Optional[Callable]=None) -> 'QuadrangleMesh':
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
        cell = bm.concatenate((idx[0:-1, 0:-1].reshape(-1, 1),
                               idx[1:, 0:-1].reshape(-1, 1),
                               idx[1:, 1:].reshape(-1, 1),
                               idx[0:-1, 1:].reshape(-1, 1),), axis=1)

        if threshold is not None:
            if bm.backend_name in ["numpy", "pytorch"]:
                bc = bm.sum(node[cell, :], axis=1) / cell.shape[1]
                isDelCell = threshold(bc)
                cell = cell[~isDelCell]
                isValidNode = bm.zeros(NN, dtype=bm.bool)
                isValidNode[cell] = True
                node = node[isValidNode]
                idxMap = bm.zeros(NN, dtype=cell.dtype)
                idxMap[isValidNode] = range(isValidNode.sum())
                cell = idxMap[cell]
            elif bm.backend_name == "jax":
                bc = bm.sum(node[cell, :], axis=1) / cell.shape[1]
                isDelCell = threshold(bc)
                cell = cell[~isDelCell]
                isValidNode = bm.zeros(NN, dtype=bm.bool)
                isValidNode = isValidNode.at[cell].set(True)
                node = node[isValidNode]
                idxMap = bm.zeros(NN, dtype=cell.dtype)
                idxMap.at[isValidNode].set(bm.arange(isValidNode.sum()))
                cell = idxMap[cell]
            else:
                raise ValueError("Unsupported backend")

        return cls(node, cell)

    @classmethod
    def from_unit_square(cls, nx=10, ny=10, threshold:Optional[Callable]=None) -> 'QuadrangleMesh':
        """
        Generate a quadrilateral mesh for a unit square.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return QuadrangleMesh instance
        """
        return cls.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny, threshold=threshold)

    @classmethod
    def from_polygon_gmsh(cls, vertices: list[tuple], h: float) -> 'QuadrangleMesh':
        """
        Generate a quadrilateral mesh for a polygonal region by gmsh.

        @param vertices List of tuples representing vertices of the polygon
        @param h Parameter controlling mesh density
        @return QuadrilateralMesh instance
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

        # 设置网格算法选项，使用 Quadrangle 2D 算法
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # 生成网格
        gmsh.model.mesh.generate(2)

        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = bm.tensor(node_coords, dtype=bm.float64).reshape(-1, 3)[:, 0:2].copy()

        # 获取四边形单元信息
        quadrilateral_type = 3  # 四边形单元的类型编号为 3
        quad_tags, quad_connectivity = gmsh.model.mesh.getElementsByType(quadrilateral_type)
        cell = bm.tensor(quad_connectivity, dtype=bm.int64).reshape(-1, 4) - 1

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of quadrilaterals: {cell.shape[0]}")

        gmsh.finalize()

        NN = len(node)
        if bm.backend_name in ["numpy", "pytorch"]:
            isValidNode = bm.zeros(NN, dtype=bm.bool)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
        elif bm.backend_name == "jax":
            isValidNode = bm.zeros(NN, dtype=bm.bool)
            isValidNode.at[cell].set(True)
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype)
            idxMap.at[isValidNode].set(bm.arange(isValidNode.sum()))
        else:
            raise ValueError("Unsupported backend")
        cell = idxMap[cell]

        return cls(node, cell)

    @classmethod
    def from_fuel_rod_gmsh(cls,R1,R2,L,w,h,meshtype='normal'):
        raise NotImplementedError

    @classmethod
    def from_one_quadrangle(cls, meshtype='square') -> 'QuadrangleMesh':
        """
        Generate a quadrilateral mesh for a single quadrangle.

        @param meshtype Type of quadrangle mesh, options are 'square', 'zhengfangxing', 'rectangle', 'rec', 'juxing', 'rhombus', 'lingxing' (default: 'square')
        @return QuadrangleMesh instance
        """
        if meshtype in {'square'}:
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0]], dtype=bm.float64)
        elif meshtype in {'rectangle'}:
            node = bm.tensor([
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0]], dtype=bm.float64)
        elif meshtype in {'rhombus'}:
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.5, bm.sqrt(3) / 2],
                [0.5, bm.sqrt(3) / 2]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int64)
        return cls(node, cell)

    @classmethod
    def from_triangle_mesh(cls, mesh) -> 'QuadrangleMesh':
        """
        把每个三角形分成三个四边形
        @param mesh: 三角形网格
        @return:
        """
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        node0 = mesh.node
        cell0 = mesh.cell
        ec = mesh.entity_barycenter('edge')
        cc = mesh.entity_barycenter('cell')
        cell2edge = mesh.cell2edge

        node = bm.concatenate([node0, ec, cc], axis=0)
        idx = bm.arange(NC)

        cell1 = bm.concatenate([(NN+NE+idx).reshape(-1, 1),
                                (cell2edge[:, 0] + NN).reshape(-1, 1),
                                (cell0[:, 2]).reshape(-1, 1),
                                (cell2edge[:, 1] + NN).reshape(-1, 1)], axis=1)
        cell2 = bm.concatenate([(cell1[:, 0]).reshape(-1, 1),
                                (cell2edge[:, 1] + NN).reshape(-1, 1),
                                (cell0[:, 0]).reshape(-1, 1),
                                (cell2edge[:, 2] + NN).reshape(-1, 1)], axis=1)
        cell3 = bm.concatenate([(cell1[:, 0]).reshape(-1, 1),
                                (cell2edge[:, 2] + NN).reshape(-1, 1),
                                (cell0[:, 1]).reshape(-1, 1),
                                (cell2edge[:, 0] + NN).reshape(-1, 1)], axis=1)
        cell = bm.concatenate([cell1, cell2, cell3], axis=0)

        return cls(node, cell)



    @classmethod
    def polygon_domain_generator(cls, num_vertices=20, radius=1.0, center=[0.0, 0.0]):
        raise NotImplementedError

    @classmethod
    def rand_quad_mesh_generator(cls, num, filename=None, h=0.382, radius=0.5, center=[0.5, 0.5]):
        """
        @brief 随机生成指定区域的，指定数量的，随机四边形网格

        @param num: 需要生成的网格数量
        @param filename: 输出文件名，如果非 None，输出文件，如果为 None，返回网格列表
        @param h: 网格密度
        @param radius: 区域半径
        @param center: 区域中点坐标

        @return: None 或网格列表
        """
        raise NotImplementedError


QuadrangleMesh.set_ploter('2d')
