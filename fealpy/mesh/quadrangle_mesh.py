import numpy as np
from .mesh_base import Mesh, Plotable
from .mesh_data_structure import Mesh2dDataStructure
from scipy.sparse import coo_matrix


class QuadrangleMeshDataStructure(Mesh2dDataStructure):
    localEdge = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
    localFace = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
    ccw = np.array([0, 1, 2, 3])


## @defgroup MeshGenerators TetrhedronMesh Common Region Mesh Generators
class QuadrangleMesh(Mesh, Plotable):
    """
    @brief 非结构四边形网格数据结构对象
    """

    def __init__(self, node, cell):
        assert cell.shape[-1] == 4
        self.node = node
        NN = node.shape[0]
        self.ds = QuadrangleMeshDataStructure(NN, cell)

        self.meshtype = 'quad'
        self.p = 1 # 最低次的四边形

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.meshdata = {}

        self.face_tangent = self.edge_tangent
        self.face_unit_tangent = self.edge_unit_tangent

        self.edge_shape_function = self._shape_function
        self.grad_edge_shape_function = self._grad_shape_function

        self.face_shape_function = self._shape_function
        self.grad_face_shape_function = self._grad_shape_function

        self.face_to_ipoint = self.edge_to_ipoint

    def ref_cell_measure(self):
        return 1.0

    def ref_face_measure(self):
        return 1.0

    def integrator(self, q, etype='cell'):
        from ..quadrature import GaussLegendreQuadrature
        qf = GaussLegendreQuadrature(q)
        if etype in {'cell', 2}:
            from ..quadrature import TensorProductQuadrature
            return TensorProductQuadrature((qf, qf))
        elif etype in {'edge', 'face', 1}:
            return qf

    def entity_measure(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return 0
        else:
            raise ValueError(f"Invalid entity type '{etype}'.")

    def cell_area(self, index=np.s_[:]):
        """
        @brief 根据散度定理计算多边形的面积
        @note 请注意下面的计算方式不方便实现部分单元面积的计算
        """
        NC = self.number_of_cells()
        node = self.entity('node')
        edge = self.entity('edge')
        edge2cell = self.ds.edge_to_cell()

        t = self.edge_tangent()
        val = t[:, 1]*node[edge[:, 0], 0] - t[:, 0]*node[edge[:, 0], 1]

        a = np.zeros(NC, dtype=self.ftype)
        np.add.at(a, edge2cell[:, 0], val)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        np.add.at(a, edge2cell[isInEdge, 1], -val[isInEdge])

        a /= 2.0

        return a[index]

    def bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把积分点变换到实际网格实体上的笛卡尔坐标点
        """
        node = self.entity('node')
        if isinstance(bc, tuple):
            assert len(bc) == 2
            cell = self.entity('cell')[index]

            bc0 = bc[0].reshape(-1, 2) # (NQ0, 2)
            bc1 = bc[1].reshape(-1, 2) # (NQ1, 2)
            bc = np.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4) # (NQ0, NQ1, 2, 2)

            # node[cell].shape == (NC, 4, 2)
            # bc.shape == (NQ, 4)
            p = np.einsum('...j, cjk->...ck', bc, node[cell[:, [0, 3, 1, 2]]]) # (NQ, NC, 2)

            if p.shape[0] == 1: # 如果只有一个积分点
                p = p.reshape(-1, 2)
        else:
            edge = self.entity('edge')[index]
            p = np.einsum('...j, ejk->...ek', bc, node[edge]) # (NQ, NE, 2)
        return p

    edge_bc_to_point = bc_to_point
    face_bc_to_point = bc_to_point
    cell_bc_to_point = bc_to_point

    def shape_function(self, bc, p=1):
        """
        @brief 四边形单元上的形函数
        """
        assert isinstance(bc, tuple)
        GD = len(bc)
        phi = [self._shape_function(val, p=p) for val in bc]
        ldof = (p+1)**GD
        return np.einsum('im, jn->ijmn', phi[0], phi[1]).reshape(-1, ldof)

    cell_shape_function = shape_function

    def grad_shape_function(self, bc, p=1, variables='x', index=np.s_[:]):
        """
        @brief  四边形单元形函数的导数

        @note 计算单元形函数关于参考单元变量 u=(xi, eta) 或者实际变量 x 梯度。

        bc 是一个长度为 2 的 tuple

        bc[i] 是一个一维积分公式的重心坐标数组

        这里假设 bc[0] == bc[1] == ... = bc[TD-1]
        """
        assert isinstance(bc, tuple) and len(bc) == 2

        Dlambda = np.array([-1, 1], dtype=self.ftype)

        phi0 = self._shape_function(bc[0], p=p)
        R0 = self._grad_shape_function(bc[0], p=p)
        gphi0 = np.einsum('...ij, j->...i', R0, Dlambda) # (..., ldof)

        phi1 = self._shape_function(bc[1], p=p)
        R1 = self._grad_shape_function(bc[1], p=p)
        gphi1 = np.einsum('...ij, j->...i', R1, Dlambda) # (..., ldof)

        n = phi0.shape[0]*phi1.shape[0] # 张量积分点的个数
        ldof = phi0.shape[-1]*phi1.shape[-1]
        shape = (n, ldof, 2)
        gphi = np.zeros(shape, dtype=self.ftype)

        gphi[..., 0] = np.einsum('im, kn->ikmn', gphi0, phi1).reshape(-1, ldof)
        gphi[..., 1] = np.einsum('im, kn->ikmn', phi0, gphi1).reshape(-1, ldof)

        if variables == 'u':
            return gphi
        elif variables == 'x':
            J = self.jacobi_matrix(bc, index=index)
            G = self.first_fundamental_form(J)
            G = np.linalg.inv(G)
            gphi = np.einsum('...ikm, ...imn, ...ln->...ilk', J, G, gphi)
            return gphi

    cell_grad_shape_function = grad_shape_function

    def jacobi_matrix(self, bc, index=np.s_[:]):
        """
        @brief 计算参考单元 (xi, eta) 到实际 Lagrange 四边形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        gphi = self.grad_shape_function(bc, p=1, variables='u', index=index)
        J = np.einsum( 'cim, ...in->...cmn', node[cell[:, [0, 3, 1, 2]]], gphi)
        return J

    def first_fundamental_form(self, J):
        """
        @brief 由 Jacobi 矩阵计算第一基本形式。
        """
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        G = np.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = np.einsum('...d, ...d->...', J[..., i], J[..., i])
            for j in range(i+1, TD):
                G[..., i, j] = np.einsum('...d, ...d->...', J[..., i], J[..., j])
                G[..., j, i] = G[..., i, j]
        return G

    def edge_frame(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上的局部标架
        """
        assert self.geo_dimension() == 2
        t = self.edge_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        n = t@w
        return n, t

    def edge_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def edge_unit_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    face_normal = edge_normal
    face_unit_normal = edge_unit_normal

    def prolongation_matrix(self, p0:int, p1:int):
        """
        @brief 生成从 p0 元到 p1 元的延拓矩阵，假定 0 < p0 < p1
        """

        assert 0 < p0 < p1

        TD = self.top_dimension()

        gdof1 = self.number_of_global_ipoints(p1)
        gdof0 = self.number_of_global_ipoints(p0)

        # 1. 网格节点上的插值点 
        NN = self.number_of_nodes()
        I = range(NN)
        J = range(NN)
        V = np.ones(NN, dtype=self.ftype)
        P = coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 2. 网格边内部的插值点 
        NE = self.number_of_edges()
        # p1 元在边上插值点对应的重心坐标
        bcs = self.multi_index_matrix(p1, 1)/p1 
        # p0 元基函数在 p1 元对应的边内部插值点处的函数值
        phi = self.edge_shape_function(bcs[1:-1], p=p0) # (ldof1 - 2, ldof0)  
       
        e2p1 = self.edge_to_ipoint(p1)[:, 1:-1]
        e2p0 = self.edge_to_ipoint(p0)
        shape = (NE, ) + phi.shape

        I = np.broadcast_to(e2p1[:, :, None], shape=shape).flat
        J = np.broadcast_to(e2p0[:, None, :], shape=shape).flat
        V = np.broadcast_to( phi[None, :, :], shape=shape).flat

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 3. 单元内部的插值点
        NC = self.number_of_cells()
        # p1 元在单元上对应插值点的重心坐标
        bcs = self.multi_index_matrix(p1, 1)/p1
        # p0 元基函数在 p1 元对应的单元内部插值点处的函数值
        phi = self.cell_shape_function((bcs[1:-1], bcs[1:-1]), p=p0) #
        c2p1 = self.cell_to_ipoint(p1).reshape(NC, p1+1, p1+1)[:, 1:-1, 1:-1]
        c2p1 = c2p1.reshape(NC, -1)
        c2p0 = self.cell_to_ipoint(p0)

        shape = (NC, ) + phi.shape

        I = np.broadcast_to(c2p1[:, :, None], shape=shape).flat
        J = np.broadcast_to(c2p0[:, None, :], shape=shape).flat
        V = np.broadcast_to( phi[None, :, :], shape=shape).flat

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        return P.tocsr()


    def number_of_local_ipoints(self, p, iptype='cell'):
        if iptype in {'cell', 2}:
            return (p+1)*(p+1)
        elif iptype in {'face', 'edge',  1}:
            return p + 1
        elif iptype in {'node', 0}:
            return 1

    def number_of_global_ipoints(self, p):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        return NN + (p-1)*NE + (p-1)*(p-1)*NC

    def interpolation_points(self, p, index=np.s_[:]):
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
        ipoints = np.zeros((gdof, GD), dtype=self.ftype)
        ipoints[:NN, :] = node

        NE = self.number_of_edges()

        edge = self.entity('edge')

        multiIndex = self.multi_index_matrix(p, 1)
        w = multiIndex[1:-1, :]/p
        ipoints[NN:NN+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', w,
                node[edge,:]).reshape(-1, GD)

        w = np.einsum('im, jn->ijmn', w, w).reshape(-1, 4)
        ipoints[NN+(p-1)*NE:, :] = np.einsum('ij, kj...->ki...', w,
                node[cell[:, [0, 3, 1, 2]]]).reshape(-1, GD)

        return ipoints

    def cell_to_ipoint(self, p, index=np.s_[:]):
        """
        @brief 获取单元上的双 p 次插值点
        """

        cell = self.entity('cell')

        if p==1:
            return cell[index, [0, 3, 1, 2]] # 先排 y 方向，再排 x 方向

        edge2cell = self.ds.edge_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        cell2ipoint = np.zeros((NC, (p+1)*(p+1)), dtype=self.itype)
        c2p= cell2ipoint.reshape((NC, p+1, p+1))

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

        c2p[:, 1:-1, 1:-1] = NN + NE*(p-1) + np.arange(NC*(p-1)*(p-1)).reshape(NC, p-1, p-1)

        return cell2ipoint[index]

    def uniform_refine(self, n=1):
        """
        @brief 一致加密四边形网格
        """
        for i in range(n):
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            # Find the cutted edge
            cell2edge = self.ds.cell_to_edge()
            edgeCenter = self.entity_barycenter('edge')
            cellCenter = self.entity_barycenter('cell')

            edge2center = np.arange(NN, NN + NE)

            cell = self.ds.cell
            cp = [cell[:, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[:, i]].reshape(-1, 1) for i in range(4)]
            cc = np.arange(NN + NE, NN + NE + NC).reshape(-1, 1)

            cell = np.zeros((4*NC, 4), dtype=np.int_)
            cell[0::4, :] = np.r_['1', cp[0], ep[0], cc, ep[3]]
            cell[1::4, :] = np.r_['1', ep[0], cp[1], ep[1], cc]
            cell[2::4, :] = np.r_['1', cc, ep[1], cp[2], ep[2]]
            cell[3::4, :] = np.r_['1', ep[3], cc, ep[2], cp[3]]

            self.node = np.r_['0', self.node, edgeCenter, cellCenter]
            self.ds.reinit(NN + NE + NC, cell)

    def number_of_corner_nodes(self):
        return self.ds.NN

    def reorder_cell(self, idx):
        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        cell = self.entity('cell')
        cell = cell[np.arange(NC).reshape(-1, 1), self.ds.localCell[idx]]
        self.ds.reinit(NN, cell)

    def angle(self):
        NC = self.number_of_cells()
        node = self.entity('node')
        cell = self.ds.cell
        localEdge = self.ds.local_edge()
        angle = np.zeros((NC, 4), dtype=np.float)
        iprev = [3, 0, 1, 2]
        for i, j in localEdge:
            k = iprev[i]
            v0 = node[cell[:, j], :] - node[cell[:, i], :]
            v1 = node[cell[:, k], :] - node[cell[:, i], :]
            angle[:, i] = np.arccos(
                    np.sum(
                        v0*v1, axis=1
                    )/np.sqrt(
                        np.sum(v0**2, axis=1)*np.sum(v1**2, axis=1)))
        return angle

    def jacobi_at_corner(self):
        NC = self.number_of_cells()
        node = self.entity('node')
        cell = self.entity('cell')
        localEdge = self.ds.local_edge()
        jacobi = np.zeros((NC, 4), dtype=np.float)
        iprev = [3, 0, 1, 2]
        for i, j in localEdge:
            k = iprev[i]
            v0 = node[cell[:, j], :] - node[cell[:, i], :]
            v1 = node[cell[:, k], :] - node[cell[:, i], :]
            jacobi[:, i] = v0[:, 0]*v1[:, 1] - v0[:, 1]*v1[:, 0]
        return jacobi

    def cell_quality(self):
        jacobi = self.jacobi_at_corner()
        return jacobi.sum(axis=1)/4

    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_Quad = 9
            return VTK_Quad
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
        cax = axes.plot_trisurf(
                node[:, 0], node[:, 1],
                uh, cmap=cmap, lw=0.0)
        axes.figure.colorbar(cax, ax=axes)
        return axes

    ## @ingroup MeshGenerators
    @classmethod
    def from_polygon_gmsh(cls, vertices, h):
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
        node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)[:, 0:2].copy()

        # 获取四边形单元信息
        quadrilateral_type = 3  # 四边形单元的类型编号为 3
        quad_tags, quad_connectivity = gmsh.model.mesh.getElementsByType(quadrilateral_type)
        cell = np.array(quad_connectivity, dtype=np.uint64).reshape(-1, 4) - 1

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of quadrilaterals: {cell.shape[0]}")

        gmsh.finalize()

        NN = len(node)
        isValidNode = np.zeros(NN, dtype=np.bool_)
        isValidNode[cell] = True
        node = node[isValidNode]
        idxMap = np.zeros(NN, dtype=cell.dtype)
        idxMap[isValidNode] = range(isValidNode.sum())
        cell = idxMap[cell]

        return cls(node, cell)


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
        NN = (nx+1)*(ny+1)
        NC = nx*ny
        node = np.zeros((NN,2))
        X, Y = np.mgrid[
                box[0]:box[1]:(nx+1)*1j,
                box[2]:box[3]:(ny+1)*1j]
        node[:, 0] = X.flat
        node[:, 1] = Y.flat

        idx = np.arange(NN).reshape(nx+1, ny+1)
        cell = np.zeros((NC, 4), dtype=np.int_)
        cell[:, 0] = idx[0:-1, 0:-1].flat
        cell[:, 1] = idx[1:, 0:-1].flat
        cell[:, 2] = idx[1:, 1:].flat
        cell[:, 3] = idx[0:-1, 1:].flat

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
    def from_unit_square(cls, nx=10, ny=10, threshold=None):
        """
        Generate a quadrilateral mesh for a unit square.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return QuadrangleMesh instance
        """
        return cls.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny, threshold=threshold)


    @classmethod
    def from_one_quadrangle(cls, meshtype='square'):
        """
        Generate a quadrilateral mesh for a single quadrangle.

        @param meshtype Type of quadrangle mesh, options are 'square', 'zhengfangxing', 'rectangle', 'rec', 'juxing', 'rhombus', 'lingxing' (default: 'square')
        @return QuadrangleMesh instance
        """
        if meshtype in {'square'}:
            node = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0]], dtype=np.float64)
        elif meshtype in {'rectangle'}:
            node = np.array([
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0]], dtype=np.float64)
        elif meshtype in {'rhombus'}:
            node = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.5, np.sqrt(3) / 2],
                [0.5, np.sqrt(3) / 2]], dtype=np.float64)
        cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
        return cls(node, cell)

    @classmethod
    def from_triangle_mesh(cls, mesh):
        """
        @brief 把每个三角形分成三个四边形
        """
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        node0 = mesh.entity('node')
        cell0 = mesh.entity('cell')
        ec = mesh.entity_barycenter('edge')
        cc = mesh.entity_barycenter('cell')
        cell2edge = mesh.ds.cell_to_edge()

        node = np.r_['0', node0, ec, cc]
        cell = np.zeros((3*NC, 4), dtype=np.int_)
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
        return cls(node, cell)


QuadrangleMesh.set_ploter('2d')
