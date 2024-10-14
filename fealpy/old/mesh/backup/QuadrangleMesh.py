import numpy as np
from .TriangleMesh import TriangleMesh
from .Mesh2d import Mesh2d, Mesh2dDataStructure
from ..quadrature import TensorProductQuadrature, GaussLegendreQuadrature


class QuadrangleMeshDataStructure(Mesh2dDataStructure):
    localEdge = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
    localFace = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
    ccw = np.array([0, 1, 2, 3])

    NVE = 2
    NVF = 2
    NVC = 4

    NEC = 4
    NFC = 4


    def __init__(self, NN, cell):
        super().__init__(NN, cell)


## @defgroup MeshGenerators TetrhedronMesh Common Region Mesh Generators
class QuadrangleMesh(Mesh2d):
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

    def geo_dimension(self):
        return self.node.shape[-1]

    def integrator(self, q, etype='cell'):
        qf = GaussLegendreQuadrature(q)
        if etype in {'cell', 2}:
            return TensorProductQuadrature((qf, qf)) 
        elif etype in {'edge', 'face', 1}:
            return qf 

    def multi_index_matrix(self, p, etype='edge'):
        """
        @brief 获取网格边上的 p 次的多重指标矩阵

        @param[in] p 正整数 

        @return multiIndex  ndarray with shape (ldof, 2)
        """
        if etype in {'edge', 'face', 1}:
            ldof = p+1
            multiIndex = np.zeros((ldof, 2), dtype=np.int_)
            multiIndex[:, 0] = np.arange(p, -1, -1)
            multiIndex[:, 1] = p - multiIndex[:, 0]
            return multiIndex
        else:
            raise ValueError(f"etype is {etype}! For QuadrangleMesh, we just support etype with value `edge`, `face` or `1`")

    def edge_shape_function(self, bc, p=1):
        """
        @brief the shape function on edge  
        """
        if p == 1:
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

    def grad_edge_shape_function(self, bc, p=1):
        """
        @brief 计算形状为 (..., TD+1) 的重心坐标数组 bc 中, 每一个重心坐标处的 p 次 Lagrange 形函数值关于该重心坐标的梯度。
        """
        TD = bc.shape[-1] - 1
        multiIndex = self.multi_index_matrix(p) 
        ldof = multiIndex.shape[0] # p 次 Lagrange 形函数的个数

        c = np.arange(1, p+1)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=bc.dtype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=bc.dtype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
        return R # (..., ldof, TD+1)

    face_shape_function = edge_shape_function
    grad_face_shape_function = grad_edge_shape_function

    def shape_function(self, bc, p=1):
        """
        @brief 四边形单元上的形函数
        """
        assert isinstance(bc, tuple) and len(bc) == 2
        phi0 = self.edge_shape_function(bc[0], p=p) # x direction
        phi1 = self.edge_shape_function(bc[1], p=p) # y direction
        phi = np.einsum('im, kn->ikmn', phi0, phi1)
        shape = phi.shape[:-2] + (-1, )
        phi = phi.reshape(shape) # 展平自由度
        shape = (-1, 1) + phi.shape[-1:] # 增加一个单元轴，方便广播运算
        phi = phi.reshape(shape) # 展平积分点
        return phi

    def grad_shape_function(self, bc, p=1, variables='x'):
        """
        @brief  四边形单元形函数的导数

        @note 计算单元形函数关于参考单元变量 u=(xi, eta) 或者实际变量 x 梯度。

        bc 是一个长度为 2 的 tuple

        bc[i] 是一个一维积分公式的重心坐标数组

        这里假设 bc[0] == bc[1] == ... = bc[TD-1]
        """
        assert isinstance(bc, tuple) and len(bc) == 2

        Dlambda = np.array([-1, 1], dtype=self.ftype)
        # 一维基函数值
        # (NQ, p+1)
        phi = self.edge_shape_function(bc[0], p=p)  

        # 关于**一维变量重心坐标**的导数
        # lambda_0 = 1 - u 
        # lambda_1 = v 
        # (NQ, ldof, 2) 
        R = self.grad_edge_shape_function(bc[0], p=p)  

        # 关于一维变量的导数
        gphi = np.einsum('...ij, j->...i', R, Dlambda) # (..., ldof)

        gphi0 = np.einsum('im, kn->ikmn', gphi, phi)
        gphi1 = np.einsum('kn, im->kinm', phi, gphi)
        n = gphi0.shape[0]*gphi0.shape[1]
        shape = (n, (p+1)*(p+1), 2)
        gphi = np.zeros(shape, dtype=self.ftype)
        gphi[..., 0].flat = gphi0.flat
        gphi[..., 1].flat = gphi1.flat

        if variables == 'u':
            return gphi
        elif variables == 'x':
            J = self.jacobi_matrix(bc)
            G = self.first_fundamental_form(J)
            G = np.linalg.inv(G)
            gphi = np.einsum('...ikm, ...imn, ...ln->...ilk', J, G, gphi)
            return gphi

    def jacobi_matrix(self, bc, index=np.s_[:]):
        """
        @brief 
        """
        assert isinstance(bc, tuple) and len(bc) == 2
        NQ = len(bc[0])

        phi = bc[0]
        gphi = np.ones((NQ, 2), dtype=self.ftype)
        gphi[:, 0] = -1

        gphi0 = np.einsum('im, kn->ikmn', gphi, phi)
        gphi1 = np.einsum('kn, im->kinm', phi, gphi)
        ldof = gphi0.shape[0]*gphi0.shape[1]
        shape = (ldof, 4, 2)
        gphi = np.zeros(shape, dtype=self.ftype)
        gphi[..., 0].flat = gphi0.flat
        gphi[..., 1].flat = gphi1.flat

        cell = self.entity('cell')
        node = self.entity('node')
        J = np.einsum( 'ijn, ...ijk->...ink', node[cell[index, [0, 3, 1, 2]]], gphi)
        return J

    def first_fundamental_form(self, J):
        """
        @brief 由 Jacobi 矩阵计算第一基本形式。
        """
        shape = J.shape[0:-2] + (2, 2)
        G = np.zeros(shape, dtype=self.ftype)
        for i in range(2):
            G[..., i, i] = np.einsum('...d, ...d->...', J[..., i], J[..., i])
            for j in range(i+1, 2):
                G[..., i, j] = np.einsum('...d, ...d->...', J[..., i], J[..., j])
                G[..., j, i] = G[..., i, j]
        return G


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

    def edge_bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把边上积分点变换到网格边上的笛卡尔坐标点 
        """
        node = self.node
        entity = self.entity('edge')[index]
        p = np.einsum('...j, ijk->...ik', bc, node[entity])
        return p

    def cell_bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把单元上的积分点变换到网格单元上的笛卡尔坐标点 
        """
        assert len(bc) == 2
        node = self.entity('node')
        cell = self.entity('cell')[index]
        bc0 = bc[0] # (NQ0, 2)
        bc1 = bc[1] # (NQ1, 2)
        bc = np.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4) # (NQ0, NQ1, 2, 2)
        p = np.einsum('...j, cjk->...ck', bc, node[cell[:, [0, 3, 1, 2]]]) # (NQ, NC, 2)
        return p

    def uniform_refine(self, n=1):
        """
        @brief 一致加密四边形网格
        """
        for i in range(n):
            N = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            # Find the cutted edge  
            cell2edge = self.ds.cell_to_edge()
            edgeCenter = self.entity_barycenter('edge')
            cellCenter = self.entity_barycenter('cell')

            edge2center = np.arange(N, N+NE)

            cell = self.ds.cell
            cp = [cell[:, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[:, i]].reshape(-1, 1) for i in range(4)]
            cc = np.arange(N + NE, N + NE + NC).reshape(-1, 1)
 
            cell = np.zeros((4*NC, 4), dtype=np.int_)
            cell[0::4, :] = np.r_['1', cp[0], ep[0], cc, ep[3]] 
            cell[1::4, :] = np.r_['1', ep[0], cp[1], ep[1], cc]
            cell[2::4, :] = np.r_['1', cc, ep[1], cp[2], ep[2]]
            cell[3::4, :] = np.r_['1', ep[3], cc, ep[2], cp[3]]

            self.node = np.r_['0', self.node, edgeCenter, cellCenter]
            self.ds.reinit(N + NE + NC, cell)


    def number_of_local_ipoints(self, p):
        return (p+1)*(p+1)
    
    def number_of_global_ipoints(self, p):
        NP = self.number_of_nodes()
        if p > 1:
            NE = self.number_of_edges()
            NP += (p-1)*NE
        if p > 2:
            NC = self.number_of_cells()
            NP += (p-1)*(p-1)*NC
        return NP

    def interpolation_points(self, p, index=np.s_[:]):
        """
        @brief 获取四边形网格上所有 p 次插值点
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
            multiIndex = self.multi_index_matrix(p, 'edge')
            bc = multiIndex[1:-1, :]/p
            w = np.einsum('im, jn->ijmn', bc, bc).reshape(-1, 4)
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

    def node_to_ipoint(self, p, index=np.s_[:]):
        NN = self.number_of_nodes()
        return np.arange(NN)[index]

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

    def to_trimesh(self):
        cell = self.entity('cell')
        node = self.entity('node')
        hexCell2face = self.ds.cell_to_face()
        localCell = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int_)
        cell = cell[:, localCell].reshape(-1, 3)

        celldata = self.celldata
        nodedata = self.nodedata

        mesh = TriangleMesh(node, cell)
        for key in celldata:
            mesh.celldata[key] = np.tile(celldata[key], (6, 1)).T.reshape(-1)
        mesh.nodedata = nodedata
        return mesh

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



