from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import SimplexMesh, estr2dim
from .plot import Plotable


class TriangleMesh(SimplexMesh, Plotable):
    def __init__(self, node: TensorLike, cell: TensorLike) -> None:
        """
        """
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)
        kwargs = bm.context(cell) 
        self.node = node
        self.cell = cell
        self.localEdge = bm.tensor([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.localFace = bm.tensor([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.ccw = bm.tensor([0, 1, 2], **kwargs)

        self.localCell = bm.tensor([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)

        self.construct()

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}


    # entity
    def entity_measure(self, etype: Union[int, str], index: Optional[Index]=None) -> TensorLike:
        """
        """
        node = self.node
        kwargs = bm.context(node)

        if isinstance(etype, str):
            etype = estr2dim(self, etype)

        if etype == 0:
            return bm.tensor([0,], **kwargs)
        elif etype == 1:
            edge = self.entity(1, index)
            return bm.edge_length(edge, node)
        elif etype == 2:
            cell = self.entity(2, index)
            return bm.simplex_measure(cell, node)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
  
    # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell',
                           qtype: str='legendre'): # TODO: other qtype
        from ..quadrature import TriangleQuadrature
        from ..quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        kwargs = {'dtype': self.ftype}
        if etype == 2:
            quad = TriangleQuadrature(q, **kwargs)
        elif etype == 1:
            quad = GaussLegendreQuadrature(q, **kwargs)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
        return quad

    # shape function
    def grad_lambda(self, index: Index=_S) -> TensorLike:
        """
        """
        node = self.node
        cell = self.cell[index]
        GD = self.GD
        if GD == 2:
            return bm.triangle_grad_lambda_2d(cell, node)
        elif GD == 3:
            return bm.triangle_grad_lambda_3d(cell, node)
    
    def rot_lambda(self, index: Index=_S): # TODO
        pass
    
    def grad_shape_function(self, bc, p=1, index: Index=_S, variables='x'):
        """
        @berif 这里调用的是网格空间基函数的梯度
        """
        R = bm.simplex_grad_shape_function(bc, p)
        if variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = bm.einsum('...ij, kjm -> k...im', R, Dlambda)
            return gphi  # (NC, NQ, ldof, GD)
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
        pass

    # ipoint
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return bm.simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        num = (NN, NE, NC)
        return bm.simplex_gdof(p, num)
    
    def interpolation_points(self, p: int, index: Index=_S):
        """Fetch all p-order interpolation points on the triangle mesh."""
        node = self.entity('node')
        if p == 1:
            return node
        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")

        ipoint_list = []
        kwargs = {'dtype': self.ftype}

        GD = self.geo_dimension()
        ipoint_list.append(node) # ipoints[:NN, :]

        edge = self.entity('edge')
        w = bm.zeros((p - 1, 2), dtype=bm.float64)
        w[:, 0] = bm.arange(p - 1, 0, -1, dtype=bm.float64) / p
        w[:, 1] = bm.flip(w[:, 0], axis=0) 
        ipoints_from_edge = bm.einsum('ij, ...jm->...im', w,
                                         node[edge, :]).reshape(-1, GD) # ipoints[NN:NN + (p - 1) * NE, :]
        ipoint_list.append(ipoints_from_edge)

        if p >= 3:
            TD = self.top_dimension()
            cell = self.entity('cell')
            multiIndex = bm.multi_index_matrix(p, TD, dtype=self.ftype)
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:, 0] | isEdgeIPoints[:, 1] |
                                isEdgeIPoints[:, 2])
            multiIndex = multiIndex[isInCellIPoints, :]
            w = multiIndex / p
            
            ipoints_from_cell = bm.einsum('ij, kj...->ki...', w,
                                          node[cell, :]).reshape(-1, GD) # ipoints[NN + (p - 1) * NE:, :]
            ipoint_list.append(ipoints_from_cell)

        return bm.concatenate(ipoint_list, axis=0)  # (gdof, GD)

    def cell_to_ipoint(self, p: int, index: Index=_S):
        cell = self.cell
        if p == 1:
            return cell[index]

        mi = self.multi_index_matrix(p, 2)
        idx0, = bm.nonzero(mi[:, 0] == 0)
        idx1, = bm.nonzero(mi[:, 1] == 0)
        idx2, = bm.nonzero(mi[:, 2] == 0)
        kwargs = {'dtype': self.itype}

        face2cell = self.face_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        e2p = self.edge_to_ipoint(p)
        ldof = self.number_of_local_ipoints(p, 'cell')
        c2p = bm.zeros((NC, ldof), **kwargs)

        flag = face2cell[:, 2] == 0
        c2p[face2cell[flag, 0][:, None], idx0] = e2p[flag]

        flag = face2cell[:, 2] == 1
        idx1_ = bm.flip(idx1, axis=0)
        c2p[face2cell[flag, 0][:, None], idx1_] = e2p[flag]

        flag = face2cell[:, 2] == 2
        c2p[face2cell[flag, 0][:, None], idx2] = e2p[flag]

        iflag = face2cell[:, 0] != face2cell[:, 1]

        flag = iflag & (face2cell[:, 3] == 0)
        idx0_ = bm.flip(idx0, axis=0)
        c2p[face2cell[flag, 1][:, None], idx0_] = e2p[flag]

        flag = iflag & (face2cell[:, 3] == 1)
        c2p[face2cell[flag, 1][:, None], idx1] = e2p[flag]

        flag = iflag & (face2cell[:, 3] == 2)
        idx2_ = bm.flip(idx2, axis=0)
        c2p[face2cell[flag, 1][:, None], idx2_] = e2p[flag]

        cdof = (p-1)*(p-2)//2
        flag = bm.sum(mi > 0, axis=1) == 3
        c2p[:, flag] = NN + NE*(p-1) + bm.arange(NC*cdof, **kwargs).reshape(NC, cdof)
        return c2p[index]

    def face_to_ipoint(self, p: int, index: Index=_S):
        return self.edge_to_ipoint(p, index)
    def cell_to_edge_sign(self):
        NC = self.number_of_cells()
        NEC = self.number_of_edges_of_cells()
        edge2cell = self.face_to_cell() #TODO：ds没有edge_to_cell
        cell2edgeSign = bm.zeros((NC, NEC), dtype=bm.bool)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]]=True
        return cell2edgeSign
    def prolongation_matrix(self, po: int, p1: int):
        """
        @brief 生成从 p0 元到 p1 元的延拓矩阵，假定 0 < p0 < p1
        """
        pass

    def edge_frame(self, index: Index=_S):
        """
        @brief 计算二维网格中每条边上的局部标架
        """
        pass
    def edge_unit_tangent(self, index=_S):
        """
        @brief Calculate the tangent vector with unit length of each edge.See `Mesh.edge_tangent`.
        """
        node = self.entity('node') 
        edge = self.entity('edge', index=index)
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        length = bm.sqrt(bm.square(v).sum(axis=1))
        return v/length.reshape(-1, 1)

    
    def edge_normal(self, index: Index=_S):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_tangent(index=index)
        w = bm.array([[0, -1], [1, 0]], dtype=self.ftype)
        return v@w
    def edge_unit_normal(self, index: Index=_S):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_unit_tangent(index=index)
        w = bm.array([[0, -1], [1, 0]], dtype=self.ftype)
        return v@w



    def uniform_refine(self, n=1, surface=None, interface=None, returnim=False):
        """

        """

        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            NE = self.number_of_edges()
            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')
            cell2edge = self.cell_to_edge()
            edge2newNode = bm.arange(NN, NN + NE)
            newNode = (node[edge[:, 0], :] + node[edge[:, 1], :]) / 2.0

            self.node = bm.concatenate((node, newNode), axis=0)
            p = bm.concatenate((cell, edge2newNode[cell2edge]), axis=1)
            self.cell = bm.concatenate(
                    (p[:,[0,5,4]], p[:,[5,1,3]], p[:,[4,3,2]], p[:,[3,4,5]]),
                    axis=0)
            #TODO: call self.clear() 清理暂存的数据
            self.construct()

    def is_crossed_cell(self, point, segment):
        """
        @berif 给定一组线段，找到这些线段的一个邻域单元集合, 且这些单元要满足一定的连通
        性
        """
        pass
    
    def location(self, points):
        """
        @breif  给定一组点 p , 找到这些点所在的单元

        这里假设：

        1. 所有点在网格内部，
        2. 网格中没有洞
        3. 区域还要是凸的
        """
        pass

    def circumcenter(self, index: Index=_S, returnradius=False):
        """
        @brief 计算三角形外接圆的圆心和半径
        """
        node = self.node
        cell = self.cell
        GD = self.geo_dimension()

        v0 = node[cell[index, 2], :] - node[cell[index, 1], :]
        v1 = node[cell[index, 0], :] - node[cell[index, 2], :]
        v2 = node[cell[index, 1], :] - node[cell[index, 0], :]
        nv = bm.cross(v2, -v1)
        if GD == 2:
            area = nv / 2.0
            x2 = bm.sum(node ** 2, axis=1, keepdims=True)
            w0 = x2[cell[index, 2]] + x2[cell[index, 1]]
            w1 = x2[cell[index, 0]] + x2[cell[index, 2]]
            w2 = x2[cell[index, 1]] + x2[cell[index, 0]]
            W = bm.array([[0, -1], [1, 0]], dtype=self.ftype)
            fe0 = w0 * v0 @ W
            fe1 = w1 * v1 @ W
            fe2 = w2 * v2 @ W
            c = 0.25 * (fe0 + fe1 + fe2) / area.reshape(-1, 1)
            R = bm.sqrt(bm.sum((c - node[cell[index, 0], :]) ** 2, axis=1))
        elif GD == 3:
            length = bm.sqrt(bm.sum(nv ** 2, axis=1))
            n = nv / length.reshape((-1, 1))
            l02 = bm.sum(v1 ** 2, axis=1, keepdims=True)
            l01 = bm.sum(v2 ** 2, axis=1, keepdims=True)
            d = 0.5 * (l02 * bm.cross(n, v2) + l01 * bm.cross(-v1, n)) / length.reshape(-1, 1)
            c = node[cell[index, 0]] + d
            R = bm.sqrt(bm.sum(d ** 2, axis=1))

        if returnradius:
            return c, R
        else:
            return c

    def angle(self):
        pass

    def show_angle(self, axes, angle=None):
        """
        @brief 显示网格角度的分布直方图
        """
        pass
    
    def cell_quality(self, measure='radius_ratio'):
        if measure == 'radius_ratio':
            return radius_ratio(self)

    def show_quality(self, axes, qtype=None, quality=None):
        """
        @brief 显示网格质量分布的分布直方图
        """
        pass

    def edge_swap(self):
        pass

    def odt_iterate(self):
        pass

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

    def bisect(): #TODO
        pass

    def coarsen(self, isMarkedCell=None, options={}):
        pass

    def label(self, node=None, cell=None, cellidx=None):
        """
        单元顶点的重新排列，使得cell[:, [1, 2]] 存储了单元的最长边
        Parameter
        -------
        Return 
        -------
        cell ： in-place modify
        """
        pass

    def delete_degree_4(self):
        pass

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
        pass

    def bisect_1(self, isMarkedCell=None, options={'disp': True}):
        pass

    def jacobian_matrix(self, index: Index=_S):
        """
        @brief 获得三角形单元对应的 Jacobian 矩阵
        """
        NC = self.number_of_cells()
        GD = self.geo_dimension()

        node = self.entity('node')
        cell = self.entity('cell')

        J = bm.zeros((NC, GD, 2), dtype=self.ftype)

        J[..., 0] = node[cell[:, 1]] - node[cell[:, 0]]
        J[..., 1] = node[cell[:, 2]] - node[cell[:, 0]]

        return J

    def point_to_bc(self, point):
        """
        @brief 找到定点 point 所在的单元，并计算其重心坐标 
        """
        pass

    def mark_interface_cell(self, phi):
        """
        @brief 标记穿过界面的单元
        """
        pass

    def mark_interface_cell_with_curvature(self, phi, hmax=None):
        """
        @brief 标记曲率大的单元
        """
        pass

    def mark_interface_cell_with_type(self, phi, interface):
        """
        @brief 等腰直角三角形，可以分为两类
            - Type A：两条直角边和坐标轴平行
            - Type B: 最长边和坐标轴平行
        """
        pass

    def bisect_interface_cell_with_curvature(self, interface, hmax):
        pass

    def show_function(self, plot, uh, cmap=None):
        pass

    @classmethod
    def show_lattice(cls, p=1, shownltiindex=False):
        """
        @berif 展示三角形上的单纯形格点
        """
        pass

    @classmethod
    def show_shape_function(cls, p=1, funtype='L'):
        """
        @brief 可视化展示三角形单元上的 p 次基函数
        """
        pass

    @classmethod
    def show_global_basis_function(cls, p=3):
        """
        @brief 展示通过单元基函数的拼接+零扩展的方法获取整体基函数的过程
        """
        pass

    @classmethod
    def from_one_triangle(cls, meshtype='iso'):
        if meshtype == 'equ':
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, bm.sqrt(bm.tensor(3)) / 2]], dtype=bm.float64)
        elif meshtype == 'iso':
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2]], dtype=bm.int32)
        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_square_domain_with_fracture(cls):
        node = bm.tensor([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]], dtype=bm.float64)

        cell = bm.tensor([
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5]], dtype=bm.int32)

        return cls(node, cell)

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
        return cls.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny, 
                threshold=threshold, ftype=bm.float64, itype=bm.int32)

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
        x = bm.linspace(box[0], box[1], nx+1, dtype=bm.float64)
        y = bm.linspace(box[2], box[3], ny+1, dtype=bm.float64)
        X, Y = bm.meshgrid(x, y, indexing='ij')
    
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

        idx = bm.arange(NN).reshape(nx + 1, ny + 1)
        cell0 = bm.concatenate((
            idx[1:, 0:-1].T.reshape(-1, 1),
            idx[1:, 1:].T.reshape(-1, 1),
            idx[0:-1, 0:-1].T.reshape(-1, 1),
            ), axis=1)
        cell1 = bm.concatenate((
            idx[0:-1, 1:].T.reshape(-1, 1),
            idx[0:-1, 0:-1].T.reshape(-1, 1),
            idx[1:, 1:].T.reshape(-1, 1)
            ), axis=1)
        cell = bm.concatenate((cell0, cell1), axis=0)

        if threshold is not None:
            bc = bm.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = bm.zeros(NN, dtype=bm.bool)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = bm.arange(isValidNode.sum())
            cell = idxMap[cell]

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_sphere_surface(cls, refine=0):
        """
        @brief  Generate a triangular mesh on a unit sphere surface.
        @return the triangular mesh.
        """
        t = (bm.sqrt(bm.tensor(5)) - 1) / 2
        node = bm.array([
            [0, 1, t], [0, 1, -t], [1, t, 0], [1, -t, 0],
            [0, -1, -t], [0, -1, t], [t, 0, 1], [-t, 0, 1],
            [t, 0, -1], [-t, 0, -1], [-1, t, 0], [-1, -t, 0]], dtype=bm.float64)
        cell = bm.array([
            [6, 2, 0], [3, 2, 6], [5, 3, 6], [5, 6, 7],
            [6, 0, 7], [3, 8, 2], [2, 8, 1], [2, 1, 0],
            [0, 1, 10], [1, 9, 10], [8, 9, 1], [4, 8, 3],
            [4, 3, 5], [4, 5, 11], [7, 10, 11], [0, 10, 7],
            [4, 11, 9], [8, 4, 9], [5, 7, 11], [10, 9, 11]], dtype=bm.int32)
        mesh = cls(node, cell)
        mesh.uniform_refine(refine)
        node = mesh.node
        cell = mesh.entity('cell')
        d = bm.sqrt(node[:, 0] ** 2 + node[:, 1] ** 2 + node[:, 2] ** 2) - 1
        l = bm.sqrt(bm.sum(node ** 2, axis=1))
        n = node / l[..., None]
        node = node - d[..., None] * n
        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_ellipsoid(cls, radius=[9, 3, 1], refine=0):
        """
        a: 椭球的长半轴
        b: 椭球的中半轴
        c: 椭球的短半轴
        """
        a, b, c = radius
        mesh = TriangleMesh.from_unit_sphere_surface()
        mesh.uniform_refine(refine)
        node = mesh.node
        cell = mesh.entity('cell')
        node[:, 0]*=a 
        node[:, 1]*=b 
        node[:, 2]*=c
        return cls(node, cell)
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_ellipsoid_surface(cls, ntheta=10, nphi=10,
                               radius=(1, 1, 1),
                               theta=None,
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
        if theta is None:
            theta = (bm.pi / 4, 3 * bm.pi / 4)

        a, b, c = radius
        if phi is None:  # 默认为一封闭的带状区域
            NN = (ntheta + 1) * nphi
        else:  # 否则为四边形区域
            NN = (ntheta + 1) * (nphi + 1)

        NC = ntheta * nphi

        if phi is None:
            theta = bm.linspace(theta[0], theta[1], ntheta+1, dtype=bm.float64)
            l = bm.linspace(0, 2*bm.pi, nphi+1, dtype=bm.float64)
            U, V = bm.meshgrid(theta, l, indexing='ij')
            U = U[:, 0:-1]  # 去掉最后一列
            V = V[:, 0:-1]  # 去年最后一列
        else:
            theta = bm.linspace(theta[0], theta[1], ntheta+1, dtype=bm.float64)
            phi = bm.linspace(phi[0], phi[1], nphi+1, dtype=bm.float64)
            U, V = bm.meshgrid(theta, phi, indexing='ij')

        node = bm.zeros((NN, 3), dtype=bm.float64)
        X = a * bm.sin(U) * bm.cos(V)
        Y = b * bm.sin(U) * bm.sin(V)
        Z = c * bm.cos(U)
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), axis=1)
        
        idx = bm.zeros((ntheta + 1, nphi + 1), dtype=bm.int32)
        if phi is None:
            idx[:, 0:-1] = bm.arange(NN).reshape(ntheta + 1, nphi)
            idx[:, -1] = idx[:, 0]
        else:
            idx = bm.arange(NN).reshape(ntheta + 1, nphi + 1)
        cell = bm.zeros((2 * NC, 3), dtype=bm.int32)
        cell0 = bm.concatenate((
            idx[1:, 0:-1].T.reshape(-1, 1),
            idx[1:, 1:].T.reshape(-1, 1),
            idx[0:-1, 0:-1].T.reshape(-1, 1)), axis=1)
        cell1 = bm.concatenate((
            idx[0:-1, 1:].T.reshape(-1, 1),
            idx[0:-1, 0:-1].T.reshape(-1, 1),
            idx[1:, 1:].T.reshape(-1, 1)), axis=1)
        cell = bm.concatenate((cell0, cell1), axis=1).reshape(-1, 3)

        if returnuv:
            return cls(node, cell), U.flatten(), V.flatten()
        else:
            return cls(node, cell)

    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_TRIANGLE = 5
            return VTK_TRIANGLE
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, fname=None, etype='cell', index: Index=_S):
        """
        @brief 把网格转化为 vtk 的数据格式
        """
        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = bm.concatenate((node, bm.zeros((node.shape[0], 1), dtype=bm.float64)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        NV = cell.shape[-1]

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell), axis=1)
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                         nodedata=self.nodedata,
                         celldata=self.celldata)

TriangleMesh.set_ploter('2d')
