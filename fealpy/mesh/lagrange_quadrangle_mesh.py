from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import estr2dim
from .mesh_base import TensorMesh
from .quadrangle_mesh import QuadrangleMesh

class LagrangeQuadrangleMesh(TensorMesh):
    def __init__(self, node: TensorLike, cell: TensorLike, p=1, surface=None,
            construct=False):
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)

        kwargs = bm.context(cell)
        GD = node.shape[1]
        self.p = p
        self.cell = cell
        self.surface = surface

        self.node = node
        self.ccw = bm.tensor([0, 2, 3, 1], **kwargs)

        if construct:
            self.construct()

        self.meshtype = 'lquad'
        self.linearmesh = None # 网格的顶点必须在球面上

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}

    def reference_cell_measure(self):
        return 1
    
    def interpolation_points(self, p: int, index: Index=_S):
        """Fetch all p-order interpolation points on the quadrangle mesh."""
        #TODO
        pass

    @classmethod
    def from_quadrangle_mesh(cls, mesh, p: int, surface=None):
        bnode = mesh.entity('node')
        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        if surface is not None:
            bnode[:],_ = surface.project(bnode)
            node,_ = surface.project(node)

        lmesh = cls(node, cell, p=p, construct=False)
        lmesh.linearmesh = mesh

        lmesh.edge2cell = mesh.edge2cell # (NF, 4)
        lmesh.cell2edge = mesh.cell_to_edge()
        lmesh.edge  = mesh.edge_to_ipoint(p)
        return lmesh

    # quadrature
    def quadrature_formula(self, q, etype: Union[int, str] = 'cell'):
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        qf = GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)
        if etype == 2:
            return TensorProductQuadrature((qf, qf))
        elif etype == 1:
            return TensorProductQuadrature((qf, ))
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def bc_to_point(self, bc: TensorLike, index: Index=_S, etype='cell'):
        node = self.node
        TD = len(bc)
        phi = self.shape_function(bc, p=p)
        p = bm.einsum('cqn, cni -> cqi', phi, node[entity])
        return p

    # shape function
    def shape_function(self, bc: TensorLike, p=None):
        """
        @berif 
        bc 是一个长度为 TD 的 tuple 数组
        bc[i] 是一个一维积分公式的重心坐标数组
        假设 bc[0]==bc[1]== ... ==bc[TD-1]
        """
        return self.linearmesh.cell_shape_function(bc, p)

    def grad_shape_function(self, bc: TensorLike, p=None, 
            index: Index=_S, variables='x'):
        return self.linearmesh.cell_grad_shape_function(bc, p, index, variables)

    # ipoints
    def number_of_local_ipoints(self, p:int, iptype:Union[int, str]='cell'):
        """
        @berif 每个ltri单元上插值点的个数
        """
        #TODO
        pass

    def number_of_global_ipoints(self, p:int):
        """
        @berif ltri网格上插值点总数
        """
        # TODO
        pass

    def cell_to_ipoint(self, p:int, index:Index=_S):
        """
        @berif 获取单元与插值点的对应关系
        """
        #TODO 复制粘贴 QuadrangleMesh 的代码并测试
        pass

    def face_to_ipoint(self, p: int, index: Index=_S):
        #TODO test
        return self.edge_to_ipoint(p, index)

    def entity_measure(self, etype=2, index:Index=_S):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return bm.zeros(1, dtype=bm.float64)
        else:
            raise ValueError(f"entity type:{etype} is erong!")

    def cell_area(self, q=None, index: Index=_S):
        """
        Calculate the area of a cell.
        """
        p = self.p
        q = p if q is None else q

        qf = self.quadrature_formula(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        G = self.first_fundamental_form(bcs)
        d = bm.sqrt(bm.linalg.det(G))
        a = bm.einsum('q, cq -> c', ws, d)
        return a

    def edge_length(self, q=None, index: Index=_S):
        """
        Calculate the length of the side.
        """
        p = self.p
        q = p if q is None else q
        qf = self.quadrature_formula(q, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        J = self.jacobi_matrix(bcs, index=index)
        l = bm.sqrt(bm.sum(J**2, axis=(-1, -2)))
        a = bm.einsum('q, cq -> c', ws, l)
        return a

    def cell_unit_normal(self, bc: TensorLike, index: Index=_S):
        """
        When calculating the surface,the direction of the unit normal at the integration point.
        """
        J = self.jacobi_matrix(bc, index=index)
        n = bm.cross(J[..., 0], J[..., 1], axis=-1)
        if self.GD == 3:
            l = bm.sqrt(bm.sum(n**2, axis=-1, keepdims=True))
            n /= l
        return n

    def jacobi_matrix(self, bc: TensorLike, p=None, index: Index=_S, return_grad=False):
        """
        @berif 计算参考单元 （xi, eta) 到实际 Lagrange 四边形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        TD = len(bc)
        entity = self.entity(TD, index)
        gphi = self.grad_shape_function(bc, p=p, variables='u')
        J = bm.einsum(
                'cin, cqim -> cqnm',
                self.node[entity[index], :], gphi) #(NC,ldof,GD),(NC,NQ,ldof,TD)
        if return_grad is False:
            return J #(NC,NQ,GD,TD)
        else:
            return J, gphi

    # fundamental form
    def first_fundamental_form(self, bc: Union[TensorLike, Tuple[TensorLike]],
            index: Index=_S, return_jacobi=False, return_grad=False):
        """
        Compute the first fundamental form of a mesh surface at integration points.
        """
        TD = len(bc)

        J = self.jacobi_matrix(bc, index=index,
                return_grad=return_grad)

        if return_grad:
            J, gphi = J

        shape = J.shape[0:-2] + (TD, TD)
        G = bm.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = bm.sum(J[..., i]**2, axis=-1)
            for j in range(i+1, TD):
                G[..., i, j] = bm.sum(J[..., i]*J[..., j], axis=-1)
                G[..., j, i] = G[..., i, j]
        if (return_jacobi is False) & (return_grad is False):
            return G
        elif (return_jacobi is True) & (return_grad is False):
            return G, J
        elif (return_jacobi is False) & (return_grad is True):
            return G, gphi
        else:
            return G, J, gphi

    def second_fundamental_form(self, bc: Union[TensorLike, Tuple[TensorLike]],
            index: Index=_S, return_jacobi=False, return_grad=False):
        """
        Compute the second fundamental form of a mesh surface at integration points.
        """
        #TODO
        pass

    # tools
    def integral(self, f, q=3, celltype=False) -> TensorLike:
        """
        @brief 在网格中数值积分一个函数
        """
        GD = self.geo_dimension()
        qf = self.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)

        rm = self.reference_cell_measure()
        G = self.first_fundamental_form(bcs)
        d = bm.sqrt(bm.linalg.det(G)) # 第一基本形式开方

        if callable(f):
            if getattr(f, 'coordtype', None) == 'barycentric':
                f = f(bcs)
            else:
                f = f(ps)

        cm = self.entity_measure('cell')

        if isinstance(f, (int, float)): #  u 为标量常函数
            e = f*cm
        elif bm.is_tensor(f):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = bm.einsum('q, cq..., cq -> c...', ws*rm, f, d)
        else:
            raise ValueError(f"Unsupported type of return value: {f.__class__.__name__}.")

        if celltype:
            return e
        else:
            return bm.sum(e)

    def error(self, u, v, q=3, power=2, celltype=False) -> TensorLike:
        """
        @brief Calculate the error between two functions.
        """
        GD = self.geo_dimension()
        qf = self.quadrature_formula(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)

        rm = self.reference_cell_measure()
        G = self.first_fundamental_form(bcs)
        d = bm.sqrt(bm.linalg.det(G)) # 第一基本形式开方

        if callable(u):
            if getattr(u, 'coordtype', None) == 'barycentric':
                u = u(bcs)
            else:
                u = u(ps)

        if callable(v):
            if getattr(v, 'coordtype', None) == 'barycentric':
                v = v(bcs)
            else:
                v = v(ps)
        cm = self.entity_measure('cell')
        NC = self.number_of_cells()
        if v.shape[-1] == NC:
            v = bm.swapaxes(v, 0, -1)
        #f = bm.power(bm.abs(u - v), power)
        f = bm.abs(u - v)**power
        if len(f.shape) == 1:
            f = f[:, None]

        if isinstance(f, (int, float)): # f为标量常函数
            e = f*cm
        elif bm.is_tensor(f):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = bm.einsum('q, cq..., cq -> c...', ws*rm, f, d)

        if celltype is False:
            #e = bm.power(bm.sum(e), 1/power)
            e = bm.sum(e)**(1/power)
        else:
            e = bm.power(bm.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e # float or (NC, )

    # 可视化
    def vtk_cell_type(self, etype='cell'):
        """

        Notes
        -----
            返回网格单元对应的 vtk 类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_QUADRILATERAL = 70 
            return VTK_LAGRANGE_QUADRILATERAL
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE


    def to_vtk(self, etype='cell', index: Index=_S, fname=None):
        """
        Parameters
        ----------

        @berif 把网格转化为 VTK 的格式
        """
        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = bm.concatenate((node, bm.zeros((node.shape[0], 1), dtype=bm.float64)), axis=1)

        cell = self.entity(etype, index)
        cellType = self.vtk_cell_type(etype)
        idx = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell[:, idx]), axis=1)
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)

