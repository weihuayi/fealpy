from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import simplex_gdof, simplex_ldof 
from .mesh_base import HomogeneousMesh, estr2dim
from .triangle_mesh import TriangleMesh


class LagrangeTriangleMesh(HomogeneousMesh):
    def __init__(self, node: TensorLike, cell: TensorLike, p=None, curve=None, 
            surface=None, construct=False):
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)

        kwargs = bm.context(cell)

        if p is None:
            NV = cell.shape[-1]
            self.p = int(-3 + bm.sqrt(1 + 8 * NV)) // 2 
        else:
            NV = (p + 1) * (p + 2) // 2
            if cell.shape[-1] != NV:
                raise ValueError(f"cell.shape[-1] != {NV}, p = {p}.")
            else:
                self.p = p

        self.node = node
        self.cell = cell
        self.surface = surface

        self.construct_local_edge()
        self.localFace = self.localEdge
        self.construct_local_cell()
        if construct:
            self.construct()

        self.meshtype = 'ltri'
        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}

    def reference_cell_measure(self):
        return 0.5

    def construct_local_edge(self) -> None:
        """
        Generate the local edges for Lagrange elements of order p.
        """
        TD = self.top_dimension()
        multiIndex = bm.multi_index_matrix(self.p, TD)

        edge0 = bm.where(multiIndex[:, 0] == 0)[0]
        edge1 = bm.where(multiIndex[:, 1] == 0)[0]
        edge2 = bm.where(multiIndex[:, 2] == 0)[0]

        self.localEdge = bm.stack([
            edge0, bm.flip(edge1), edge2], axis=0)  # (3, p+1)

    def construct_local_cell(self) -> TensorLike:
        """
        TODO: Generate the local cells for Lagrange elements of order p.
        """
        self.localCell = None

    def interpolation_points(self, p: int):
        """Fetch all p-order interpolation points on the triangle mesh."""

        if p < 1:
            raise ValueError(f"p must be at least 1, but got {p}.")

        GD = self.geo_dimension()
        node = self.entity('node')
        edge = self.entity('edge')

        isCornerNode = bm.zeros(len(node), dtype=bm.bool)
        isCornerNode = bm.set_at(isCornerNode, edge[:, [0, -1]], True)
        if p == 1:
            return node[isCornerNode]
        elif p == self.p:
            return node

        ipoints = []
        ipoints.append(node[isCornerNode])  # the corner nodes 

        # edge interior interpolation points 
        w = bm.multi_index_matrix(p, 1, dtype=self.ftype)
        w = w[1:-1]/p
        ipoints.append(self.bc_to_point(w).reshape(-1, GD))

        # cell interior interpolation_points 
        if p >= 3:
            TD = self.top_dimension()
            multiIndex = bm.multi_index_matrix(p, TD, dtype=self.ftype)
            isInCellIPoints = bm.sum(multiIndex > 0, axis=-1) == 3
            multiIndex = multiIndex[isInCellIPoints, :]
            w = multiIndex / p
            
            ipoints.append(self.bc_to_point(w).reshape(-1, GD))

        return bm.concatenate(ipoints, axis=0)

    @classmethod
    def from_box(cls, box, p: int, nx=2, ny=2):
        mesh = TriangleMesh.from_box(box, nx, ny)
        return cls.from_triangle_mesh(mesh, p)

    @classmethod
    def from_curve_triangle_mesh(cls, mesh, p: int, curve=None):
        init_node = mesh.entity('node')

        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        if curve is not None:
            boundary_edge = mesh.boundary_edge_flag()
            e2p = mesh.edge_to_ipoint(p)[boundary_edge].flatten()

            init_node[:], _ = curve.project(init_node) 
            node[e2p], _ = curve.project(node[e2p])

        lmesh = cls(node, cell, p=p, construct=True)
        lmesh.linearmesh = mesh

        lmesh.edge2cell = mesh.edge2cell # (NF, 4)
        lmesh.cell2edge = mesh.cell2edge
        lmesh.edge  = mesh.edge_to_ipoint(p)
        return lmesh

    @classmethod
    def from_triangle_mesh(cls, mesh, p: int, surface=None):
        init_node = mesh.entity('node')
        cls.surface = surface
        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        if surface is not None:
            init_node[:], _ = surface.project(init_node) 
            node, _ = surface.project(node)

        lmesh = cls(node, cell, p=p, construct=True)
        lmesh.linearmesh = mesh

        lmesh.edge2cell = mesh.edge2cell # (NF, 4)
        lmesh.cell2edge = mesh.cell2edge
        lmesh.edge  = mesh.edge_to_ipoint(p)
        return lmesh 

    def uniform_refine(self , n:int = 1 ):
        """
        """
        GD = self.geo_dimension()
        node = self.entity('node')
        edge = self.entity('edge')
        cell = self.entity('cell')

        isCornerNode = bm.zeros(len(node), dtype=bm.bool)
        isCornerNode = bm.set_at(isCornerNode, edge[:, [0, -1]], True)

        nodes = [] 

        # the corner nodes 
        nodes.append(node[isCornerNode])


        # the edge interior interpolation points        
        w = bm.multi_index_matrix(2 * self.p + 1, 1)[1:-1]/2
        nodes.append(self.bc_to_point(w).reshape(-1, self.GD))

        # the edge in cell 
        A = bm.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]], dtype=bm.float64)

        w = bm.multi_index_matrix(self.p + 1, 1)[1:-1]/self.p
        w0 = bm.einsum('ij, jk->ik', w, A[[4, 5], :])
        w1 = bm.enisum('ij, jk->ik', w, A[[5, 3], :])
        w2 = bm.einsum('ij, jk->ik', w, A[[3, 4], :])
        w = bm.concatenate((w0, w1, w2), axis=0)
        nodes.append(self.bc_to_point(w).reshape(-1, self.GD))

        if self.p >= 3:
            TD = self.top_dimension()
            multiIndex = bm.multi_index_matrix(self.p, TD)
            isInCellNodes = bm.sum(multiIndex > 0, axis=-1) == 3
            w = multiIndex[isInCellNodes, :] / self.p
            w0 = bm.einsum('ij, jk->ik', w, A[[0, 5, 4], :])
            w1 = bm.einsum('ij, jk->ik', w, A[[1, 3, 5], :])
            w2 = bm.einsum('ij, jk->ik', w, A[[2, 4, 3], :])
            w3 = bm.einsum('ij, jk->ik', w, A[[3, 4, 5], :])
            w = bm.concatenate((w0, w1, w2, w3), axis=0)
            nodes.append(self.bc_to_point(w).reshape(-1, self.GD))

        node = bm.concatenate(nodes, axis=0)
        return node


    # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell'):
        from ..quadrature import TriangleQuadrature
        from ..quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 2:
            quad = TriangleQuadrature(q)
        elif etype == 1:
            quad = GaussLegendreQuadrature(q)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad

    def bc_to_point(self, bc: TensorLike, index: Index=_S, etype='cell'):
        """
        """
        node = self.node
        TD = bc.shape[-1] - 1
        entity = self.entity(TD, index=index) # 
        phi = self.shape_function(bc) # (NC, NQ, NVC)
        p = bm.einsum('cqn, cni -> cqi', phi, node[entity])
        return p
    
    # shape function
    def shape_function(self, bc: TensorLike, p: int=None, variables='x',index: Index=_S):
        p = self.p if p is None else p 
        phi = bm.simplex_shape_function(bc, p=p)
        if variables == 'u':
            return phi
        elif variables == 'x':
            return phi[None, :, :]

    def grad_shape_function(self, bc: TensorLike, p: int=None, 
                            index: Index=_S, variables='x'):
        """
        @berif 计算单元形函数关于参考单元变量 u=(xi, eta) 或者实际变量 x 梯度。

        lambda_0 = 1 - xi - eta
        lambda_1 = xi
        lambda_2 = eta

        """
        p = self.p if p is None else p 
        TD = bc.shape[-1] - 1
        if TD == 2:
            Dlambda = bm.array([[-1, -1], [1, 0], [0, 1]], dtype=bm.float64)
        else:
            Dlambda = bm.array([[-1], [1]], dtype=bm.float64)
        R = bm.simplex_grad_shape_function(bc, p=p) # (NQ, ldof, TD+1)
        gphi = bm.einsum('qij, jn -> qin', R, Dlambda) # (NQ, ldof, TD)
        
        if variables == 'u':
            return gphi[None, :, :, :] #(1, ..., ldof, TD)
        elif variables == 'x':
            J = self.jacobi_matrix(bc, index=index)
            G = self.first_fundamental_form(J)
            d = bm.linalg.inv(G)
            gphi = bm.einsum('cqkm, cqmn, qln -> cqlk', J, d, gphi) 
            return gphi

    # ipoints
    def number_of_local_ipoints(self, p:int, iptype:Union[int, str]='cell'):
        """
        @berif 每个ltri单元上插值点的个数
        """
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p:int):
        """
        @berif ltri网格上插值点总数
        """
        NN = self.linearmesh.number_of_nodes()
        NE = self.linearmesh.number_of_edges()
        NC = self.linearmesh.number_of_cells()
        num = (NN, NE, NC)
        return simplex_gdof(p, num) 

    def cell_to_ipoint(self, p:int, index:Index=_S):
        """
        @berif 获取单元与插值点的对应关系
        """
        sp = self.p
        cell = self.cell[:, [0, -sp-1, 1]]  # 取角点

        if p == 1:
            return cell[index]

        TD = self.top_dimension()
        mi = self.multi_index_matrix(p, TD)
        idx0, = bm.nonzero(mi[:, 0] == 0)
        idx1, = bm.nonzero(mi[:, 1] == 0)
        idx2, = bm.nonzero(mi[:, 2] == 0)

        face2cell = self.face_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        e2p = self.edge_to_ipoint(p)
        ldof = self.number_of_local_ipoints(p, 'cell')

        kwargs = bm.context(cell)
        c2p = bm.zeros((NC, ldof), **kwargs)

        flag = face2cell[:, 2] == 0
        c2p = bm.set_at(c2p, (face2cell[flag, 0][:, None], idx0), e2p[flag])

        flag = face2cell[:, 2] == 1
        idx1_ = bm.flip(idx1, axis=0)
        c2p = bm.set_at(c2p, (face2cell[flag, 0][:, None], idx1_), e2p[flag])

        flag = face2cell[:, 2] == 2
        c2p = bm.set_at(c2p, (face2cell[flag, 0][:, None], idx2), e2p[flag])

        iflag = face2cell[:, 0] != face2cell[:, 1]
        flag = iflag & (face2cell[:, 3] == 0)
        idx0_ = bm.flip(idx0, axis=0)
        c2p = bm.set_at(c2p, (face2cell[flag, 1][:, None], idx0_), e2p[flag])

        flag = iflag & (face2cell[:, 3] == 1)
        c2p = bm.set_at(c2p, (face2cell[flag, 1][:, None], idx1), e2p[flag])

        flag = iflag & (face2cell[:, 3] == 2)
        idx2_ = bm.flip(idx2, axis=0)
        c2p = bm.set_at(c2p, (face2cell[flag, 1][:, None], idx2_),  e2p[flag])

        cdof = (p-1)*(p-2)//2
        flag = bm.sum(mi > 0, axis=1) == 3
        val = NN + NE*(p-1) + bm.arange(NC*cdof, **kwargs).reshape(NC, cdof)
        c2p = bm.set_at(c2p, (..., flag), val)
        return c2p[index]

    def edge_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        """Get the relationship between edges and integration points."""
        NN = self.linearmesh.number_of_nodes()
        NE = self.number_of_edges()
        edges = self.edge[index]
        # kwargs = {'dtype': edges.dtype}
        kwargs = bm.context(edges)
        indices = bm.arange(NE, **kwargs)[index]
        return bm.concatenate([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + bm.arange(0, p-1, **kwargs) + NN,
            edges[:, -1].reshape(-1, 1),
        ], axis=-1)

    def face_to_ipoint(self, p: int, index: Index=_S):
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
        GD = self.geo_dimension()

        qf = self.quadrature_formula(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        n = bm.cross(J[..., 0], J[..., 1], axis=-1)
        if GD == 3:
            n = bm.sqrt(bm.sum(n**2, axis=-1)) # (NC, NQ)
        a = bm.einsum('q, cq -> c', ws, n)/2.0
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

    def jacobi_matrix(self, bc: TensorLike, index: Index=_S, return_grad=False):
        """
        @berif 计算参考单元 （xi, eta) 到实际 Lagrange 三角形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        TD = bc.shape[-1] - 1
        entity = self.entity(TD, index)
        gphi = self.grad_shape_function(bc, variables='u')
        J = bm.einsum(
                'cin, cqim -> cqnm',
                self.node[entity[index], :], gphi) #(NC,ldof,GD),(NC,NQ,ldof,TD)
        if return_grad is False:
            return J #(NC,NQ,GD,TD)
        else:
            return J, gphi

    # fundamental form
    def first_fundamental_form(self, J: TensorLike, index: Index=_S):
        """
        Compute the first fundamental form of a mesh surface at integration points.
        """
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        G = bm.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = bm.sum(J[..., i]**2, axis=-1)
            for j in range(i+1, TD):
                G[..., i, j] = bm.sum(J[..., i]*J[..., j], axis=-1)
                G[..., j, i] = G[..., i, j]
        return G
 
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
        J = self.jacobi_matrix(bcs)
        G = self.first_fundamental_form(J) 
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
        J = self.jacobi_matrix(bcs)
        G = self.first_fundamental_form(J) 
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
        @berif  返回网格单元对应的 vtk类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_TRIANGLE = 69
            return VTK_LAGRANGE_TRIANGLE 
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
