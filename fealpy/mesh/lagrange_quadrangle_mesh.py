from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import estr2dim,tensor_ldof,tensor_gdof
from .mesh_base import TensorMesh
from .quadrangle_mesh import QuadrangleMesh


class LagrangeQuadrangleMesh(TensorMesh):
    """A class for constructing and operating on p-order Lagrange quadrilateral meshes, 
    inheriting from TensorMesh.
    
    Parameters:
        node (TensorLike):  A tensor of nodal coordinates, shape (NN, GD).
    
        cell (TensorLike): A tensor defining the connectivity of quadrilateral cells, 
        with (p+1)^2 nodes per cell.
    
        p (int, optional): The interpolation order of the Lagrange element. 
        If None, it is inferred from the number of cell vertices.
        
        boundary (optional): A boundary projection operator applied only to boundary nodes.
    
        surface (optional): A surface projection operator applied to all mesh nodes.
    
    Attributes:
        p (int): Lagrange interpolation order.
        
        GD (int): Geometric dimension of the mesh (2 or 3).
        
        node (TensorLike): Coordinates of all mesh nodes.
        
        cell (TensorLike): Connectivity array of quadrilateral cells.
        
        localEdge (TensorLike): Local edge connectivity for each quadrilateral 
        in counter-clockwise order.
        
        meshtype (str): Mesh type identifier, fixed to 'lquad'.
        
    Methods:
        
    """
    def __init__(self, 
                 node: TensorLike, 
                 cell: TensorLike, 
                 p: Optional[int]=None, 
                 boundary=None, 
                 surface=None):
        
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)
        
        if p is None:
            NV = cell.shape[-1]
            self.p = int(-1 + bm.sqrt(NV))
        else:
            NV = (p+1) * (p+1)
            if cell.shape[-1] != NV:
                raise ValueError(f"cell.shape[-1] != {NV}, p = {p}.")
            else:
                self.p = p
        
        self.node = node     
        self.cell = cell   
        
        self.surface = surface
        self.boundary = boundary

        self.construct_local_edge(p)
        self.localFace = self.localEdge
        self.construct_local_cell(p)
        self.construct()
        
        if self.surface is not None:
            self.node, _ = self.surface.project(self.node)
            
        if self.boundary is not None:
            isBdNode = self.boundary_node_flag()
            self.node[isBdNode] = self.boundary.project(self.node[isBdNode])

        self.meshtype = 'lquad'
        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}

    def reference_cell_measure(self):
        return 1.0
    
    def construct_local_edge(self,  p: Optional[int]=None) -> None:
        """Generate the local edges for Lagrange elements of order p.
        
        Parameters:
            p(int, optional): the order of the Lagrange element. If p is None,
                it will be set to self.p.

        Returns:
            None: the local edges will be stored in self.localEdge.
        """
        p = self.p if p is None else p

        k = bm.arange((p + 1)**2, dtype=self.itype , device=self.device)
        k = k.reshape((p + 1, p + 1))  # (p+1, p+1), row: y direction, col: x direction
        
        # Extract the four edges of the quadrilateral in consistent order
        edge0 = k[0, :]                # bottom edge, left to right
        edge1 = k[:, -1]               # right edge, bottom to top
        edge2 = bm.flip(k[-1, :])      # top edge, right to left
        edge3 = bm.flip(k[:, 0])       # left edge, top to bottom
    
        self.localEdge = bm.stack([edge0, edge1, edge2, edge3], axis=0)

    def construct_local_cell(self, p: Optional[int]=None) -> None:
        """Generate the local cell for Lagrange elements of order p.
        
        Parameters:
            p(int, optional): the order of the Lagrange element. If p is None,
                it will be set to self.p.

        Returns:
            None: the local edges will be stored in self.localCell.
        """
        p = self.p if p is None else p
        self.localCell = None
        
    def number_of_corner_nodes(self) -> int:
        """ Get the number of corner nodes in the mesh.

        Returns:
            int: the number of corner nodes."""
        return self.cell_corner_node_flag().sum()
    
    def cell_corner_node_flag(self) -> TensorLike:
        """
        Get the flag array of the corner nodes in each cell.

        Returns:
            TensorLike: a boolean array of shape (NN, ), where NN is the number
            of nodes in the mesh. The value is True if the node is a cell corner
            node, and False otherwise.
        """

        NN = self.number_of_nodes()
        edge = self.entity('edge')

        isCornerNode = bm.zeros(NN, dtype=bm.bool)
        isCornerNode = bm.set_at(isCornerNode, edge[:, [0, -1]], True)
        return isCornerNode
    
    def interpolation_points(self,  p: Optional[int]=None):
        """Fetch all p-order interpolation points on the quadrilateral mesh."""
    
        node = self.entity('node')
        GD = self.geo_dimension()
        
        if p is None:
            return self.entity('node')
        elif p < 1:
            raise ValueError(f"p must be at least 1, but got {p}.")
        elif p == self.p:
            return node
        
        isCornerNode = self.cell_corner_node_flag()
        if p == 1:
            return node[isCornerNode]

        w = self.multi_index_matrix(p, 1, dtype=self.ftype)[1:-1, :] / p
        ipoints0 = self.bc_to_point((w,)).reshape(-1, GD)
        ipoints1 = bm.zeros((0, GD), dtype=self.ftype) 
        
        if p > 2:
            ipoints1 = self.bc_to_point((w, w)).reshape(-1, GD) 

        ipoints = bm.concatenate((node, ipoints0, ipoints1), axis=0)
        return ipoints
    
    def uniform_refine(self, n: int = 1):
        """Uniform refine the lagrange quadrangle mesh n times.

        Parameters:
            n (int): Times refine the  quadrangle mesh. Default is 1.

        Returns:
            mesh: The mesh obtained after uniformly refining n times.
            List(CSRTensor): The prolongation matrix from the finest to the the coarsest。
            
        Notes:
            子单元编号约定：
            1 | 3
            --+--
            0 | 2
            即 0: 左下, 1: 左上, 2: 右下, 3: 右上。
        """
        for _ in range(n):
            GD = self.geo_dimension()
            node = self.entity('node')        
            edge = self.entity('edge')       
            cell = self.entity('cell')        
            edge2cell = self.edge_to_cell()  

            ikwargs = bm.context(edge)   
            fkwargs = bm.context(node)   

            # corner nodes
            isCornerNode = bm.zeros(len(node), dtype=bm.bool)
            isCornerNode = bm.set_at(isCornerNode, edge[:, [0, -1]], True)

            nodes_parts = []
            edges_parts = []
            edge2cells_parts = []

            # corner nodes (first block)
            nodes_parts.append(node[isCornerNode])
            NN = len(nodes_parts[0])           # number of corner nodes
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            # each cell is split into 4 subcells
            cell2subcell = bm.arange(4 * NC, **ikwargs).reshape(NC, 4)

            # midpoints of each edge (one per edge)
            mid_pts = 0.5 * (node[edge[:, 0]] + node[edge[:, -1]])
            nodes_parts.append(mid_pts)

            start_mid = NN
            end_mid = start_mid + NE
            mid_idx = bm.arange(start_mid, end_mid, **ikwargs)  # global indices of midpoints

            # edge internal nodes (if p > 1), each edge has (p-1) internal point
            if self.p > 1:
                nn = self.p - 1
                # 参数 [1/p, 2/p, ..., (p-1)/p]
                t = bm.linspace(1/(self.p), (self.p-1)/self.p, nn, endpoint=True)
                # 对每条边，按比例插值
                edge_start = node[edge[:, 0]]
                edge_end   = node[edge[:, -1]]
                edge_internal_pts = bm.stack([
                    (1 - ti) * edge_start + ti * edge_end
                    for ti in t
                ], axis=1).reshape(-1, self.GD)
                nodes_parts.append(edge_internal_pts)

            # cell midpoints and interior points
            if self.p > 1:
                # 参数网格 (i/p, j/p), i,j = 1..p-1
                t = bm.linspace(1/(self.p), (self.p-1)/self.p, self.p-1, endpoint=True)
                t1, t2 = bm.meshgrid(t, t, indexing='ij')
                tt = bm.stack([t1.ravel(), t2.ravel()], axis=-1)  # ((p-1)^2, 2)

                # 四边形四个顶点
                v0 = node[cell[:, 0]]
                v1 = node[cell[:, 1]]
                v2 = node[cell[:, 2]]
                v3 = node[cell[:, 3]]

                # 双线性插值公式
                def bilinear_interp(xi, eta, v0, v1, v2, v3):
                    return ((1-xi)*(1-eta))[...,None]*v0 + xi*(1-eta)[...,None]*v1 \
                        + xi*eta[...,None]*v2 + (1-xi)*eta[...,None]*v3

                interior_pts = []
                for xi, eta in tt:
                    pts = bilinear_interp(xi, eta, v0, v1, v2, v3)
                    interior_pts.append(pts)
                interior_pts = bm.concatenate(interior_pts, axis=0).reshape(-1, self.GD)
                nodes_parts.append(interior_pts)

            # collect new nodes
            node_new = bm.concatenate(nodes_parts, axis=0)

            # edges
            if self.p == 1:
                e0 = bm.stack((edge[:, 0], mid_idx), axis=1)
                e1 = bm.stack((mid_idx, edge[:, -1]), axis=1)
            else:
                nn = self.p - 1  # 内部节点数
                start_idx = edge[:, [0]]               # 每条边起点
                end_idx   = edge[:, [-1]]              # 每条边终点
                eint_idx = bm.arange(end_mid, end_mid + nn*NE, **ikwargs).reshape(NE, nn)

                # 构造子边 (逐段拼接)
                sub_edges = []
                # 第一段：start -> 第一个内部点
                sub_edges.append(bm.stack((start_idx[:, 0], eint_idx[:, 0]), axis=1))
                # 中间段：内部点 -> 内部点
                for k in range(nn-1):
                    sub_edges.append(bm.stack((eint_idx[:, k], eint_idx[:, k+1]), axis=1))
                # 最后一段：最后一个内部点 -> end
                sub_edges.append(bm.stack((eint_idx[:, -1], end_idx[:, 0]), axis=1))

                # 合并
                e_all = bm.concatenate(sub_edges, axis=0)
                edges_parts.append(e_all)

            # edge2cell mapping
            e2c0 = bm.stack((
                cell2subcell[edge2cell[:, 0], 0],
                cell2subcell[edge2cell[:, 1], 0],
                bm.full(NE, 0, **ikwargs),
                bm.full(NE, 0, **ikwargs),
            ), axis=1)
            e2c1 = bm.stack((
                cell2subcell[edge2cell[:, 0], 1],
                cell2subcell[edge2cell[:, 1], 1],
                bm.full(NE, 1, **ikwargs),
                bm.full(NE, 1, **ikwargs),
            ), axis=1)
            edge2cells_parts.extend([e2c0, e2c1])

            # finalize global arrays
            edge_new = bm.concatenate(edges_parts, axis=0)
            edge2cell_new = bm.concatenate(edge2cells_parts, axis=0)

            # assemble cell2edge mapping for refined mesh: 4*NC cells each with 4 edges (indices into edge_new)
            cell2edge = bm.zeros((4 * NC, 4), **ikwargs)
            cell2edge = bm.set_at(cell2edge, (edge2cell_new[:, 0], edge2cell_new[:, 2]), range(len(edge_new)))
            cell2edge = bm.set_at(cell2edge, (edge2cell_new[:, 1], edge2cell_new[:, 3]), range(len(edge_new)))

            # commit new topology to mesh
            self.node = node_new
            self.edge = edge_new
            self.edge2cell = edge2cell_new
            self.cell2edge = cell2edge
            self.face = edge_new
            self.face2cell = edge2cell_new
            self.cell2face = cell2edge

            # 重新构造 cell
            self.construct_global_cell(cell2subcell, **ikwargs)

    def construct_global_cell(self, icell=None, **ikwargs):
        """Construct the global cell array after uniform refinement."""
        p = self.p
        NC = self.number_of_cells()
        ldof = (p+1)*(p+1)
        cell = bm.zeros((4*NC, ldof), **ikwargs)

        TD = self.top_dimension()
        mi = self.multi_index_matrix(p, TD)

        idx_left = bm.nonzero(mi[:,0]==0)[0]
        idx_right = bm.nonzero(mi[:,0]==p)[0]
        idx_bottom = bm.nonzero(mi[:,1]==0)[0]
        idx_top = bm.nonzero(mi[:,1]==p)[0]

        edge2cell = self.edge_to_cell()
        edge = self.entity('edge')
        
        # 定义每个边的局部索引
        sides = [(0, idx_left), (1, bm.flip(idx_bottom)), (2, idx_top), (3, bm.flip(idx_right))]

        # 填充边 DOF
        for side, idx in sides:
            flag = edge2cell[:, 2] == side
            if flag.any():
                for j, c in enumerate(idx):
                    # 每列逐个填充
                    bm.set_at(cell, (edge2cell[flag,0], c), edge[flag][:, j])

        # 处理共享边
        iflag = edge2cell[:, 0] != edge2cell[:, 1]
        for side, idx in sides:
            flag = iflag & (edge2cell[:, 3] == side)
            if flag.any():
                for j, c in enumerate(idx):
                    bm.set_at(cell, (edge2cell[flag,1], c), edge[flag][:, j])

        # 内部自由度 (p >= 3)
        if p >= 3 and icell is not None:
            internal_flag = (bm.sum((mi > 0) & (mi < p), axis=1) == 2)
            # icell shape should match number of cells x number of internal DOFs
            bm.set_at(cell, (..., internal_flag), icell)
            
        self.cell = cell

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
        """"Map coordinates from the reference element to the physical element.
        
        Parameters:
            bc(TensorLike): the barycentric coordinates of the integration points.
            index(Index): the index of the mesh entities.

        Returns:
            TensorLike: the coordinates of the integration points in the
            physical space.
        """
        node = self.node
        TD = len(bc) 
        entity = self.entity(TD, index=index) # 
        phi = self.shape_function(bc, p=self.p) # (NQ, NVC)
        p = bm.einsum('qn, cni -> cqi', phi, node[entity])
        return p

    # ipoints
    def number_of_local_ipoints(self, p:int, iptype:Union[int, str]='cell'):
        """The number of interpolation points on each lquad element.
        
        Parameters:
            p(int): the order of the Lagrange polynomial.
        """
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return tensor_ldof(p, iptype)

    def number_of_global_ipoints(self, p:int) -> int:
        """The total number of interpolation points on the lquad mesh.
        
        Parameters:
            p(int): the order of the Lagrange polynomial.
        """
        num = [self.entity(i).shape[0] for i in range(self.TD+1)]
        return tensor_gdof(p, num)

    def cell_to_ipoint(self, p:int, index:Index=_S):
        """
        Construct the map matrix from cells to interpolation points.

        Parameters:
            p(int): the order of the Lagrange polynomial.
            index(Index): the index of the mesh cells.

        Returns:
            TensorLike: the map matrix from cells to interpolation points.
            The shape is (NC, ldof), where NC is the number of cells and ldof is
            the number of local interpolation points.
        """
        cell = self.entity('cell')
        
        edge2cell = self.edge2cell
        if p == 0:
            return bm.arange(len(cell)).reshape((-1, 1))[index]
        if p == 1:
            return cell[index, [0, 3, 1, 2]]
        
        NN = self.number_of_corner_nodes()
        NC = self.number_of_cells()
        NE = self.number_of_edges()
        
        cell2ipoint = bm.zeros((NC, (p + 1) * (p + 1)), dtype=self.itype, device=bm.get_device(cell))
        c2p = cell2ipoint.reshape((NC, p + 1, p + 1))
        e2p = self.edge_to_ipoint(p)

        flag = edge2cell[:, 2] == 0
        c2p = bm.set_at(c2p, (edge2cell[flag, 0], slice(None), 0), e2p[flag])

        flag = edge2cell[:, 2] == 1
        c2p = bm.set_at(c2p, (edge2cell[flag, 0], -1, slice(None)), e2p[flag])

        flag = edge2cell[:, 2] == 2
        c2p = bm.set_at(c2p, (edge2cell[flag, 0], slice(None), -1), bm.flip(e2p[flag], axis=-1))

        flag = edge2cell[:, 2] == 3
        c2p = bm.set_at(c2p, (edge2cell[flag, 0], 0, slice(None)), bm.flip(e2p[flag], axis=-1))

        iflag = edge2cell[:, 0] != edge2cell[:, 1]
        flag = iflag & (edge2cell[:, 3] == 0)
        c2p = bm.set_at(c2p, (edge2cell[flag, 1], slice(None), 0), bm.flip(e2p[flag], axis=-1))

        flag = iflag & (edge2cell[:, 3] == 1)
        c2p = bm.set_at(c2p, (edge2cell[flag, 1], -1, slice(None)), bm.flip(e2p[flag], axis=-1))

        flag = iflag & (edge2cell[:, 3] == 2)
        c2p = bm.set_at(c2p, (edge2cell[flag, 1], slice(None), -1), e2p[flag])

        flag = iflag & (edge2cell[:, 3] == 3)
        c2p = bm.set_at(c2p, (edge2cell[flag, 1], 0, slice(None)), e2p[flag])

        c2p = bm.set_at(c2p, (slice(None), slice(1, -1), slice(1, -1)), NN + NE * (p - 1) +
                        bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p - 1, p - 1))

        return cell2ipoint[index]
    
    def edge_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        """Get the relationship between edges and integration points.
        
        Parameters:
            p(int): the order of the Lagrange polynomial.
             index(Index): the index of the mesh edge.
        """
        NN = self.number_of_corner_nodes()
        NE = self.number_of_edges()
        
        edges = self.edge[index]
        ikwargs = bm.context(edges)
        indices = bm.arange(NE, **ikwargs)[index]
        return bm.concatenate([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + bm.arange(0, p-1, **ikwargs) + NN,
            edges[:, 1].reshape(-1, 1),
        ], axis=-1)
    
    def face_to_ipoint(self, p: int, index: Index=_S):
        return self.edge_to_ipoint(p, index)

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

    def cell_area(self, q=None, index: Index=_S):
        """Calculate the area of a cell."""
        p = self.p
        q = p if q is None else q

        qf = self.quadrature_formula(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        G = self.first_fundamental_form(J)
        d = bm.sqrt(bm.linalg.det(G))
        a = bm.einsum('q, cq -> c', ws, d)
        return a

    def edge_length(self, q=None, index: Index=_S):
        """Calculate the length of the side."""
        p = self.p
        q = p if q is None else q
        qf = self.quadrature_formula(q, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        J = self.jacobi_matrix(bcs, index=index)
        l = bm.sqrt(bm.sum(J**2, axis=(-1, -2)))
        a = bm.einsum('q, cq -> c', ws, l)
        return a

    def cell_unit_normal(self, bc: TensorLike, index: Index=_S):
        """When calculating the surface,the direction of the unit normal at the integration point.
        
        Parameters:
            bc(TensorLike): the barycentric coordinates of the integration
            points.
        """
        J = self.jacobi_matrix(bc, index=index)
        n = bm.cross(J[..., 0], J[..., 1], axis=-1)
        if self.GD == 3:
            l = bm.sqrt(bm.sum(n**2, axis=-1, keepdims=True))
            n /= l
        return n

    def jacobi_matrix(self, bc: tuple, index: Index=_S, return_grad=False):
        """Compute the Jacobian matrix of the mapping from the reference element (xi, eta) to the physical Lagrange quadrilateral (x).
        
        Parameters:
            bc(TensorLike): the barycentric coordinates of the integration
            points.
            index(Index): the index of the mesh entities.

        Notes:
            x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        TD = len(bc)
        entity = self.entity(TD, index)
        gphi = self.grad_shape_function(bc, p = self.p)
        J = bm.einsum('cim, qin -> cqmn',
                self.node[entity[index], :], gphi) #(NC,ldof,GD),(NQ,ldof,TD)
        if return_grad is False:
            return J #(NC,NQ,GD,TD)
        else:
            return J, gphi

    # fundamental form
    def first_fundamental_form(self, J: TensorLike, index: Index=_S):
        """Compute the first fundamental form of a mesh surface at integration points."""
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
        """Numerically integrate a function over the mesh."""
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
        """Calculate the error between two functions."""
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
        """Return the corresponding VTK type of the mesh cell."""
        if etype in {'cell', 2}:
            VTK_LAGRANGE_QUADRILATERAL = 70 
            return VTK_LAGRANGE_QUADRILATERAL
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE


    def to_vtk(self, etype='cell', index: Index=_S, fname=None):
        """Convert the mesh to VTK format."""
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
            
    @classmethod
    def from_box(cls, box, p: int, nx=2, ny=2):
        """Construct a higher-order quadrilateral mesh from a rectangular box.
        
        Parameters:
            box (Sequence[float]): A sequence of 4 floats defining the box as [xmin, xmax, ymin, ymax].
            p (int): The order of the Lagrange polynomial.
            nx (int): Number of divisions along the x-axis.
            ny (int): Number of divisions along the y-axis.
        """
        mesh = QuadrangleMesh.from_box(box, nx, ny)
        return cls.from_quadrangle_mesh(mesh, p)
    
    @classmethod
    def from_quadrangle_mesh(cls, mesh, p: int, boundary=None, surface=None):
        """Construct a higher-order quadrilateral mesh from a linear quadrilateral mesh.
        
        Parameters:
            mesh (QuadrangleMesh): A linear quadrilateral mesh.
            p (int): The order of the Lagrange polynomial.
            boundary (optional): A boundary projection operator applied only to boundary nodes.
            surface (optional): A surface projection operator applied to all mesh nodes.
        """
        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        
        mesh = cls(node, cell, p=p, boundary=boundary, surface=surface)

        return mesh