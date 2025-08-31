from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import estr2dim,tensor_ldof,tensor_gdof
from .mesh_base import TensorMesh
from .quadrangle_mesh import QuadrangleMesh


class LagrangeQuadrangleMesh(TensorMesh):
    def __init__(self, node: TensorLike, cell: TensorLike, p=None, curve=None,
                 surface=None,construct=False):
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)

        kwargs = bm.context(cell)
        
        if p is None:
            NV = cell.shape[-1]
            self.p = int(-1 + bm.sqrt(NV))
        else:
            NV = (p+1) * (p+1)
            if cell.shape[-1] != NV:
                raise ValueError(f"cell.shape[-1] != {NV}, p = {p}.")
            else:
                self.p = p
        
        self.GD = node.shape[1]
        self.node = node
        self.cell = cell
        self.surface = surface

        self.construct_local_edge()
        self.localFace = self.localEdge
        self.construct_local_cell()
        
        if construct:
            self.construct()

        self.meshtype = 'lquad'
        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}

    def reference_cell_measure(self):
        return 1.0
    
    def construct_local_edge(self) -> None:
        """Generate the local edges for Lagrange elements of order p."""
        k = bm.arange((self.p + 1)**2, dtype=self.itype , device=self.device)
        k = k.reshape((self.p + 1, self.p + 1))  # (p+1, p+1), row: y direction, col: x direction
    
        # Extract the four edges of the quadrilateral in consistent order
        edge0 = k[0, :]                # bottom edge, left to right
        edge1 = k[:, -1]               # right edge, bottom to top
        edge2 = bm.flip(k[-1, :])      # top edge, right to left
        edge3 = bm.flip(k[:, 0])       # left edge, top to bottom
    
        self.localEdge = bm.stack([edge0, edge1, edge2, edge3], axis=0)
        
    def construct_local_cell(self):
        """Generate the local cell for Lagrange elements of order p."""
        self.localcell = None
    
    def interpolation_points(self, p: int):
        """Fetch all p-order interpolation points on the quadrilateral mesh."""
    
        if p < 1:
            raise ValueError(f"p must be at least 1, but got {p}.")
        
        node = self.entity('node')
        GD = self.geo_dimension()

        multiIndex = self.multi_index_matrix(p, 1, dtype=self.ftype)
        w = multiIndex[1:-1, :] / p
        ipoints0 = self.bc_to_point((w,)).reshape(-1, GD)
        ipoints1 = bm.zeros((0, GD), dtype=self.ftype) 
        
        if p >= 3:
            ipoints1 = self.bc_to_point((w, w)).reshape(-1, GD) 

        ipoints = bm.concatenate((node, ipoints0, ipoints1), axis=0)
        return ipoints
    
    @classmethod
    def from_box(cls, box, p: int, nx=2, ny=2):
        mesh = QuadrangleMesh.from_box(box, nx, ny)
        return cls.from_quadrangle_mesh(mesh, p)
    
    @classmethod
    def from_quadrangle_mesh(cls, mesh, p: int, surface=None):
        init_node = mesh.entity('node')
        
        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        if surface is not None:
            init_node[:],_ = surface.project(init_node)
            node,_ = surface.project(node)

        lmesh = cls(node, cell, p=p, construct=True)

        lmesh.edge2cell = mesh.edge2cell # (NF, 4)
        lmesh.cell2edge = mesh.cell2edge
        lmesh.edge  = mesh.edge_to_ipoint(p)
        return lmesh
    
    def uniform_refine(self, n: int = 1):
        pass
    
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
        """"Map coordinates from the reference element to the physical element."""
        node = self.node
        TD = len(bc) 
        entity = self.entity(TD, index=index) # 
        phi = self.shape_function(bc, p=self.p) # (NQ, NVC)
        p = bm.einsum('qn, cni -> cqi', phi, node[entity])
        return p

    # ipoints
    def number_of_local_ipoints(self, p:int, iptype:Union[int, str]='cell'):
        """The number of interpolation points on each lquad element."""
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return tensor_ldof(p, iptype)

    def number_of_global_ipoints(self, p:int) -> int:
        """The total number of interpolation points on the lquad mesh."""
        num = [self.entity(i).shape[0] for i in range(self.TD+1)]
        return tensor_gdof(p, num)

    def cell_to_ipoint(self, p:int, index:Index=_S):
        """Obtain the bi-p interpolation points on the element."""
        sp = self.p
        cell = self.cell[:, [0, -sp-1, 1]]  # 取角点
        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edge2cell = self.edge2cell
        if p == 0:
            return bm.arange(len(cell)).reshape((-1, 1))[index]
        if p == 1:
            return cell[index, [0, 3, 1, 2]]
        
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
        """Get the relationship between edges and integration points."""
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        
        edges = self.edge[index]
        kwargs = bm.context(edges)
        indices = bm.arange(NE, **kwargs)[index]
        return bm.concatenate([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + bm.arange(0, p-1, **kwargs) + NN,
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
        """When calculating the surface,the direction of the unit normal at the integration point."""
        J = self.jacobi_matrix(bc, index=index)
        n = bm.cross(J[..., 0], J[..., 1], axis=-1)
        if self.GD == 3:
            l = bm.sqrt(bm.sum(n**2, axis=-1, keepdims=True))
            n /= l
        return n

    def jacobi_matrix(self, bc: tuple, index: Index=_S, return_grad=False):
        """Compute the Jacobian matrix of the mapping from the reference element (xi, eta) to the physical Lagrange quadrilateral (x).
        
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