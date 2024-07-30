from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import SimplexMesh, estr2dim


class TriangleMesh(SimplexMesh):
    def __init__(self, node: TensorLike, cell: TensorLike) -> None:
        """
        """
        super().__init__(TD=2)
        kwargs = {'dtype': cell.dtype}
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
        node = self.node

        if isinstance(etype, str):
            etype = estr2dim(self, etype)

        if etype == 0:
            return bm.tensor([0,], dtype=self.ftype)
        elif etype == 1:
            edge = self.entity(1, index)
            return bm.edge_length(edge, node)
        elif etype == 2:
            cell = self.entity(2, index)
            return bm.simplex_measure(cell, node)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
  
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
        w = bm.zeros((p - 1, 2), **kwargs)
        w[:, 0] = bm.arange(p - 1, 0, -1, **kwargs) / p
        w[:, 1] = bm.flip(w[:, 0], axis=0) 
        ipoints_from_edge = bm.einsum('ij, ...jm->...im', w,
                                         node[edge, :]).reshape(-1, GD) # ipoints[NN:NN + (p - 1) * NE, :]
        ipoint_list.append(ipoints_from_edge)

        if p >= 3:
            TD = self.top_dimension()
            cell = self.entity('cell')
            multiIndex = self.multi_index_matrix(p, TD)
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:, 0] | isEdgeIPoints[:, 1] |
                                isEdgeIPoints[:, 2])
            multiIndex = multiIndex[isInCellIPoints, :]
            w = multiIndex.astype(self.ftype) / p

            ipoints_from_cell = bm.einsum('ij, kj...->ki...', w,
                                          node[cell, :]).reshape(-1, GD) # ipoints[NN + (p - 1) * NE:, :]
            ipoint_list.append(ipoints_from_cell)

        return np.concatenate(ipoint_list, axis=0)  # (gdof, GD)

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
        c2p[:, flag] = NN + NE*(p-1) + np.arange(NC*cdof, **kwargs).reshape(NC, cdof)
        return c2p[index]

    def face_to_ipoint(self, p: int, index: Index=_S):
        return self.edge_to_ipoint(p, index)

    def uniform_refine(self, n=1, surface=None, interface=None, returnim=False):
        """
        @brief 一致加密三角形网格
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
            self.construct()

    @classmethod
    def from_one_triangle(cls, meshtype='iso', ftype=bm.float64, itype=bm.int32):
        if meshtype == 'equ':
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3) / 2]], dtype=ftype)
        elif meshtype == 'iso':
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2]], dtype=itype)
        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_square_domain_with_fracture(cls, ftype=bm.float64, itype=bm.int32):
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
            [1.0, 1.0]], dtype=ftype)

        cell = bm.tensor([
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5]], dtype=itype)

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_square(cls, nx=10, ny=10, threshold=None, ftype=bm.float64, itype=bm.int32):
        """
        Generate a triangle mesh for a unit square.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TriangleMesh instance
        """
        return cls.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny, 
                threshold=threshold, ftype=ftype, itype=itype)

    ## @ingroup MeshGenerators
    @classmethod
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10, threshold=None, ftype=bm.float64, itype=bm.int32):
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
        x = bm.linspace(box[0], box[1], nx+1)
        y = bm.linspace(box[2], box[3], ny+1)
        X, Y = bm.meshgrid(x, y, indexing='ij')
    
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

        idx = bm.arange(NN).reshape(nx + 1, ny + 1)
        cell0 = bm.concatenate((
            idx[1:, 0:-1].reshape(-1, 1),
            idx[1:, 1:].reshape(-1, 1),
            idx[0:-1, 0:-1].reshape(-1, 1),
            ), axis=1)
        cell1 = bm.concatenate((
            idx[0:-1, 1:].reshape(-1, 1),
            idx[0:-1, 0:-1].reshape(-1, 1),
            idx[1:, 1:].reshape(-1, 1)
            ), axis=1)
        cell = bm.concatenate((cell0, cell1), axis=0)

        if threshold is not None:
            bc = bm.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = bm.zeros(NN, dtype=bm.bool_)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_torus_surface(cls, R, r, nu, nv, ftype=bm.float64, itype=bm.int32):
        """
        """
        NN = nu * nv
        NC = nu * nv
        node = bm.zeros((NN, 3), dtype=ftype)

        x = bm.linspace(0, 2*bm.pi, nu)
        y = bm.linspace(0, 2*bm.pi, nv)
        U, V = bm.meshgrid(x, y, indexing='ij')
        
        X = (R + r * bm.cos(V)) * bm.cos(U)
        Y = (R + r * bm.cos(V)) * bm.sin(U)
        Z = r * bm.sin(V)
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), axis=1)

        idx = bm.zeros((nu + 1, nv + 1), dtype=itype)
        idx[0:-1, 0:-1] = bm.arange(NN).reshape(nu, nv)
        idx[-1, :] = idx[0, :]
        idx[:, -1] = idx[:, 0]
        cell = bm.zeros((2 * NC, 3), dtype=itype)
        cell0 = bm.concatenate((
            idx[1:, 0:-1].reshape(-1, 1),
            idx[1:, 1:].reshape(-1, 1),
            idx[0:-1, 0:-1].reshape(-1, 1),
            ), axis=1)
        cell1 = bm.concatenate((
            idx[0:-1, 1:].reshape(-1, 1),
            idx[0:-1, 0:-1].reshape(-1, 1),
            idx[1:, 1:].reshape(-1, 1)
            ), axis=1)
        cell = bm.concatenate((cell0, cell1), axis=0)
        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_sphere_surface(cls, refine=0, ftype=bm.float64, itype=bm.int32):
        """
        @brief  Generate a triangular mesh on a unit sphere surface.
        @return the triangular mesh.
        """
        t = (bm.sqrt(5) - 1) / 2
        node = bm.tensor([
            [0, 1, t], [0, 1, -t], [1, t, 0], [1, -t, 0],
            [0, -1, -t], [0, -1, t], [t, 0, 1], [-t, 0, 1],
            [t, 0, -1], [-t, 0, -1], [-1, t, 0], [-1, -t, 0]], dtype=ftype)
        cell = bm.tensor([
            [6, 2, 0], [3, 2, 6], [5, 3, 6], [5, 6, 7],
            [6, 0, 7], [3, 8, 2], [2, 8, 1], [2, 1, 0],
            [0, 1, 10], [1, 9, 10], [8, 9, 1], [4, 8, 3],
            [4, 3, 5], [4, 5, 11], [7, 10, 11], [0, 10, 7],
            [4, 11, 9], [8, 4, 9], [5, 7, 11], [10, 9, 11]], dtype=itype)
        mesh = cls(node, cell)
        mesh.uniform_refine(refine)
        node = mesh.node
        cell = mesh.cell
        # project
        d = bm.sqrt(node[:, 0] ** 2 + node[:, 1] ** 2 + node[:, 2] ** 2) - 1
        l = bm.sqrt(bm.sum(node ** 2, axis=1))
        n = node / l[..., None]
        node = node - d[..., None] * n
        return cls(node, cell)

