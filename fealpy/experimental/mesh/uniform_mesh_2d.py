import numpy as np 
from typing import Union, Optional, Sequence, Tuple, Any

from .utils import entitymethod, estr2dim

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S, Union, Tuple
from .. import logger

from .mesh_base import StructuredMesh

class UniformMesh2d(StructuredMesh):
    """
    Topological data structure of a structured quadrilateral mesh

    The ordering of the nodes in each element is as follows:

    1 ------- 3
    |         |
    |         |
    |         |
    0 ------- 2

    The ordering of entities in the entire mesh is as follows:

    * Node numbering rule: first in the y direction, then in the x direction
    * Edge numbering rule: first in the y direction, then in the x direction
    * Cell numbering rule: first in the y direction, then in the x direction

    """
    def __init__(self, extent = (0, 1, 0, 1), h = (1.0, 1.0), origin = (0.0, 0.0), 
                itype=None, ftype=None):
        
        super().__init__(TD=2, itype=itype, ftype=ftype)

        # Mesh properties
        self.extent = [int(e) for e in extent]
        self.h = [float(val) for val in h]
        self.origin = [float(o) for o in origin]

        # Mesh dimensions
        self.nx = self.extent[1] - self.extent[0]
        self.ny = self.extent[3] - self.extent[2]
        self.NN = (self.nx + 1) * (self.ny + 1)
        self.NE = self.ny * (self.nx + 1) + self.nx * (self.ny + 1)
        self.NF = self.NE
        self.NC = self.nx * self.ny

        # Mesh datas
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}

        self.meshtype = 'UniformMesh2d'


    @entitymethod(0)
    def _get_node(self) -> TensorLike:
        """
        @berif Generate the nodes in a structured mesh.
        """
        nx = self.nx
        ny = self.ny

        box = [self.origin[0], self.origin[0] + nx * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]

        x = bm.linspace(box[0], box[1], nx + 1, dtype=self.ftype)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=self.ftype)
        xx, yy = bm.meshgrid(x, y, indexing='ij')
        node = bm.concatenate((xx[..., None], yy[..., None]), axis=-1)

        return node
    
    @entitymethod(1)
    def _get_edge(self) -> TensorLike:
        """
        @berif Generate the edges in a structured mesh.
        """
        nx = self.nx
        ny = self.ny
        NN = self.NN
        NE = self.NE

        idx = bm.arange(NN, dtype=self.itype).reshape(nx + 1, ny + 1)
        edge = bm.zeros((NE, 2), dtype=self.itype)
        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            NE0 = 0
            NE1 = nx * (ny + 1)
            edge[NE0:NE1, 0] = idx[:-1, :].reshape(-1)
            edge[NE0:NE1, 1] = idx[1:, :].reshape(-1)
            edge[NE0 + ny:NE1:ny + 1, :] = bm.flip(edge[NE0 + ny:NE1:ny + 1], axis=[1])

            NE0 = NE1
            NE1 += ny * (nx + 1)
            edge[NE0:NE1, 0] = idx[:, :-1].reshape(-1)
            edge[NE0:NE1, 1] = idx[:, 1:].reshape(-1)
            edge[NE0:NE0 + ny, :] = bm.flip(edge[NE0:NE0 + ny], axis=[1])

            return edge
        elif bm.backend_name == 'jax':
            NE0 = 0
            NE1 = nx * (ny + 1)
            edge = edge.at[NE0:NE1, 0].set(idx[:-1, :].reshape(-1))
            edge = edge.at[NE0:NE1, 1].set(idx[1:, :].reshape(-1))
            edge = edge.at[NE0 + ny:NE1:ny + 1, :].set(bm.flip(edge[NE0 + ny:NE1:ny + 1], axis=1))

            NE0 = NE1
            NE1 += ny * (nx + 1)
            edge = edge.at[NE0:NE1, 0].set(idx[:, :-1].reshape(-1))
            edge = edge.at[NE0:NE1, 1].set(idx[:, 1:].reshape(-1))
            edge = edge.at[NE0:NE0 + ny, :].set(bm.flip(edge[NE0:NE0 + ny], axis=1))

            return edge
        else:
            raise NotImplementedError("Backend is not yet implemented.")
    
    @entitymethod(2)
    def _get_cell(self) -> TensorLike:
        """
        @berif Generate the cells in a structured mesh.
        """
        nx = self.nx
        ny = self.ny

        NN = self.NN
        NC = self.NC
        cell = bm.zeros((NC, 4), dtype=self.itype)
        idx = bm.arange(NN).reshape(nx + 1, ny + 1)
        c = idx[:-1, :-1]
        cell_0 = c.reshape(-1)
        cell_1 = cell_0 + 1
        cell_2 = cell_0 + ny + 1
        cell_3 = cell_2 + 1
        cell = bm.concatenate([cell_0[:, None], cell_1[:, None], cell_2[:, None], 
                            cell_3[:, None]], axis=-1)

        return cell
    
    def geo_dimension(self):
        return 2

    def top_dimension(self):
        return 2
    
    def number_of_nodes_of_cells(self):
        return 4

    def number_of_edges_of_cells(self):
        return 4

    def number_of_faces_of_cells(self):
        return 4

    
    def entity(self, etype: Union[int, str], index=_S) -> TensorLike:
        """
        @brief Get the entities of the specified type.
        """
        GD = 2
        if isinstance(etype, str):
           etype = estr2dim(self, etype)

        if etype == 2:
            return self.cell[index, ...]
        elif etype == 1:
            return self.edge[index, ...]
        elif etype == 0:
            return self.node.reshape(-1, GD)[index, ...]
        else:
            raise ValueError("`etype` is wrong!")
    
    def entity_measure(self, etype: Union[int, str]) -> TensorLike:
        """
        @brief Get the measure of the entities of the specified type.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)

        if etype == 0:
            return bm.tensor(0, dtype=self.ftype)
        elif etype == 1:
            return self.h[0], self.h[1]
        elif etype == 2:
            return self.h[0] * self.h[1]
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
       
    def cell_barycenter(self) -> TensorLike:
        '''
        @brief Calculate the barycenter coordinates of the cells.
        '''
        nx = self.nx
        ny = self.ny
        box = [self.origin[0] + self.h[0] / 2, self.origin[0] + self.h[0] / 2 + (nx - 1) * self.h[0],
               self.origin[1] + self.h[1] / 2, self.origin[1] + self.h[1] / 2 + (ny - 1) * self.h[1]]
        x = bm.linspace(box[0], box[1], nx)
        y = bm.linspace(box[2], box[3], ny)
        X, Y = bm.meshgrid(x, y, indexing='ij')
        bc = bm.zeros((nx, ny, 2), dtype=self.ftype)
        bc = bm.concatenate((X[..., None], Y[..., None]), axis=-1)

        return bc

    def edge_barycenter(self) -> Tuple:
        """
        @brief Calculate the coordinates range for the edge centers.
        """
        bcx = self.edgex_barycenter()
        bcy = self.edgey_barycenter()

        return bcx, bcy

    def edgex_barycenter(self) -> TensorLike:
        """
        @brief Calculate the coordinates range for the edge centers in the x-direction.
        """
        nx = self.nx
        ny = self.ny
        box = [self.origin[0] + self.h[0] / 2, self.origin[0] + self.h[0] / 2 + (nx - 1) * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]
        x = bm.linspace(box[0], box[1], nx)
        y = bm.linspace(box[2], box[3], ny + 1)
        X, Y = bm.meshgrid(x, y, indexing='ij') 
        bc = bm.zeros((nx, ny + 1, 2), dtype=self.ftype)
        bc = bm.concatenate((X[..., None], Y[..., None]), axis=-1)

        return bc

    def edgey_barycenter(self) -> TensorLike:
        """
        @brief Calculate the coordinates range for the edge centers in the x-direction.
        """
        nx = self.nx
        ny = self.ny
        box = [self.origin[0] + self.h[0] / 2, self.origin[0] + self.h[0] / 2 + (nx - 1) * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]
        x = bm.linspace(box[0], box[1], nx + 1)
        y = bm.linspace(box[2], box[3], ny)
        X, Y = bm.meshgrid(x, y, indexing='ij') 
        bc = bm.zeros((nx, ny + 1, 2), dtype=self.ftype)
        bc = bm.concatenate((X[..., None], Y[..., None]), axis=-1)

        return bc

    def bc_to_point(self, bcs: Tuple, index=_S):
        """
        @brief Transform the barycentric coordinates of integration points
        to Cartesian coordinates on the actual mesh entities.

        Returns
            TensorLike: (NQ, NC, GD)
        """
        node = self.entity('node')

        assert len(bcs) == 2
        cell = self.entity('cell')[index]

        bcs0 = bcs[0].reshape(-1, 2)
        bcs1 = bcs[1].reshape(-1, 2)
        bcs = bm.einsum('im, jn -> ijmn', bcs0, bcs1).reshape(-1, 4)

        p = bm.einsum('...j, cjk -> ...ck', bcs, node[cell[:]])
        if p.shape[0] == 1:
            p = p.reshape(-1, 2)

        return p
    
    def quadrature_formula(self, q: int, etype:Union[int, str]='cell'):
        """
        @brief Get the quadrature formula for numerical integration.
        """
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

    def number_of_local_ipoints(self, p, etype:Union[int, str]='cell'):
        """
        @brief Get the number of local interpolation points on the mesh.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 2:
            return (p+1) * (p+1)
        elif etype == 1:
            return p + 1
        elif etype == 0:
            return 1
        
    def number_of_global_ipoints(self, p: int) -> int:
        """
        @brief Get the number of global interpolation points on the mesh.
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        return NN + (p-1)*NE + (p-1)*(p-1)*NC

    def interpolation_points(self, p):
        cell = self.cell
        edge = self.edge
        node = self.entity('node')

        GD = self.geo_dimension()
        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")
        if p == 1:
            return node.reshape(-1, GD)

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            NN = self.number_of_nodes()
            gdof = self.number_of_global_ipoints(p)
            ipoints = bm.zeros((gdof, GD), dtype=self.ftype)
            ipoints[:NN, :] = node

            NE = self.number_of_edges()
            multiIndex = self.multi_index_matrix(p, 1, dtype=node.dtype)
            w = multiIndex[1:-1, :] / p
            ipoints[NN:NN + (p-1) * NE, :] = bm.einsum('ij, ...jm -> ...im', w,
                    node[edge,:]).reshape(-1, GD)

            w = bm.einsum('im, jn -> ijmn', w, w).reshape(-1, 4)
            ipoints[NN + (p-1) * NE:, :] = bm.einsum('ij, kj... -> ki...', w,
                    node[cell[:]]).reshape(-1, GD)

            return ipoints
        elif bm.backend_name == 'jax':
            NN = self.number_of_nodes()
            gdof = self.number_of_global_ipoints(p)
            ipoints = bm.zeros((gdof, GD), dtype=self.ftype)
            ipoints = ipoints.at[:NN, :].set(node)

            NE = self.number_of_edges()
            multiIndex = self.multi_index_matrix(p, 1, dtype=node.dtype)
            w = multiIndex[1:-1, :] / p
            ipoints = ipoints.at[NN:NN + (p-1) * NE, :].set(bm.einsum('ij, ...jm -> ...im', 
                                                        w, node[edge, :]).reshape(-1, GD))

            w = bm.einsum('im, jn -> ijmn', w, w).reshape(-1, 4)
            ipoints = ipoints.at[NN + (p-1) * NE:, :].set(bm.einsum('ij, kj... -> ki...', 
                                                        w, node[cell[:]]).reshape(-1, GD))
            
            return ipoints
        else:
            raise NotImplementedError("Backend is not yet implemented.")
    
    def jacobi_matrix(self, bcs: TensorLike, index: Index=_S) -> TensorLike:
        """
        @brief Compute the Jacobi matrix for the mapping from the reference element (xi, eta) 
               to the actual Lagrange quadrilateral (x)

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        gphi = self.grad_shape_function(bcs, p=1, variables='u', index=index)
        J = bm.einsum( 'cim, ...in -> ...cmn', node[cell[:]], gphi)

        return J
    
    def first_fundamental_form(self, J: TensorLike) -> TensorLike:
        """
        @brief Compute the first fundamental form from the Jacobi matrix.
        """
        TD = J.shape[-1]

        shape = J.shape[0:-2] + (TD, TD)
        G = bm.zeros(shape, dtype=self.ftype)
        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            for i in range(TD):
                G[..., i, i] = bm.einsum('...d, ...d -> ...', J[..., i], J[..., i])
                for j in range(i+1, TD):
                    G[..., i, j] = bm.einsum('...d, ...d -> ...', J[..., i], J[..., j])
                    G[..., j, i] = G[..., i, j]
                    
            return G
        elif bm.backend_name == 'jax':
            for i in range(TD):
                G = G.at[..., i, i].set(bm.einsum('...d, ...d -> ...', 
                                                J[..., i], J[..., i]))
                for j in range(i + 1, TD):
                    G = G.at[..., i, j].set(bm.einsum('...d, ...d -> ...', 
                                                    J[..., i], J[..., j]))
                    G = G.at[..., j, i].set(G[..., i, j])
            return G
        else:
            raise NotImplementedError("Backend is not yet implemented.")
       
    def shape_function(self, bcs: TensorLike, p: int=1, 
                    mi: Optional[TensorLike]=None) -> TensorLike:
        """
        @brief Compute the shape function of a 2D structured mesh.

        Returns:
            TensorLike: Shape function with shape (NQ, ldof).
        """
        assert isinstance(bcs, tuple)

        TD = bcs[0].shape[-1] - 1
        if mi is None:
            mi = bm.multi_index_matrix(p, TD, dtype=self.itype)
        phi = [bm.simplex_shape_function(val, p, mi) for val in bcs]
        ldof = self.number_of_local_ipoints(p, etype=2)

        return bm.einsum('im, jn -> ijmn', phi[0], phi[1]).reshape(-1, ldof)
    
    def grad_shape_function(self, bcs: Tuple[TensorLike], p: int=1, index: Index=_S, 
                        variables: str='x') -> TensorLike:
        '''
        @brief Calculate the gradient of shape functions on a 2D structured grid.

        @note Compute the gradient of the shape functions with respect to the reference element variable u = (xi, eta)
        or the actual variable x.

        Returns:
        gphi : TensorLike
        The shape of gphi depends on the 'variables' parameter:
        - If variables == 'u': gphi has shape (NQ, ldof, GD).
        - If variables == 'x': gphi has shape (NQ, NCN, ldof, GD).
        '''
        assert isinstance(bcs, tuple)

        Dlambda = bm.array([-1, 1], dtype=self.ftype)

        phi0 = bm.simplex_shape_function(bcs[0], p=p)
        R0 = bm.simplex_grad_shape_function(bcs[0], p=p)
        gphi0 = bm.einsum('...ij, j -> ...i', R0, Dlambda)

        phi1 = bm.simplex_shape_function(bcs[1], p=p)
        R1 = bm.simplex_grad_shape_function(bcs[1], p=p)
        gphi1 = bm.einsum('...ij, j -> ...i', R1, Dlambda)

        n = phi0.shape[0] * phi1.shape[0]
        ldof = self.number_of_local_ipoints(p, etype=2)

        shape = (n, ldof, 2)
        gphi = bm.zeros(shape, dtype=self.ftype)

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            gphi[..., 0] = bm.einsum('im, kn -> ikmn', gphi0, phi1).reshape(-1, ldof)
            gphi[..., 1] = bm.einsum('im, kn -> ikmn', phi0, gphi1).reshape(-1, ldof)
        elif bm.backend_name == 'jax':
            gphi = gphi.at[..., 0].set(bm.einsum('im, kn -> ikmn', gphi0, phi1).reshape(-1, ldof))
            gphi = gphi.at[..., 1].set(bm.einsum('im, kn -> ikmn', phi0, gphi1).reshape(-1, ldof))
        else:
            raise NotImplementedError("Backend is not yet implemented.")

        if variables == 'u':
            return gphi
        elif variables == 'x':
            J = self.jacobi_matrix(bcs, index=index)
            G = self.first_fundamental_form(J)
            G = bm.linalg.inv(G)
            gphi = bm.einsum('...ikm, ...imn, ...ln -> ...ilk', J, G, gphi)

            return gphi
        
    def edge_to_cell(self):
        """
        @brief Adjacency relationship between edges and cells, 
        storing information about the two cells adjacent to each edge.
        """

        nx = self.nx
        ny = self.ny
        NC = self.NC
        NE = self.NE

        edge2cell = bm.zeros((NE, 4), dtype=self.itype)

        idx = bm.arange(NC, dtype=self.itype).reshape(nx, ny).T

        # x direction
        idx0 = bm.arange(nx * (ny + 1), dtype=self.itype).reshape(nx, ny + 1).T
        # y direction
        idx1 = bm.arange((nx + 1) * ny, dtype=self.itype).reshape(nx + 1, ny).T
        NE0 = nx * (ny + 1)

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            # left element
            edge2cell[idx0[:-1], 0] = idx
            edge2cell[idx0[:-1], 2] = 0
            edge2cell[idx0[-1], 0] = idx[-1]
            edge2cell[idx0[-1], 2] = 1

            # right element
            edge2cell[idx0[1:], 1] = idx
            edge2cell[idx0[1:], 3] = 1
            edge2cell[idx0[0], 1] = idx[0]
            edge2cell[idx0[0], 3] = 0

            # left element
            edge2cell[NE0 + idx1[:, 1:], 0] = idx
            edge2cell[NE0 + idx1[:, 1:], 2] = 3
            edge2cell[NE0 + idx1[:, 0], 0] = idx[:, 0]
            edge2cell[NE0 + idx1[:, 0], 2] = 2

            # right element
            edge2cell[NE0 + idx1[:, :-1], 1] = idx
            edge2cell[NE0 + idx1[:, :-1], 3] = 2
            edge2cell[NE0 + idx1[:, -1], 1] = idx[:, -1]
            edge2cell[NE0 + idx1[:, -1], 3] = 3

            return edge2cell
        elif bm.backend_name == 'jax':
            # left element
            edge2cell = edge2cell.at[idx0[:-1], 0].set(idx)
            edge2cell = edge2cell.at[idx0[:-1], 2].set(0)
            edge2cell = edge2cell.at[idx0[-1], 0].set(idx[-1])
            edge2cell = edge2cell.at[idx0[-1], 2].set(1)

            # right element
            edge2cell = edge2cell.at[idx0[1:], 1].set(idx)
            edge2cell = edge2cell.at[idx0[1:], 3].set(1)
            edge2cell = edge2cell.at[idx0[0], 1].set(idx[0])
            edge2cell = edge2cell.at[idx0[0], 3].set(0)

            # left element
            edge2cell = edge2cell.at[NE0 + idx1[:, 1:], 0].set(idx)
            edge2cell = edge2cell.at[NE0 + idx1[:, 1:], 2].set(3)
            edge2cell = edge2cell.at[NE0 + idx1[:, 0], 0].set(idx[:, 0])
            edge2cell = edge2cell.at[NE0 + idx1[:, 0], 2].set(2)

            # right element
            edge2cell = edge2cell.at[NE0 + idx1[:, :-1], 1].set(idx)
            edge2cell = edge2cell.at[NE0 + idx1[:, :-1], 3].set(2)
            edge2cell = edge2cell.at[NE0 + idx1[:, -1], 1].set(idx[:, -1])
            edge2cell = edge2cell.at[NE0 + idx1[:, -1], 3].set(3)

            return edge2cell
        else:
            raise NotImplementedError("Backend is not yet implemented.")
        
    # def cell_to_cell(self):
    #     """
    #     @brief Adjacency relationship between cells, storing the indices of neighboring cells
    #     for each cell.
    #     """
    #     NC = self.NC
    #     nx = self.nx
    #     ny = self.ny

    #     idx = bm.arange(NC).reshape(nx, ny)
    #     cell2cell = np.zeros((NC, 4), dtype=self.itype)

    #     # x direction
    #     NE0 = 0
    #     NE1 = ny
    #     NE2 = nx * ny
    #     cell2cell[NE0: NE1, 0] = idx[0, :].flatten()
    #     cell2cell[NE1: NE2, 0] = idx[:-1, :].flatten()
    #     cell2cell[NE0: NE2 - NE1, 1] = idx[1:, :].flatten()
    #     cell2cell[NE2 - NE1: NE2, 1] = idx[-1, :].flatten()

    #     # y direction
    #     idx0 = bm.arange(0, nx * ny, ny).reshape(nx, 1)
    #     idx0 = idx0.flatten()

    #     idx1 = idx0 + ny - 1
    #     idx1 = idx1.flatten()

    #     # TODO: Provide a unified implementation that is not backend-specific
    #     if bm.backend_name == 'numpy':
    #         cell2cell[idx0, 2] = idx0
    #         ii = np.setdiff1d(idx.flatten(), idx0)
    #         cell2cell[ii, 2] = ii - 1

    #         cell2cell[idx1, 3] = idx1
    #         ii = np.setdiff1d(idx.flatten(), idx1)
    #         cell2cell[ii, 3] = ii + 1

    #         return cell2cell

    #     elif bm.backend_name == 'pytorch':
    #         raise NotImplementedError("PyTorch is not yet implemented.")
    #     elif bm.backend_name == 'jax':
    #         raise NotImplementedError("Jax is not yet implemented.")
    #     else:
    #         raise NotImplementedError("Backend is not yet implemented.")

        
    def boundary_node_flag(self):
        """
        @brief Determine if a point is a boundary point.
        """
        NN = self.NN
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdPoint = bm.zeros((NN,), dtype=bm.bool_)
        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            isBdPoint[edge[isBdEdge, :]] = True
            return isBdPoint
        elif bm.backend_name == 'jax':
            isBdPoint = isBdPoint.at[edge[isBdEdge, :]].set(True)
            return isBdPoint
        else:
            raise NotImplementedError("Backend is not yet implemented.")

    
    def boundary_edge_flag(self):
        """
        @brief Determine if an edge is a boundary edge.
        """
        edge2cell = self.edge_to_cell()
        isBdEdge = edge2cell[:, 0] == edge2cell[:, 1]
        return isBdEdge
    
    
    def boundary_cell_flag(self):
        """
        @brief Determine if a cell is a boundary cell.
        """
        NC = self.NC

        edge2cell = self.edge_to_cell()
        isBdCell = bm.zeros((NC,), dtype=bm.bool_)
        isBdEdge = self.boundary_edge_flag()

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            isBdCell[edge2cell[isBdEdge, 0]] = True
            return isBdCell
        elif bm.backend_name == 'jax':
            isBdCell = isBdCell.at[edge2cell[isBdEdge, 0]].set(True)
            return isBdCell
        else:
            raise NotImplementedError("Backend is not yet implemented.")
    
    
    def uniform_refine(self, n: int=1):
        """
        @brief Uniformly refine the 2D structured mesh.

        Note:
        clear method is used at the end to clear the cache of entities. This is necessary because even after refinement, 
        the entities remain the same as before refinement due to the caching mechanism.
        Structured mesh have their own entity generation methods, so the cache needs to be manually cleared.
        Unstructured mesh do not require this because they do not have entity generation methods.
        """
        for i in range(n):
            self.extent = [i * 2 for i in self.extent]
            self.h = [h / 2.0 for h in self.h]
            self.nx = self.extent[1] - self.extent[0]
            self.ny = self.extent[3] - self.extent[2]

            self.NC = self.nx * self.ny
            self.NF = self.NE
            self.NE = self.ny * (self.nx + 1) + self.nx * (self.ny + 1)
            self.NN = (self.nx + 1) * (self.ny + 1)
        self.clear() 
        



