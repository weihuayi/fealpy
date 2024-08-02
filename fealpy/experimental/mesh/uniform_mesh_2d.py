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

        self.face_to_ipoint = self.edge_to_ipoint


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
        node = bm.concatenate((xx[..., np.newaxis], yy[..., np.newaxis]), axis=-1)

        return node
    
    @entitymethod(1)
    def _get_edge(self):
        """
        @berif Generate the edges in a structured mesh.
        """
        nx = self.nx
        ny = self.ny
        NN = self.NN
        NE = self.NE

        idx = bm.arange(NN, dtype=self.itype).reshape(nx + 1, ny + 1)
        edge = bm.zeros((NE, 2), dtype=self.itype)

        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            NE0 = 0
            NE1 = nx * (ny + 1)
            edge[NE0:NE1, 0] = idx[:-1, :].reshape(-1)
            edge[NE0:NE1, 1] = idx[1:, :].reshape(-1)
            edge[NE0 + ny:NE1:ny + 1, :] = bm.flip(edge[NE0 + ny:NE1:ny + 1], axis=[1])

            #edge[NE0 + ny:NE1:ny + 1, :] = edge[NE0 + ny:NE1:ny + 1, -1::-1]

            NE0 = NE1
            NE1 += ny * (nx + 1)
            edge[NE0:NE1, 0] = idx[:, :-1].reshape(-1)
            edge[NE0:NE1, 1] = idx[:, 1:].reshape(-1)
            edge[NE0:NE0 + ny, :] = bm.flip(edge[NE0:NE0 + ny], axis=[1])

            #edge[NE0:NE0 + ny, :] = edge[NE0:NE0 + ny, -1::-1]

            return edge
        elif bm.backend_name == 'jax':
            # TODO: Jax backend is not yet implemented.
            raise NotImplementedError("Jax backend is not yet implemented.")
        else:
            raise NotImplementedError("Backend is not yet implemented.")
    
    @entitymethod(2)
    def _get_cell(self):
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
        #cell[:, 0] = c.reshape(-1)
        #cell[:, 1] = cell[:, 0] + 1
        #cell[:, 2] = cell[:, 0] + ny + 1
        #cell[:, 3] = cell[:, 2] + 1
        cell_0 = c.reshape(-1)
        cell_1 = cell_0 + 1
        cell_2 = cell_0 + ny + 1
        cell_3 = cell_2 + 1
        cell = bm.concatenate([cell_0[:, None], cell_1[:, None], cell_2[:, None], cell_3[:, None]], axis=-1)

        return cell
    
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
       
    # def cell_barycenter(self):
    #     GD = self.geo_dimension()
    #     nx = self.nx
    #     ny = self.ny
    #     box = [self.origin[0] + self.h[0] / 2, self.origin[0] + self.h[0] / 2 + (nx - 1) * self.h[0],
    #            self.origin[1] + self.h[1] / 2, self.origin[1] + self.h[1] / 2 + (ny - 1) * self.h[1]]
    #     bc = bm.zeros((nx, ny, 2), dtype=self.ftype)
    #     bc[..., 0], bc[..., 1] = np.mgrid[
    #                              box[0]:box[1]:nx * 1j,
    #                              box[2]:box[3]:ny * 1j]
    #     return bc

    # def edge_barycenter(self):
    #     """
    #     @brief
    #     """
    #     bcx = self.edgex_barycenter()
    #     bcy = self.edgey_barycenter()
    #     return bcx, bcy

    # ## @ingroup FDMInterface
    # def edgex_barycenter(self):
    #     """
    #     @brief
    #     """
    #     GD = self.geo_dimension()
    #     nx = self.nx
    #     ny = self.ny
    #     box = [self.origin[0] + self.h[0] / 2, self.origin[0] + self.h[0] / 2 + (nx - 1) * self.h[0],
    #            self.origin[1], self.origin[1] + ny * self.h[1]]
    #     bc = np.zeros((nx, ny + 1, 2), dtype=self.ftype)
    #     bc[..., 0], bc[..., 1] = np.mgrid[
    #                              box[0]:box[1]:nx * 1j,
    #                              box[2]:box[3]:(ny + 1) * 1j]
    #     return bc

    # ## @ingroup FDMInterface
    # def edgey_barycenter(self):
    #     """
    #     @breif
    #     """
    #     GD = self.geo_dimension()
    #     nx = self.nx
    #     ny = self.ny
    #     box = [self.origin[0], self.origin[0] + nx * self.h[0],
    #            self.origin[1] + self.h[1] / 2, self.origin[1] + self.h[1] / 2 + (ny - 1) * self.h[1]]
    #     bc = np.zeros((nx + 1, ny, 2), dtype=self.ftype)
    #     bc[..., 0], bc[..., 1] = np.mgrid[
    #                              box[0]:box[1]:(nx + 1) * 1j,
    #                              box[2]:box[3]:ny * 1j]
    #     return bc

    # def gradient(self, f, order=1):
    #     """
    #     @brief 求网格函数 f 的梯度
    #     """
    #     hx = self.h[0]
    #     hy = self.h[1]
    #     fx, fy = np.gradient(f, hx, hy, edge_order=order)
    #     return fx, fy

    # ## @ingroup FDMInterface
    # def divergence(self, f_x, f_y, order=1):
    #     """
    #     @brief 求向量网格函数 (fx, fy) 的散度
    #     """

    #     hx = self.h[0]
    #     hy = self.h[1]
    #     f_xx, f_xy = np.gradient(f_x, hx, edge_order=order)
    #     f_yx, f_yy = np.gradient(f_y, hy, edge_order=order)
    #     return f_xx + f_yy
    
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
        for i in range(TD):
            G[..., i, i] = bm.einsum('...d, ...d -> ...', J[..., i], J[..., i])
            for j in range(i+1, TD):
                G[..., i, j] = bm.einsum('...d, ...d -> ...', J[..., i], J[..., j])
                G[..., j, i] = G[..., i, j]
                
        return G
       
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
        #ldof = phi0.shape[-1] * phi1.shape[-1]
        ldof = self.number_of_local_ipoints(p, etype=2)

        shape = (n, ldof, 2)
        gphi = bm.zeros(shape, dtype=self.ftype)

        gphi[..., 0] = bm.einsum('im, kn -> ikmn', gphi0, phi1).reshape(-1, ldof)
        gphi[..., 1] = bm.einsum('im, kn -> ikmn', phi0, gphi1).reshape(-1, ldof)

        if variables == 'u':
            return gphi
        elif variables == 'x':
            J = self.jacobi_matrix(bcs, index=index)
            G = self.first_fundamental_form(J)
            G = bm.linalg.inv(G)
            gphi = bm.einsum('...ikm, ...imn, ...ln -> ...ilk', J, G, gphi)
            return gphi
    
    def uniform_refine(self, n=1):
        # TODO: There is a problem with this code
        for i in range(n):
            self.extent = [i * 2 for i in self.extent]
            self.h = [h / 2.0 for h in self.h]
            self.nx = self.extent[1] - self.extent[0]
            self.ny = self.extent[3] - self.extent[2]

            self.NC = self.nx * self.ny
            self.NF = self.NE
            self.NE = self.ny * (self.nx + 1) + self.nx * (self.ny + 1)
            self.NN = (self.nx + 1) * (self.ny + 1)
        
        del self.node
        del self.edge
        del self.cell
        # TODO: Implement cache clearing mechanism



