import numpy as np 
from typing import Union, Optional, Sequence, Tuple, Any

from .utils import entitymethod, estr2dim

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S, Union, Tuple
from .. import logger

from .mesh_base import StructuredMesh, TensorMesh
from .plot import Plotable

class UniformMesh2d(StructuredMesh, TensorMesh, Plotable):
    """
    Topological data structure of a structured quadrilateral mesh

    The ordering of the nodes in each element is as follows:

    1 ------- 3
    |         |
    |         |
    |         |
    0 ------- 2

    The ordering of the edges in each element is as follows:

     ----1---- 
    |         |
    2         3
    |         |
     ----0---- 

    The ordering of entities in the entire mesh is as follows:

    * Node numbering rule: first in the y direction, then in the x direction
    * Edge numbering rule: first in the y direction, then in the x direction
    * Cell numbering rule: first in the y direction, then in the x direction

    Example for a 2x2 mesh:

    Nodes:
    2 ------- 5 ------- 8
    |         |         |
    |         |         |
    |         |         |
    1 ------- 4 ------- 7
    |         |         |
    |         |         |
    |         |         |
    0 ------- 3 ------- 6

    Edges:
     ---2--- ---5--- 
    |       |       |
    7       9       11
    |       |       |
     ---1--- ---4---
    |       |       |
    6       8       10
    |       |       |
     ---0--- ---3--- 

    Cells:
      -------   -------  
    |    1    |    3    |
      -------   -------  
    |    0    |    2    |
      -------   -------  

    """

    def __init__(self, extent = (0, 1, 0, 1), h = (1.0, 1.0), origin = (0.0, 0.0), 
                itype=None, ftype=None):
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64
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

        # Specify the counterclockwise drawing
        self.ccw = bm.array([0, 2, 3, 1], dtype=self.itype)


    # 实体生成方法
    @entitymethod(0)
    def _get_node(self) -> TensorLike:
        """
        @berif Generate the nodes in a structured mesh.
        """
        GD = 2
        nx = self.nx
        ny = self.ny

        box = [self.origin[0], self.origin[0] + nx * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]

        x = bm.linspace(box[0], box[1], nx + 1, dtype=self.ftype)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=self.ftype)
        xx, yy = bm.meshgrid(x, y, indexing='ij')
        node = bm.concatenate((xx[..., None], yy[..., None]), axis=-1)

        return node.reshape(-1, GD)
    
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
    
    # 实体拓扑
    def number_of_nodes_of_cells(self):
        return 4

    def number_of_edges_of_cells(self):
        return 4

    def number_of_faces_of_cells(self):
        return 4
    
    
    # 实体几何
    def entity_measure(self, etype: Union[int, str], index: Index = _S) -> TensorLike:
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
        GD = self.geo_dimension()
        nx = self.nx
        ny = self.ny
        box = [self.origin[0] + self.h[0] / 2, self.origin[0] + self.h[0] / 2 + (nx - 1) * self.h[0],
               self.origin[1] + self.h[1] / 2, self.origin[1] + self.h[1] / 2 + (ny - 1) * self.h[1]]
        x = bm.linspace(box[0], box[1], nx)
        y = bm.linspace(box[2], box[3], ny)
        X, Y = bm.meshgrid(x, y, indexing='ij')
        bc = bm.zeros((nx, ny, GD), dtype=self.ftype)
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
        GD = self.geo_dimension()
        box = [self.origin[0] + self.h[0] / 2, self.origin[0] + self.h[0] / 2 + (nx - 1) * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]
        x = bm.linspace(box[0], box[1], nx)
        y = bm.linspace(box[2], box[3], ny + 1)
        X, Y = bm.meshgrid(x, y, indexing='ij') 
        bc = bm.zeros((nx, ny + 1, GD), dtype=self.ftype)
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
    
    def bc_to_point(self, bcs: Union[Tuple, TensorLike], index=_S):
        """
        @brief Transform the barycentric coordinates of integration points
        to Cartesian coordinates on the actual mesh entities.

        Returns
            TensorLike: (NQ, NC, GD) or (NQ, NE, GD)
        """
        node = self.entity('node')
        if isinstance(bcs, tuple):
            assert len(bcs) == 2
            cell = self.entity('cell', index)

            bcs0 = bcs[0].reshape(-1, 2)
            bcs1 = bcs[1].reshape(-1, 2)
            bcs = bm.einsum('im, jn -> ijmn', bcs0, bcs1).reshape(-1, 4)

            p = bm.einsum('...j, cjk -> ...ck', bcs, node[cell[:]])
        else:
            edge = self.entity('edge', index=index)
            p = bm.einsum('...j, ejk -> ...ek', bcs, node[edge]) 

        return p    
    

    # 插值点
    def interpolation_points(self, p: int) -> TensorLike:
        '''
        @brief Generate all interpolation points of the mesh

        Ordering of 1st order interpolation points:
        2 ------- 5 ------- 8
        |         |         |
        |         |         |
        |         |         |
        1 ------- 4 ------- 7
        |         |         |
        |         |         |
        |         |         |
        0 ------- 3 ------- 6
        Ordering of 2nd order interpolation points:
        2 --11--- 5 --14----8
        |         |         |
        16        18        20
        |         |         |
        1 --10--- 4 --13----7
        |         |         |
        15        17        19
        |         |         |
        0 ---9--- 3 --12--- 6
        '''
        cell = self.entity('cell')
        edge = self.entity('edge')
        node = self.entity('node')

        GD = self.geo_dimension()
        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")
        if p == 1:
            return node

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            NN = self.number_of_nodes()
            gdof = self.number_of_global_ipoints(p)
            ipoints = bm.zeros((gdof, GD), dtype=self.ftype)
            ipoints[:NN, :] = node

            NE = self.number_of_edges()
            multiIndex = self.multi_index_matrix(p, 1, dtype=self.ftype)
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
            multiIndex = self.multi_index_matrix(p, 1, dtype=self.ftype)
            w = multiIndex[1:-1, :] / p
            ipoints = ipoints.at[NN:NN + (p-1) * NE, :].set(
                bm.einsum('ij, ...jm -> ...im', w, node[edge, :]).reshape(-1, GD))

            w = bm.einsum('im, jn -> ijmn', w, w).reshape(-1, 4)
            ipoints = ipoints.at[NN + (p-1) * NE:, :].set(
                bm.einsum('ij, kj... -> ki...', w, node[cell[:]]).reshape(-1, GD))
            
            return ipoints
        else:
            raise NotImplementedError("Backend is not yet implemented.")
        
    def cell_to_ipoint(self, p: int, index: Index=_S):
        """
        @brief Get the correspondence between mesh cells and interpolation points.

        The correspondence between cells and first-order interpolation points is as follows:
        2 ------- 5 ------- 8
        |         |         |
        |         |         |
        |         |         |
        1 ------- 4 ------- 7
        |         |         |
        |         |         |
        |         |         |
        0 ------- 3 ------- 6
        The correspondence between cells and second-order interpolation points is as follows:
        2 ---11-- 5 ---14-- 8
        |         |         |
        16   22   18   24   20
        |         |         |
        1 ---10-- 4 ---13-- 7
        |         |         |
        15   21   17   23   19
        |         |         |
        0 ---9--- 3 ---12-- 6
        """

        cell = self.entity('cell')

        if p == 1:
            return cell[index]

        edge2cell = self.edge_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        cell2ipoint = bm.zeros((NC, (p + 1) * (p + 1)), dtype=self.itype)
        c2p= cell2ipoint.reshape((NC, p + 1, p + 1))

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            e2p = self.edge_to_ipoint(p)

            # 确定哪些边在它的左边单元内是局部的第 0 号边
            flag = edge2cell[:, 2] == 0
            # 将这些边放在每个单元的第 0 列上
            c2p[edge2cell[flag, 0], :, 0] = e2p[flag]

            # 确定哪些边在它的左边单元内是局部的第 1 号边
            flag = edge2cell[:, 2] == 1
            # 将这些边放在每个单元的最后 1 列上，注意此时是逆序
            c2p[edge2cell[flag, 0], :, -1] = e2p[flag, -1::-1]

            # 确定哪些边在它的左边单元内是局部的第 2 号边
            flag = edge2cell[:, 2] == 2
            # 将这些边放在每个单元的第 0 行上
            c2p[edge2cell[flag, 0], 0, :] = e2p[flag, -1::-1]

            # 确定哪些边在它的左边单元内是局部的第 3 号边
            flag = edge2cell[:, 2] == 3
            # 将这些边放在每个单元的最后 1 行上，注意此时是逆序
            c2p[edge2cell[flag, 0], -1, :] = e2p[flag]

            # 确定哪些边是内部边
            iflag = edge2cell[:, 0] != edge2cell[:, 1]

            # 确定哪些边在它的右边单元内是局部的第 0 号边
            rflag = edge2cell[:, 3] == 0
            flag = iflag & rflag
            # 将这些边放在每个单元的第 1 列上，注意此时是逆序
            c2p[edge2cell[flag, 1], :, 0] = e2p[flag, -1::-1]

            # 确定哪些边在它的右边单元内是局部的第 1 号边
            rflag = edge2cell[:, 3] == 1
            flag = iflag & rflag
            # 将这些边放在每个单元的最后 1 列上
            c2p[edge2cell[flag, 1], :, -1] = e2p[flag]

            # 确定哪些边在它的右边单元内是局部的第 2 号边
            rflag = edge2cell[:, 3] == 2
            flag = iflag & rflag
            # 将这些边放在每个单元的第 1 行上
            c2p[edge2cell[flag, 1], 0, :] = e2p[flag]

            # 确定哪些边在它的右边单元内是局部的第 3 号边
            rflag = edge2cell[:, 3] == 3
            flag = iflag & rflag
            # 将这些边放在每个单元的最后 1 行上，注意此时是逆序
            c2p[edge2cell[flag, 1], -1, :] = e2p[flag, -1::-1]

            c2p[:, 1:-1, 1:-1] = NN + NE * (p - 1) + \
                bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p-1, p-1)
            
            return cell2ipoint[index]
        elif bm.backend_name == 'jax':
            e2p = self.edge_to_ipoint(p)
            flag = edge2cell[:, 2] == 0
            c2p = c2p.at[edge2cell[flag, 0], :, 0].set(e2p[flag])
            flag = edge2cell[:, 2] == 1
            c2p = c2p.at[edge2cell[flag, 0], :, -1].set(e2p[flag, -1::-1])
            flag = edge2cell[:, 2] == 2
            c2p = c2p.at[edge2cell[flag, 0], 0, :].set(e2p[flag, -1::-1])
            flag = edge2cell[:, 2] == 3
            c2p = c2p.at[edge2cell[flag, 0], -1, :].set(e2p[flag])

            iflag = edge2cell[:, 0] != edge2cell[:, 1]
            flag = iflag & (edge2cell[:, 3] == 0)
            c2p = c2p.at[edge2cell[flag, 1], :, 0].set(e2p[flag, -1::-1])
            flag = iflag & (edge2cell[:, 3] == 1)
            c2p = c2p.at[edge2cell[flag, 1], :, -1].set(e2p[flag])
            flag = iflag & (edge2cell[:, 3] == 2)
            c2p = c2p.at[edge2cell[flag, 1], 0, :].set(e2p[flag])
            flag = iflag & (edge2cell[:, 3] == 3)
            c2p = c2p.at[edge2cell[flag, 1], -1, :].set(e2p[flag, -1::-1])

            c2p = c2p.at[:, 1:-1, 1:-1].set(NN + NE * (p - 1) + \
                bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p-1, p-1))
            
            return cell2ipoint[index]
        else:
            raise NotImplementedError("Backend is not yet implemented.")

    
    def cell_to_ipoints(self, p:int, index: Index = _S):
        """
        @brief 获取单元上的双 p 次插值点
        """

        cell = self.entity('cell')

        if p == 0:
            return bm.arange(len(cell)).reshape((-1, 1))[index]

        if p == 1:
            return cell[index]  # 先排 y 方向，再排 x 方向

        edge2cell = self.edge_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        if bm.backend_name in ["numpy", "pytorch"]:
            cell2ipoint = bm.zeros((NC, (p + 1) * (p + 1)), dtype=self.itype)
            c2p = cell2ipoint.reshape((NC, p + 1, p + 1))

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

            c2p[:, 1:-1, 1:-1] = NN + NE * (p - 1) + bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p - 1, p - 1)
        elif bm.backend_name == "jax":
            raise NotImplementedError
        else:
            raise ValueError("Unsupported backend")
        return cell2ipoint[index]
        
    
    # 形函数
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
       

    # def shape_function(self, bcs: TensorLike, p: int=1, 
    #                 mi: Optional[TensorLike]=None) -> TensorLike:
    #     """
    #     @brief Compute the shape function of a 2D structured mesh.

    #     Returns:
    #         TensorLike: Shape function with shape (NQ, ldof).
    #     """
    #     assert isinstance(bcs, tuple)

    #     TD = bcs[0].shape[-1] - 1
    #     if mi is None:
    #         mi = bm.multi_index_matrix(p, TD, dtype=self.itype)
    #     phi = [bm.simplex_shape_function(val, p, mi) for val in bcs]
    #     ldof = self.number_of_local_ipoints(p, etype=2)

        return bm.einsum('im, jn -> ijmn', phi[0], phi[1]).reshape(-1, ldof)
    
    # def grad_shape_function(self, bcs: Tuple[TensorLike], p: int=1, index: Index=_S, 
    #                     variables: str='x') -> TensorLike:
    #     '''
    #     @brief Calculate the gradient of shape functions on a 2D structured grid.

    #     @note Compute the gradient of the shape functions with respect to the reference element variable u = (xi, eta)
    #     or the actual variable x.

    #     Returns:
    #     gphi : TensorLike
    #     The shape of gphi depends on the 'variables' parameter:
    #     - If variables == 'u': gphi has shape (NQ, ldof, GD).
    #     - If variables == 'x': gphi has shape (NQ, NCN, ldof, GD).
    #     '''
    #     assert isinstance(bcs, tuple)

    #     Dlambda = bm.array([-1, 1], dtype=self.ftype)

    #     phi0 = bm.simplex_shape_function(bcs[0], p=p)
    #     R0 = bm.simplex_grad_shape_function(bcs[0], p=p)
    #     gphi0 = bm.einsum('...ij, j -> ...i', R0, Dlambda)

    #     phi1 = bm.simplex_shape_function(bcs[1], p=p)
    #     R1 = bm.simplex_grad_shape_function(bcs[1], p=p)
    #     gphi1 = bm.einsum('...ij, j -> ...i', R1, Dlambda)

    #     n = phi0.shape[0] * phi1.shape[0]
    #     ldof = self.number_of_local_ipoints(p, etype=2)

    #     shape = (n, ldof, 2)
    #     gphi = bm.zeros(shape, dtype=self.ftype)

    #     # TODO: Provide a unified implementation that is not backend-specific
    #     if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
    #         gphi[..., 0] = bm.einsum('im, kn -> ikmn', gphi0, phi1).reshape(-1, ldof)
    #         gphi[..., 1] = bm.einsum('im, kn -> ikmn', phi0, gphi1).reshape(-1, ldof)
    #     elif bm.backend_name == 'jax':
    #         gphi = gphi.at[..., 0].set(bm.einsum('im, kn -> ikmn', gphi0, phi1).reshape(-1, ldof))
    #         gphi = gphi.at[..., 1].set(bm.einsum('im, kn -> ikmn', phi0, gphi1).reshape(-1, ldof))
    #     else:
    #         raise NotImplementedError("Backend is not yet implemented.")

    #     if variables == 'u':
    #         return gphi
    #     elif variables == 'x':
    #         J = self.jacobi_matrix(bcs, index=index)
    #         G = self.first_fundamental_form(J)
    #         G = bm.linalg.inv(G)
    #         gphi = bm.einsum('...ikm, ...imn, ...ln -> ...ilk', J, G, gphi)

    #         return gphi
        
    def edge_to_cell(self) -> TensorLike:
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

        
    def boundary_node_flag(self):
        """
        @brief Determine if a point is a boundary point.
        """
        NN = self.NN
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdPoint = bm.zeros((NN,), dtype=bm.bool)
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
        isBdCell = bm.zeros((NC,), dtype=bool)
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
    

    # 其他方法
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
        
UniformMesh2d.set_ploter('2d')



