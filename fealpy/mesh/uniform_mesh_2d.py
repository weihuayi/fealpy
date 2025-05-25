from ..backend import backend_manager as bm  

from ..typing import TensorLike, Index, _S

from typing import Union, Optional, Callable, Tuple, List, Any
from types import ModuleType
from matplotlib.pyplot import Figure
from matplotlib.projections import Axes3D
from matplotlib.axes import Axes

from .utils import entitymethod, estr2dim

from .mesh_base import StructuredMesh, TensorMesh
from .plot import Plotable

from builtins import tuple, int , float

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

    def __init__(self, extent: Tuple[int, int, int, int] = (0, 1, 0, 1), 
                h: Tuple[float, float] = (1.0, 1.0), 
                origin: Tuple[float, float] = (0.0, 0.0), 
                ipoints_ordering='yx', 
                flip_direction=None, 
                *, itype=None, ftype=None, device=None):
        """
        Initializes a 2D uniform structured mesh.
        """
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64
        super().__init__(TD=2, itype=itype, ftype=ftype)

        self.device = device

        # self.extent = bm.array(extent, dtype=itype, device=device)
        self.extent = extent
        self.h = bm.array(h, dtype=ftype, device=device) 
        self.origin = bm.array(origin, dtype=ftype, device=device)
        self.shape = (
                self.extent[1] - self.extent[0], 
                self.extent[3] - self.extent[2]
                )
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

        # Interpolation points
        if ipoints_ordering not in ['yx', 'nec']:
            raise ValueError("The ipoints_ordering parameter must be either 'yx' or 'nec'")
        self.ipoints_ordering = ipoints_ordering

        # Whether to flip
        self.flip_direction = flip_direction

        # Initialize edge adjustment mask
        self.adjusted_edge_mask = self.get_adjusted_edge_mask()

        # Specify the counterclockwise drawing
        self.ccw = bm.array([0, 2, 3, 1], dtype=self.itype, device=self.device)

        self.edge2cell = self.edge_to_cell()
        self.face2cell = self.edge2cell
        self.cell2edge = self.cell_to_edge()

        self.localEdge = bm.array([(0, 2), (1, 3), 
                                   (0, 1), (2, 3)], dtype=self.itype, device=self.device)   

    def interpolate(self, u, etype=0, keepdims=False) -> TensorLike:
        """
        Compute the interpolation of a function u on the mesh.

        Parameters:
            u: The function to be interpolated.
            etype: The type of entity on which to interpolate.

        Example:
        ```
            from fealpy.mesh import UniformMesh2d
            mesh = UniformMesh2d(extent=[0, 10, 0, 10], h=(0.1, 0.1),
                                origin=(0.0, 0.0))
            u = mesh.interpolate(lambda x: x[..., 0]**2 + x[..., 1]**2)
            print(u)
        ```
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 0:
            node = self.entity('node')
            return u(node)
        else:
            raise ValueError(f"Unsupported entity type: {etype}")

    def linear_index_map(self, etype: Union[int, str]=0):
        """
        Build and return the tensor mapping multi-dimensional 
        indices to linear indices.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 0:
            return bm.arange(
                    self.NN, 
                    dtype=self.itype, 
                    device=self.device).reshape(self.nx + 1, self.ny + 1)
        elif etype == 2:
            return bm.arange(
                    self.NC, 
                    dtype=self.itype, 
                    device=self.device).reshape(self.nx, self.ny)

    # 实体生成方法
    @entitymethod(0)
    def _get_node(self) -> TensorLike:
        """
        @berif Generate the nodes in a structured mesh.
        """
        GD = 2
        nx, ny = self.nx, self.ny
        box = [self.origin[0], self.origin[0] + nx * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]

        x = bm.linspace(box[0], box[1], nx + 1, dtype=self.ftype, device=self.device)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=self.ftype, device=self.device)
        xx, yy = bm.meshgrid(x, y, indexing='ij')

        node = bm.concatenate((xx[..., None], yy[..., None]), axis=-1)

        if self.flip_direction == 'x':
            node = bm.flip(node.reshape(nx + 1, ny + 1, GD), axis=0).reshape(-1, GD)
        elif self.flip_direction == 'y':
            node = bm.flip(node.reshape(nx + 1, ny + 1, GD), axis=1).reshape(-1, GD)

        return node.reshape(-1, GD)
    
    @entitymethod(1)
    def _get_edge(self) -> TensorLike:
        """
        @berif Generate the edges in a structured mesh.
        """
        device = self.device

        nx, ny = self.nx, self.ny
        NN = self.NN
        NE = self.NE

        idx = bm.arange(NN, dtype=self.itype, device=device).reshape(nx + 1, ny + 1)

        edge = bm.zeros((NE, 2), dtype=self.itype, device=device)
        NE0 = 0
        NE1 = nx * (ny + 1)
        edge = bm.set_at(edge, (slice(NE0, NE1), 0), idx[:-1, :].reshape(-1))
        edge = bm.set_at(edge, (slice(NE0, NE1), 1), idx[1:, :].reshape(-1))
        # edge[NE0 + ny:NE1:ny + 1, :] = bm.flip(edge[NE0 + ny:NE1:ny + 1], axis=[1])

        NE0 = NE1
        NE1 += ny * (nx + 1)
        edge = bm.set_at(edge, (slice(NE0, NE1), 0), idx[:, :-1].reshape(-1))
        edge = bm.set_at(edge, (slice(NE0, NE1), 1), idx[:, 1:].reshape(-1))
        # edge[NE0:NE0 + ny, :] = bm.flip(edge[NE0:NE0 + ny], axis=[1])

        return edge
    
    @entitymethod(2)
    def _get_cell(self) -> TensorLike:
        """
        @berif Generate the cells in a structured mesh.
        """
        nx, ny = self.nx, self.ny
        NN = self.NN
        NC = self.NC

        idx = bm.arange(NN, device=self.device).reshape(nx + 1, ny + 1)

        cell = bm.zeros((NC, 4), dtype=self.itype, device=self.device)
        c = idx[:-1, :-1]
        cell_0 = c.reshape(-1)
        cell_1 = cell_0 + 1
        cell_2 = cell_0 + ny + 1
        cell_3 = cell_2 + 1
        cell = bm.concatenate([cell_0[:, None], cell_1[:, None], 
                               cell_2[:, None], cell_3[:, None]], axis=-1)

        return cell
    
    # 实体拓扑
    def number_of_nodes_of_cells(self):
        return 4

    def number_of_edges_of_cells(self):
        return 4

    def number_of_faces_of_cells(self):
        return 4
    
    def edge_to_cell(self, index: Index=_S) -> TensorLike:
        """
        @brief Adjacency relationship between edges and cells, 
        storing information about the two cells adjacent to each edge.
        Notes:
        - The first and second columns store the indices of the left and right cells adjacent to each edge. 
        When the two indices are the same, it indicates that the edge is a boundary edge.
        - The third and fourth columns store the local indices of the edge in the left and right cells, respectively.
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

        # left element
        edge2cell = bm.set_at(edge2cell, (idx0[:-1], 0), idx)
        edge2cell = bm.set_at(edge2cell, (idx0[:-1], 2), 0)
        edge2cell = bm.set_at(edge2cell, (idx0[-1], 0), idx[-1])
        edge2cell = bm.set_at(edge2cell, (idx0[-1], 2), 1)
        # edge2cell[idx0[:-1], 0] = idx
        # edge2cell[idx0[:-1], 2] = 0
        # edge2cell[idx0[-1], 0] = idx[-1]
        # edge2cell[idx0[-1], 2] = 1

        # right element
        edge2cell = bm.set_at(edge2cell, (idx0[1:], 1), idx)
        edge2cell = bm.set_at(edge2cell, (idx0[1:], 3), 1)
        edge2cell = bm.set_at(edge2cell, (idx0[0], 1), idx[0])
        edge2cell = bm.set_at(edge2cell, (idx0[0], 3), 0)
        # edge2cell[idx0[1:], 1] = idx
        # edge2cell[idx0[1:], 3] = 1
        # edge2cell[idx0[0], 1] = idx[0]
        # edge2cell[idx0[0], 3] = 0

        # left element
        edge2cell = bm.set_at(edge2cell, (NE0 + idx1[:, 1:], 0), idx)
        edge2cell = bm.set_at(edge2cell, (NE0 + idx1[:, 1:], 2), 3)
        edge2cell = bm.set_at(edge2cell, (NE0 + idx1[:, 0], 0), idx[:, 0])
        edge2cell = bm.set_at(edge2cell, (NE0 + idx1[:, 0], 2), 2)
        # edge2cell[NE0 + idx1[:, 1:], 0] = idx
        # edge2cell[NE0 + idx1[:, 1:], 2] = 3
        # edge2cell[NE0 + idx1[:, 0], 0] = idx[:, 0]
        # edge2cell[NE0 + idx1[:, 0], 2] = 2

        # right element
        edge2cell = bm.set_at(edge2cell, (NE0 + idx1[:, :-1], 1), idx)
        edge2cell = bm.set_at(edge2cell, (NE0 + idx1[:, :-1], 3), 2)
        edge2cell = bm.set_at(edge2cell, (NE0 + idx1[:, -1], 1), idx[:, -1])
        edge2cell = bm.set_at(edge2cell, (NE0 + idx1[:, -1], 3), 3)
        # edge2cell[NE0 + idx1[:, :-1], 1] = idx
        # edge2cell[NE0 + idx1[:, :-1], 3] = 2
        # edge2cell[NE0 + idx1[:, -1], 1] = idx[:, -1]
        # edge2cell[NE0 + idx1[:, -1], 3] = 3

        return edge2cell[index]
    
    def cell_to_edge(self, index: Index=_S) -> TensorLike:
        """
        @brief 单元和边的邻接关系，储存每个单元相邻的边的编号
        """
        NC = self.NC
        NE = self.NE

        nx = self.nx
        ny = self.ny

        cell2edge = bm.zeros((NC, 4), dtype=self.itype)

        idx0 = bm.arange(nx * (ny + 1)).reshape(nx, ny + 1)
        cell2edge = bm.set_at(cell2edge, (slice(None), 0), idx0[:, :-1].flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 1), idx0[:, 1:].flatten())
        # cell2edge[:, 0] = idx0[:, :-1].flatten()
        # cell2edge[:, 1] = idx0[:, 1:].flatten()

        idx1 = bm.arange(nx * (ny + 1), NE).reshape(nx + 1, ny)
        cell2edge = bm.set_at(cell2edge, (slice(None), 2), idx1[:-1, :].flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 3), idx1[1:, :].flatten())
        # cell2edge[:, 2] = idx1[:-1, :].flatten()
        # cell2edge[:, 3] = idx1[1:, :].flatten()

        return cell2edge[index]
        
    def boundary_node_flag(self):
        """
        @brief Determine if a point is a boundary point.
        """
        NN = self.NN
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdPoint = bm.zeros((NN,), dtype=bool)
        # isBdPoint[edge[isBdEdge, :]] = True
        isBdPoint = bm.set_at(isBdPoint, edge[isBdEdge, :], True)
        
        return isBdPoint
    
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
        # isBdCell[edge2cell[isBdEdge, 0]] = True
        isBdCell = bm.set_at(isBdCell, edge2cell[isBdEdge, 0], True)

        return isBdCell

    
#################################### 实体几何 #############################################
    def entity_measure(self, etype: Union[int, str], index: Index = _S) -> TensorLike:
        """
        @brief Get the measure of the entities of the specified type.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        NC = self.number_of_cells()
        if etype == 0:
            return bm.tensor(0, dtype=self.ftype)
        elif etype == 1:
            temp1 = bm.tensor([[self.h[0]], [self.h[1]]], dtype=self.ftype)
            temp2 = bm.broadcast_to(temp1, (2, int(self.NE/2)))
            return temp2.reshape(-1)
        elif etype == 2:
            temp = bm.tensor(self.h[0] * self.h[1], dtype=self.ftype)
            return bm.broadcast_to(temp, (NC,))
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
            TensorLike: (NC, NQ, GD) or (NE, NQ, GD)
        """
        node = self.entity('node')
        if isinstance(bcs, tuple):
            assert len(bcs) == 2
            cell = self.entity('cell', index)

            bcs0 = bcs[0].reshape(-1, 2)
            bcs1 = bcs[1].reshape(-1, 2)
            bcs = bm.einsum('im, jn -> ijmn', bcs0, bcs1).reshape(-1, 4)
            p = bm.einsum('qj, cjk -> cqk', bcs, node[cell[:]])
        else:
            edge = self.entity('edge', index=index)
            p = bm.einsum('qj, ejk -> eqk', bcs, node[edge]) 

        return p

    def get_adjusted_edge_mask(self) -> TensorLike:
        """
        Determine which edges need to have their direction adjusted to ensure normals point outward.

        Returns:
            TensorLike[NE]: A boolean array where True indicates that the edge's direction should be adjusted.
        """
        nx, ny = self.nx, self.ny
        NE = self.number_of_edges()
        adjusted_edge = bm.zeros((NE,), dtype=bool)

        # 水平边调整条件：每行最后一条边 (每个 ny+1 组中的最后一条)
        NE0 = 0
        NE1 = nx * (ny + 1)
        flip_indices_horiz = bm.arange(NE0 + ny, NE1, ny + 1)
        adjusted_edge = bm.set_at(adjusted_edge, flip_indices_horiz, True)

        # 垂直边调整条件：第一列的所有边
        NE0 = nx * (ny + 1)
        flip_indices_vert = bm.arange(NE0, NE0 + ny)
        adjusted_edge = bm.set_at(adjusted_edge, flip_indices_vert, True)

        return adjusted_edge
    
    def edge_normal(self, index: Index=_S, unit: bool=False, out=None) -> TensorLike:
        """
        Calculate the normal of the edges.

        Parameters:
            index (Index, optional): The indices of the edges to calculate normals for. 
                                        Defaults to _S (all edges).
            unit (bool, optional): Whether to return unit normals. 
                                        Defaults to False.
            out (TensorLike, optional): Optional output array to store the result. 
                                        Defaults to None.

        Returns:
            TensorLike[NE, GD]: Normal vectors of the edges.
        """
        edge = self.entity('edge', index=index)
        normals = bm.edge_normal(edge, self.node, unit=unit, out=out)

        adjusted_edge_mask = self.get_adjusted_edge_mask()

        normals = bm.set_at(normals, (adjusted_edge_mask, slice(None)),
                            -normals[adjusted_edge_mask])

        return normals

        
    def edge_unit_normal(self, index: Index=_S, out=None) -> TensorLike:
        """Calculate the unit normal of the edges.
        Equivalent to `edge_normal(index=index, unit=True)`.
        """
        return self.edge_normal(index=index, unit=True, out=out)
    
    def cell_location(self, points) -> TensorLike:
        """
        @brief 给定一组点，确定所有点所在的单元

        """
        hx = self.h[0]
        hy = self.h[1]
        v = bm.real(points - bm.array(self.origin, dtype=points.dtype))
        n0 = v[..., 0] // hx
        n1 = v[..., 1] // hy

        return n0.astype('int64'), n1.astype('int64')
    
    def point_to_bc(self, points):

        x = points[..., 0]
        y = points[..., 1]

        bc_x_ = bm.real((x - self.origin[0]) / self.h[0]) % 1
        bc_y_ = bm.real((y - self.origin[1]) / self.h[1]) % 1
        bc_x = bm.array([[bc_x_, 1 - bc_x_]], dtype=bm.float64)
        bc_y = bm.array([[bc_y_, 1 - bc_y_]], dtype=bm.float64)
        val = (bc_x, bc_y)

        return val
    

#################################### 插值点 #############################################
    def interpolation_points(self, p: int, index: Index=_S) -> TensorLike:
        '''
        @brief Generate all interpolation points of the 2D mesh

        @param p: Interpolation order. Must be an integer greater than 0.
        @param index: Index to select specific interpolation points.
        @param ordering: Specify the ordering of interpolation points. 
        Options are:
        - 'yx': Interpolation points are ordered first by the y direction, then by the x direction.
        - 'nec': Interpolation points are ordered with nodes first, then edges, and finally cells.
        '''

        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")
        if p == 1:
            return self.entity('node', index=index)
        
        ordering = self.ipoints_ordering
        
        if ordering == 'yx':
            nx = self.nx
            ny = self.ny
            hx = self.h[0]
            hy = self.h[1]

            nix = nx + 1 + nx * (p - 1)
            niy = ny + 1 + ny * (p - 1)
            
            length_x = nx * hx
            length_y = ny * hy

            ix = bm.linspace(0, length_x, nix)
            iy = bm.linspace(0, length_y, niy)

            x, y = bm.meshgrid(ix, iy, indexing='ij')
            ipoints = bm.stack([x.flatten(), y.flatten()], axis=-1)
        elif ordering == 'nec':
            GD = self.geo_dimension()
            NN = self.number_of_nodes()
            gdof = self.number_of_global_ipoints(p)
            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')
            ipoints = bm.zeros((gdof, GD), dtype=self.ftype)
            ipoints[:NN, :] = node

            NE = self.number_of_edges()
            multiIndex = self.multi_index_matrix(p, 1, dtype=self.ftype)
            w = multiIndex[1:-1, :] / p

            ipoints = bm.set_at(ipoints, (slice(NN, NN + (p-1) * NE), slice(None)), 
                    bm.einsum('ij, ...jm -> ...im', w, node[edge, :]).reshape(-1, GD))

            w = bm.einsum('im, jn -> ijmn', w, w).reshape(-1, 4)

            ipoints = bm.set_at(ipoints, (slice(NN + (p-1) * NE, None), slice(None)), 
                    bm.einsum('ij, kj... -> ki...', w, node[cell[:]]).reshape(-1, GD))
        else:
            raise ValueError("Invalid ordering type. \
                                Choose 'yx' for y-direction first then x-direction, "
                                "or 'nec' for node first, then edge, and finally cell ordering.")
        
        return ipoints[index]
    
    def node_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        '''
        @brief Returns the interpolation point indices corresponding to each node in a 2D mesh.

        @param p: Interpolation order. Must be an integer greater than 0.
        @param index: Index to select specific node interpolation points.

        @return: A 1D array of size (NN,) containing the indices of interpolation points at each node.
        '''
        ordering = self.ipoints_ordering
        
        if ordering == 'yx':
            nx = self.nx
            ny = self.ny        
            nix = nx + 1 + nx * (p - 1)
            niy = ny + 1 + ny * (p - 1)
            
            node_x_indices = bm.arange(0, nix, p)
            node_y_indices = bm.arange(0, niy, p)
            
            node_y_grid, node_x_grid = bm.meshgrid(node_y_indices, node_x_indices, indexing='ij')
            
            node2ipoint = (node_y_grid * nix + node_x_grid).flatten()
            node2ipoint = bm.astype(node2ipoint, self.itype)
            # node2ipoint = node2ipoint.astype(self.itype)
        elif ordering == 'nec':
            NN = self.NN
            node2ipoint = bm.arange(0, NN, dtype=self.itype)
        else:
            raise ValueError("Invalid ordering type. Choose 'yx' or 'nec'.")
        
        return node2ipoint[index]
    
    def edge_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        '''
        @brief Returns the interpolation point indices corresponding to each edge in a 2D mesh.

        @param p: Interpolation order. Must be an integer greater than 0.

        @return: A 2D array of size (NN, p+1) containing the indices of interpolation points at each edge.
        '''
        if p <= 0:
            raise ValueError("p must be an integer larger than 0.")
        
        ordering = self.ipoints_ordering
        edges = self.edge[index]
        
        if ordering == 'yx':
            node_to_ipoint = self.node_to_ipoint(p)
            
            start_indices = node_to_ipoint[edges[:, 0]]
            end_indices = node_to_ipoint[edges[:, 1]]
            
            linspace_indices = bm.linspace(0, 1, p + 1, endpoint=True, dtype=self.ftype).reshape(1, -1)
            edge2ipoint = start_indices[:, None] * (1 - linspace_indices) + \
                          end_indices[:, None] * linspace_indices
            edge2ipoint = bm.astype(edge2ipoint, self.itype)
        elif ordering == 'nec':
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            
            indices = bm.arange(NE, dtype=self.itype)[index]
            edge2ipoint =  bm.concatenate([
                edges[:, 0].reshape(-1, 1),
                (p-1) * indices.reshape(-1, 1) + bm.arange(0, p-1, dtype=self.itype) + NN,
                edges[:, 1].reshape(-1, 1),
            ], axis=-1)
        else:
            raise ValueError("Invalid ordering type. Choose 'yx' or 'nec'.")
        return edge2ipoint

        
    def cell_to_ipoint(self, p: int, index: Index=_S):
        """
        @brief Get the correspondence between mesh cells and interpolation points.

        @param p: Interpolation order. Must be an integer greater than 0.
        @param index: Index to select specific cells. Defaults to _S (all cells).

        @return: A 2D array of size (NC, (p+1)**2) containing the indices of interpolation points at each cell.
        """
        if p == 1:
            return self.entity('cell', index=index)
        
        ordering = self.ipoints_ordering

        if ordering == 'yx':
            edge_to_ipoint = self.edge_to_ipoint(p)
            cell2edge = self.cell_to_edge(index=index)

            start_indices = edge_to_ipoint[cell2edge[:, 0]]
            end_indices = edge_to_ipoint[cell2edge[:, 1]]

            linspace_indices = bm.linspace(0, 1, p + 1, endpoint=True, dtype=self.ftype).reshape(1, -1)

            cell_ipoints_interpolated = start_indices[:, :, None] * (1 - linspace_indices) + \
                                        end_indices[:, :, None] * linspace_indices

            cell2ipoint = cell_ipoints_interpolated.reshape(-1, (p+1)**2)
            cell2ipoint = cell2ipoint.astype(self.itype)
        elif ordering == 'nec':
            edge2cell = self.edge_to_cell()
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            cell2ipoint = bm.zeros((NC, (p + 1) * (p + 1)), dtype=self.itype)
            c2p= cell2ipoint.reshape((NC, p + 1, p + 1))
            e2p = self.edge_to_ipoint(p)

            # 确定哪些边在它的左边单元内是局部的第 0 号边
            flag = edge2cell[:, 2] == 0
            c2p = bm.set_at(c2p, (edge2cell[flag, 0], slice(None), 0), e2p[flag])
            # c2p[edge2cell[flag, 0], :, 0] = e2p[flag]

            # 确定哪些边在它的左边单元内是局部的第 1 号边
            flag = edge2cell[:, 2] == 1
            c2p = bm.set_at(c2p, (edge2cell[flag, 0], slice(None), -1), e2p[flag])
            # 逆序
            # c2p = bm.set_at(c2p, (edge2cell[flag, 0], slice(None), -1), 
            #                 bm.flip(e2p[flag], axis=[1]))
            # c2p[edge2cell[flag, 0], :, -1] = e2p[flag, -1::-1]

            # 确定哪些边在它的左边单元内是局部的第 2 号边
            flag = edge2cell[:, 2] == 2
            c2p = bm.set_at(c2p, (edge2cell[flag, 0], 0, slice(None)), e2p[flag])
            # 逆序
            # c2p = bm.set_at(c2p, (edge2cell[flag, 0], 0, slice(None)), 
            #                 bm.flip(e2p[flag], axis=[1]))
            # c2p[edge2cell[flag, 0], 0, :] = e2p[flag, -1::-1]

            # 确定哪些边在它的左边单元内是局部的第 3 号边
            flag = edge2cell[:, 2] == 3
            c2p = bm.set_at(c2p, (edge2cell[flag, 0], -1, slice(None)), e2p[flag])
            # 逆序
            # c2p = bm.set_at(c2p, (edge2cell[flag, 0], -1, slice(None)), e2p[flag])
            # c2p[edge2cell[flag, 0], -1, :] = e2p[flag]

            # 确定哪些边是内部边
            iflag = edge2cell[:, 0] != edge2cell[:, 1]

            # 确定哪些边在它的右边单元内是局部的第 0 号边
            rflag = edge2cell[:, 3] == 0
            flag = iflag & rflag
            c2p = bm.set_at(c2p, (edge2cell[flag, 1], slice(None), 0), e2p[flag])
            # # 逆序
            # c2p = bm.set_at(c2p, (edge2cell[flag, 1], slice(None), 0), 
            #                 bm.flip(e2p[flag], axis=[1]))
            # c2p[edge2cell[flag, 1], :, 0] = e2p[flag, -1::-1]

            # 确定哪些边在它的右边单元内是局部的第 1 号边
            rflag = edge2cell[:, 3] == 1
            flag = iflag & rflag
            c2p = bm.set_at(c2p, (edge2cell[flag, 1], slice(None), -1), e2p[flag])
            # c2p[edge2cell[flag, 1], :, -1] = e2p[flag]

            # 确定哪些边在它的右边单元内是局部的第 2 号边
            rflag = edge2cell[:, 3] == 2
            flag = iflag & rflag
            c2p = bm.set_at(c2p, (edge2cell[flag, 1], 0, slice(None)), e2p[flag])
            # c2p[edge2cell[flag, 1], 0, :] = e2p[flag]

            # 确定哪些边在它的右边单元内是局部的第 3 号边
            rflag = edge2cell[:, 3] == 3
            flag = iflag & rflag
            c2p = bm.set_at(c2p, (edge2cell[flag, 0], -1, slice(None)), e2p[flag])
            # # 逆序
            # c2p = bm.set_at(c2p, (edge2cell[flag, 1], -1, slice(None)), 
            #                 bm.flip(e2p[flag], axis=[1]))
            # c2p[edge2cell[flag, 1], -1, :] = e2p[flag, -1::-1]

            c2p = bm.set_at(c2p, (slice(None), slice(1, -1), slice(1, -1)), 
                        NN + NE * (p - 1) + bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p - 1, p - 1))
            # c2p[:, 1:-1, 1:-1] = NN + NE * (p - 1) + \
            #     bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p-1, p-1)
            
            cell2ipoint = cell2ipoint[index]
            
        return cell2ipoint

    face_to_ipoint = edge_to_ipoint
    
    # 形函数
    def jacobi_matrix(self, bcs: TensorLike, index: Index=_S) -> TensorLike:
        """
        @brief Compute the Jacobi matrix for the mapping from the reference element (xi, eta) 
               to the actual Lagrange quadrilateral (x)

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        gphi = self.grad_shape_function(bcs, p=1, variables='u')   # (NQ, ldof, GD)
        #TODO 这里不能翻转网格，否则会导致 Jacobian 计算错误
        node_cell_flip = node[cell[:]]                             # (NC, NCN, GD)
        J = bm.einsum('cim, qin -> cqmn', node_cell_flip, gphi)    # (NC, NQ, GD, GD)

        return J
    
    # 第一基本形式
    def first_fundamental_form(self, J: TensorLike) -> TensorLike:
        """
        @brief Compute the first fundamental form from the Jacobi matrix.
        """
        TD = J.shape[-1]

        shape = J.shape[0:-2] + (TD, TD)
        G = bm.zeros(shape, dtype=self.ftype)

        for i in range(TD):
            # 计算对角元素
            diag_val = bm.einsum('...d, ...d -> ...', J[..., i], J[..., i])
            G = bm.set_at(G, (..., i, i), diag_val)
            
            for j in range(i+1, TD):
                # 计算非对角元素
                off_diag_val = bm.einsum('...d, ...d -> ...', J[..., i], J[..., j])
                G = bm.set_at(G, (..., i, j), off_diag_val)
                G = bm.set_at(G, (..., j, i), off_diag_val)

        return G  

        # # TODO: Provide a unified implementation that is not backend-specific
        # if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
        #     for i in range(TD):
        #         G[..., i, i] = bm.einsum('...d, ...d -> ...', J[..., i], J[..., i])
        #         for j in range(i+1, TD):
        #             G[..., i, j] = bm.einsum('...d, ...d -> ...', J[..., i], J[..., j])
        #             G[..., j, i] = G[..., i, j]
                    
        #     return G
        # elif bm.backend_name == 'jax':
        #     for i in range(TD):
        #         G = G.at[..., i, i].set(bm.einsum('...d, ...d -> ...', 
        #                                         J[..., i], J[..., i]))
        #         for j in range(i + 1, TD):
        #             G = G.at[..., i, j].set(bm.einsum('...d, ...d -> ...', 
        #                                             J[..., i], J[..., j]))
        #             G = G.at[..., j, i].set(G[..., i, j])
        #     return G
        # else:
        #     raise NotImplementedError("Backend is not yet implemented.")
               

    # 其他方法
    def quadrature_formula(self, q: int, etype:Union[int, str]='cell'):
        """
        @brief Get the quadrature formula for numerical integration.
        """
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        qf = GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)
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
        The clear method is used at the end to clear the cache of entities. 
        This is necessary because the entities remain the same as before refinement due to caching.
        Structured meshes have their own entity generation methods, so the cache needs to be manually cleared.
        Unstructured meshes do not require this because they do not have entity generation methods.
        """
        for i in range(n):
            # self.extent = 2*self.extent
            self.extent = [i * 2 for i in self.extent]
            self.h = self.h/2.0 
            self.nx = self.extent[1] - self.extent[0]
            self.ny = self.extent[3] - self.extent[2]

            self.NC = self.nx * self.ny
            self.NE = self.ny * (self.nx + 1) + self.nx * (self.ny + 1)
            self.NF = self.NE
            self.NN = (self.nx + 1) * (self.ny + 1)

            self.edge2cell = self.edge_to_cell()
            self.cell2edge = self.cell_to_edge()
            self.face2cell = self.edge2cell

        self.clear()


    def to_vtk(self, filename, celldata=None, nodedata=None):
        """
        @brief: Converts the mesh data to a VTK structured grid format and writes to a VTS file
        """
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        if celldata is None:
            celldata = self.celldata

        if nodedata is None:
            nodedata = self.nodedata

        # 网格参数
        nx, ny = self.nx, self.ny
        h = self.h
        origin = self.origin

        # 创建坐标点
        x = bm.linspace(origin[0], origin[0] + nx * h[0], nx + 1)
        y = bm.linspace(origin[1], origin[1] + ny * h[1], ny + 1)
        z = bm.zeros(1)

        # 按 y, x 顺序重新组织坐标数组（左上到右下）
        xy_x, xy_y = bm.meshgrid(x, y, indexing='ij')
        xy_z = bm.zeros_like(xy_x)

        if self.flip_direction == 'y':
            # 左上到右下
            yx_x = xy_x[:, ::-1].flatten()
            yx_y = xy_y[:, ::-1].flatten()
            yx_z = xy_z[:, ::-1].flatten()
        elif self.flip_direction == None:
            # 默认：左下到右上
            yx_x = xy_x.flatten()
            yx_y = xy_y.flatten()
            yx_z = xy_z.flatten()

        # 创建 VTK 网格对象
        rectGrid = vtk.vtkStructuredGrid()
        rectGrid.SetDimensions(ny + 1, nx + 1, 1)

        # 创建点
        points = vtk.vtkPoints()
        for i in range(len(yx_x)):
            points.InsertNextPoint(yx_x[i], yx_y[i], yx_z[i])
        rectGrid.SetPoints(points)

        # 添加节点数据
        if nodedata is not None:
            for name, data in nodedata.items():
                if len(data.shape) > 1 and data.shape[1] > 1:
                    # 向量场数据
                    n_components = data.shape[1]
                    data_array = numpy_to_vtk(data, deep=True)
                    data_array.SetName(name)
                    data_array.SetNumberOfComponents(n_components)
                else:
                    # 标量场数据
                    data_array = numpy_to_vtk(data.ravel(), deep=True)
                    data_array.SetName(name)
                rectGrid.GetPointData().AddArray(data_array)

        # 添加单元格数据
        if celldata is not None:
            for name, data in celldata.items():
                data_array = numpy_to_vtk(data, deep=True)
                data_array.SetName(name)
                rectGrid.GetCellData().AddArray(data_array)

        # 写入 VTK 文件
        print("Writting to vtk...")
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetInputData(rectGrid)
        writer.SetFileName(filename)
        writer.Write()

        return filename


    # 界面网格
    def is_cut_cell(self, phi: Callable, *, eps=1e-10) -> TensorLike:
        """Return a bool tensor on cells indicating whether each cell is cut
        by the given function."""
        from ..geometry.functional import msign
        cellSign = msign(phi, eps=eps)[self.entity('cell')]
        dis = bm.max(cellSign, axis=1) - bm.min(cellSign, axis=1)
        return dis > 2.0 - 1e-8

    def find_interface_node(self, phi: Callable):
        """Find vertices of cut cells, solve cut points on edges, and generate aux points on special cells.

        Returns:
            iCellNodeIndex, cutNode, auxNode, isInterfaceCell
        """
        from ..geometry.functional import find_cut_point

        NN = self.number_of_nodes()
        EPS = 0.1 * min(self.h)**2

        node = self.entity('node')
        cell = self.entity('cell')[:, [0, 2, 3, 1]]
        phiValue = phi(node)
        phiValue = bm.set_at(phiValue, bm.abs(phiValue) < EPS, 0.0)
        phiSign = bm.sign(phiValue)

        # 寻找 cut 点
        edge = self.entity('edge')
        cutEdgeIndex = bm.nonzero(phiSign[edge[:, 0]] * phiSign[edge[:, 1]] < 0)[0]
        e0 = node[edge[cutEdgeIndex, 0]]
        e1 = node[edge[cutEdgeIndex, 1]]
        cutNode = find_cut_point(phi, e0, e1)
        del e0, e1, cutEdgeIndex, edge

        # 界面单元及其顶点
        isInterfaceCell = self.is_cut_cell(phiValue)
        isICellNode = bm.zeros(NN, dtype=bm.bool)
        isICellNode = bm.set_at(isICellNode, cell[isInterfaceCell, :], True)
        iCellNodeIndex = bm.nonzero(isICellNode)[0]

        # 寻找特殊单元：界面经过两对顶点的单元；构建辅助点：单元重心
        isSpecialCell = (bm.sum(bm.abs(phiSign[cell]), axis=1) == 2) \
                        & (bm.sum(phiSign[cell], axis=1) == 0)
        scell = cell[isSpecialCell, :]
        auxNode = (node[scell[:, 0], :] + node[scell[:, 2], :]) / 2

        return iCellNodeIndex, cutNode, auxNode, isInterfaceCell

    def function(self, etype='node', dim=None, dtype=None, ex=0):
        """Return a discrete function (array) defined on nodes, mesh edges, or mesh cells with elements set to 0"""
        nx = self.nx
        ny = self.ny
        dtype = self.ftype if dtype is None else dtype
        if etype in {'node', 0}:
            if dim is None:
                uh = bm.zeros((nx + 1 + 2 * ex, ny + 1 + 2 * ex), dtype=dtype)
            else:
                uh = bm.zeros((nx + 1 + 2 * ex, ny + 1 + 2 * ex, dim), dtype=dtype)

        elif etype in {'edge', 'face', 1}:
            ex = bm.zeros((nx, ny + 1), dtype=dtype)
            ey = bm.zeros((nx + 1, ny), dtype=dtype)
            uh = (ex, ey)
        elif etype in {'edgex'}:
            uh = bm.zeros((nx, ny + 1), dtype=dtype)
        elif etype in {'edgey'}:
            uh = bm.zeros((nx + 1, ny), dtype=dtype)
        elif etype in {'cell', 2}:
            uh = bm.zeros((nx + 2 * ex, ny + 2 * ex), dtype=dtype)
        else:
            raise ValueError(f'the entity `{etype}` is not correct!')

        return uh    
    
    def update_dirichlet_bc(self, 
            gD: Callable[[TensorLike], Any], 
            uh: TensorLike
            ) -> None:
        """更新网格函数 uh 的 Dirichlet 边界值"""
        node = self.node
        isBdNode = self.boundary_node_flag().reshape(uh.shape)
        uh[isBdNode] = gD(node[isBdNode, :])

    def error(self,
            u: Callable,
            uh: TensorLike,
            errortype: str = 'all'
        ) -> Union[float, Tuple[float, float, float]]:
        """Compute error metrics between exact solution and numerical solution in 2D.
    
        Calculates various error norms between the exact solution u(x,y) and the 
        numerical solution uh on a 2D grid. Supports multiple error types including
        maximum absolute error, continuous L2 norm, and discrete l2 norm.

        Parameters
            u : Callable
                The exact solution function u(x,y) that takes node coordinates and 
                returns exact solution values. Must accept the same 2D grid nodes 
                used in the numerical solution.
            uh : TensorLike
                The numerical solution values at discrete nodes. Should have shape
                matching u(node) output.
            errortype : str, optional, default='all'
                Specifies which error norm(s) to compute:
                - 'all': returns all three error metrics
                - 'max': only maximum absolute error (L∞ norm)
                - 'L2': only continuous L2 norm error
                - 'l2': only discrete l2 norm error

        Returns
            error_metrics : Union[float, Tuple[float, float, float]]
                The computed error metric(s) depending on errortype:
                - If 'all': returns tuple (emax, e0, el2)
                    emax: maximum absolute error (L∞ norm)
                    e0: continuous L2 norm error (integral-based)
                    el2: discrete l2 norm error (average-based)
                - Otherwise returns single float for specified error type

        Notes
            The error norms are computed as:
            - L∞ norm: max|u(x_i,y_j) - uh(x_i,y_j)|
            - Continuous L2 norm: sqrt(hx*hy * Σ(u(x_i,y_j) - uh(x_i,y_j))²)
            - Discrete l2 norm: sqrt(1/((nx-1)(ny-1)) * Σ(u(x_i,y_j) - uh(x_i,y_j))²)

            where:
            - hx, hy are mesh spacings in x and y directions
            - nx, ny are numbers of grid points in each direction

        Examples
            >>> # For 2D problem with nx=ny=11 grid
            >>> exact_sol = lambda p: p[:,0]**2 + p[:,1]**2  # x² + y²
            >>> numerical_sol = bm.random.rand(121)  # dummy solution
            >>> emax, eL2, el2 = error(exact_sol, numerical_sol)
        """
        hx = self.h[0]
        hy = self.h[1]
        nx = self.nx
        ny = self.ny
        node = self.node
        uI = u(node)
        e = uI - uh

        if errortype == 'all':
            emax = bm.max(bm.abs(e))
            e0 = bm.sqrt(hx * hy * bm.sum(e ** 2))
            el2 = bm.sqrt(1 / ((nx - 1) * (ny - 1)) * bm.sum(e ** 2))

            return emax, e0, el2
        elif errortype == 'max':
            emax = bm.max(bm.abs(e))
            return emax
        elif errortype == 'L2':
            e0 = bm.sqrt(hx * hy * bm.sum(e ** 2))
            return e0
        elif errortype == 'l2':
            el2 = bm.sqrt(1 / ((nx - 1) * (ny - 1)) * bm.sum(e ** 2))
            return el2
        
    def show_function(self, plot, uh, aspect=[1, 1, 1], cmap='rainbow'):
        """显示一个定义在网格节点上的函数"""
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            axes = fig.add_subplot(111, projection='3d')
        else:
            axes = plot

        axes.set_box_aspect(aspect)
        axes.set_proj_type('ortho')

        node = self.node  # 获取二维节点上的网格坐标
        if uh.ndim == 1:
            uh = uh.reshape(self.nx+1, self.ny+1)
        return axes.plot_surface(node[..., 0].reshape(self.nx+1, self.ny+1), 
                                node[..., 1].reshape(self.nx+1, self.ny+1), 
                                uh, cmap=cmap)

    def show_animation(self,
            fig: Figure,
            axes: Union[Axes, Axes3D],
            box: List[float],
            advance: Callable[[int], Tuple[TensorLike, float]],
            fname: str = 'test.mp4',
            init: Optional[Callable] = None,
            fargs: Optional[Callable] = None,
            frames: int = 1000,
            interval: int = 50,
            plot_type: str = 'imshow',
            cmap='rainbow'
        ) -> None:
        """生成求解过程动画并保存为指定文件名的视频文件"""
        # 创建动画所需的类和函数
        import matplotlib.animation as animation
        # 绘制颜色条的类
        from matplotlib.contour import QuadContourSet

        nx, ny = self.shape

        # 初始化二维网格数据
        uh, _ = advance(0)
        uh_2d = uh.reshape(nx+1, ny+1) if uh.ndim == 1 else uh

        if isinstance(axes, Axes) and plot_type == 'imshow':
            data = axes.imshow(uh_2d, cmap=cmap, vmin=box[4], vmax=box[5],
                               extent=box[0:4], interpolation='bicubic')
        elif isinstance(axes, Axes3D) and plot_type == 'surface':
            X = self.node[..., 0].reshape(nx+1, ny+1)
            Y = self.node[..., 1].reshape(nx+1, ny+1)
            data = axes.plot_surface(X, Y, uh_2d, linewidth=0, cmap=cmap, vmin=box[4],
                                    vmax=box[5], rstride=1, cstride=1)
            axes.set_xlim(box[0], box[1])
            axes.set_ylim(box[2], box[3])
            axes.set_zlim(box[4], box[5])
        elif plot_type == 'contourf':
            X = self.node[..., 0].reshape(nx+1, ny+1)
            Y = self.node[..., 1].reshape(nx+1, ny+1)
            data = axes.contourf(X, Y, uh_2d, cmap=cmap, vmin=box[4], vmax=box[5])
            # data 的值在每一帧更新时都会发生改变 颜色条会根据这些更改自动更新
            # 后续的代码中无需对颜色条进行额外的更新操作
            # cbar = fig.colorbar(data, ax=axes)

        def func(n, *fargs):  # 根据当前帧序号计算数值解，更新图像对象的数值数组，显示当前帧序号和时刻
            nonlocal data  # 声明 data 为非局部变量 这样在 func 函数内部对 data 进行的修改会影响到外部的 data 变量
            uh, t = advance(n, *fargs)  # 计算当前时刻的数值解并返回，uh 是数值解，t 是当前时刻
            uh_2d = uh.reshape(nx+1, ny+1) if uh.ndim == 1 else uh

            if data is None:
                if isinstance(axes, Axes) and plot_type == 'imshow':
                    data = axes.imshow(uh_2d, cmap=cmap, vmin=box[4], vmax=box[5],
                                       extent=box[0:4], interpolation='bicubic')
                elif isinstance(axes, Axes3D) and plot_type == 'surface':
                    data = axes.plot_surface(X, Y, uh_2d, cmap=cmap, vmin=box[4],
                                             vmax=box[5], rstride=1, cstride=1)
                elif plot_type == 'contourf':
                    data = axes.contourf(X, Y, uh_2d, cmap=cmap, vmin=box[4], vmax=box[5])

            if isinstance(axes, Axes) and plot_type == 'imshow':
                data.set_array(uh_2d)  # 更新 data 对象的数值数组。导致图像的颜色根据新的数值解 uh 更新
                axes.set_aspect('equal')  # 设置坐标轴的长宽比。'equal' 选项使得 x 轴和 y 轴的单位尺寸相等

            elif isinstance(axes, Axes3D) and plot_type == 'surface':
                axes.clear()  # 清除当前帧的图像
                X = self.node[..., 0].reshape(nx+1, ny+1)
                Y = self.node[..., 1].reshape(nx+1, ny+1)
                data = axes.plot_surface(X, Y, uh_2d, cmap=cmap, vmin=box[4], vmax=box[5])
                axes.set_xlim(box[0], box[1])
                axes.set_ylim(box[2], box[3])
                axes.set_zlim(box[4], box[5])
            elif plot_type == 'contourf':
                # 使用 contourf 时，每次更新图像时都会生成一个新的等高线填充层
                # data.collections 保存了所有已经生成的等高线填充层
                # 更新图像时 需要将旧的等高线填充层从图形中移除 以免遮挡住新的等高线填充层
                X = self.node[..., 0].reshape(nx+1, ny+1)
                Y = self.node[..., 1].reshape(nx+1, ny+1)
                if data is not None:
                    if isinstance(data, QuadContourSet):
                        for coll in data.collections:
                            if coll in axes.collections:
                                coll.remove()
                data = axes.contourf(X, Y, uh_2d, cmap=cmap, vmin=box[4], vmax=box[5])
                axes.set_aspect('equal')

            s = "frame=%05d, time=%0.8f" % (n, t)  # 创建一个格式化的字符串，显示当前帧序号 n 和当前时刻 t
            print(s)
            axes.set_title(s)  # 将格式化的字符串设置为坐标轴的标题
            return data

        # 创建一个 funcanimation 对象
        # fig 作为画布，func 作为帧更新函数
        # init_func 作为初始化函数，用于在动画开始之前设置图像的初始状态
        # fargs 作为一个元组，包含要传递给 func 函数的额外参数
        # frames 为帧数，interval 为动画间隔时间
        ani = animation.FuncAnimation(fig, func, init_func=init, fargs=fargs, frames=frames, interval=interval)
        ani.save('{}_{}'.format(plot_type, fname))


UniformMesh2d.set_ploter('2d')
