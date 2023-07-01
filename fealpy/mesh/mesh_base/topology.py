"""
Provide abstract class for mesh with different topology dimension.
"""
from typing import Union, Optional
import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh
from ..mesh_data_structure import (
    Mesh1dDataStructure,
    Mesh2dDataStructure,
    Mesh3dDataStructure
)


##################################################
### Topology dimension 1 Mesh
##################################################

class Mesh1d(Mesh):
    """
    @brief The abstract class for meshes with topology dimension 1.

    This is a subclass of Mesh, but some methods are still anstract.

    @note: Abstract methods list:
    ```
    def uniform_refine(self, n: int=1) -> int: ...
    ```
    """
    ds: Mesh1dDataStructure

    def integrator(self, k, etype='cell'):
        """
        @brief 返回第 k 个高斯积分公式。
        """
        from ...quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(k)

    def entity_measure(self, etype: Union[int, str]='cell', index=np.s_[:], node=None):
        """
        """
        if etype in {1, 'cell', 'edge'}:
            return self.cell_length(index=index, node=None)
        elif etype in {0, 'face', 'node'}:
            return np.array([0], dtype=self.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def grad_lambda(self, index=np.s_[:]):
        """
        @brief 计算所有单元上重心坐标函数的导数
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        v = node[cell[:, 1]] - node[cell[:, 0]]
        NC = len(cell) 
        GD = self.geo_dimension()
        Dlambda = np.zeros((NC, 2, GD), dtype=self.ftype)
        h2 = np.sum(v**2, axis=-1)
        v /=h2.reshape(-1, 1)
        Dlambda[:, 0, :] = -v
        Dlambda[:, 1, :] = v
        return Dlambda

    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        return p + 1

    def number_of_global_ipoints(self, p: int) -> int:
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        return NN + (p-1)*NC

    def interpolation_points(self, p: int, index=np.s_[:]) -> NDArray:
        GD = self.geo_dimension()
        node = self.entity('node')

        if p == 1:
            return node
        else:
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            gdof = NN + NC*(p-1)
            ipoint = np.zeros((gdof, GD), dtype=self.ftype)
            ipoint[:NN] = node
            cell = self.entity('cell')
            w = np.zeros((p-1,2), dtype=np.float64)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            GD = self.geo_dimension()
            ipoint[NN:NN+(p-1)*NC] = np.einsum('ij, kj...->ki...', w,
                    node[cell]).reshape(-1, GD)

            return ipoint




##################################################
### Topology dimension 2 Mesh
##################################################

class Mesh2d(Mesh):
    """
    @brief The abstract class for meshes with topology dimension 2.

    This is a subclass of Mesh, but some methods are still anstract.

    @note: Abstract methods list:
    ```
    def uniform_refine(self, n: int=1) -> int: ...
    def integrator(self, k: int, etype: Union[int, str]): ...
    def multi_index_matrix(p: int, etype: Union[int, str]='cell') -> NDArray: ...
    def shape_function(self, bc: NDArray, p: int) -> NDArray: ...
    def grad_shape_function(self, bc: NDArray, p: int, index=np.s_[:]) -> NDArray: ...
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int: ...
    def number_of_global_ipoints(self, p: int) -> int: ...
    def interpolation_points(self, p: int) -> NDArray: ...
    def cell_to_ipoint(self, p: int, index=np.s_[:]): ...
    def face_to_ipoint(self, p: int, index=np.s_[:]): ...
    def edge_to_ipoint(self, p: int, index=np.s_[:]): ...
    def node_to_ipoint(self, p: int, index=np.s_[:]): ...
    ```
    """
    ds: Mesh2dDataStructure

    def entity_measure(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return 0
        else:
            raise ValueError(f"Invalid entity type '{etype}'.")


    def cell_area(self, index=np.s_[:]):
        """
        @brief 根据散度定理计算多边形的面积
        @note 请注意下面的计算方式不方便实现部分单元面积的计算
        """
        NC = self.number_of_cells()
        node = self.entity('node')
        edge = self.entity('edge')
        edge2cell = self.ds.edge_to_cell()

        t = self.edge_tangent()
        val = t[:, 1]*node[edge[:, 0], 0] - t[:, 0]*node[edge[:, 0], 1] 

        a = np.zeros(NC, dtype=self.ftype)
        np.add.at(a, edge2cell[:, 0], val)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        np.add.at(a, edge2cell[isInEdge, 1], -val[isInEdge])

        a /= 2.0

        return a[index]

    ## Special Methods in 2D
    def node_size(self):
        """
        @brief 计算每个网格节点邻接边的长度平均值, 做为节点处的网格尺寸值
        """
        NN = self.number_of_nodes()
        edge = self.entity('edge')
        eh = self.entity_measure('edge')
        h = np.zeros(NN, dtype=self.ftype)
        deg = np.zeros(NN, dtype=self.itype)

        val = np.broadcast_to(eh[:, None], shape=edge.shape)
        np.add.at(h, edge, val)
        np.add.at(deg, edge, 1)

        return h/deg


    def edge_frame(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上的局部标架 
        """
        assert self.geo_dimension() == 2
        t = self.edge_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        n = t@w
        return n, t

    def edge_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def edge_unit_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    face_normal = edge_normal
    face_unit_normal = edge_unit_normal

##################################################
### Topology dimension 3 Mesh
##################################################

class Mesh3d(Mesh):
    """
    @brief The abstract class for meshes with topology dimension 3.

    This is a subclass of Mesh, but some methods are still anstract.

    @note: Abstract methods list:
    ```
    ...
    ```
    """
    ds: Mesh3dDataStructure

    def entity_measure(self, etype=3, index=np.s_[:]):
        if etype in {'cell', 3}:
            return self.cell_volume(index=index)
        elif etype in {'face', 2}:
            return self.face_area(index=index)
        elif etype in {'edge', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return np.zeros(1, dtype=self.ftype)
        else:
            raise ValueError("`entitytype` is wrong!")




