"""
Provide abstract class for mesh with different topology dimension.
"""
from typing import Union
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

    def entity_measure(self, etype: Union[int, str]='cell', index=np.s_[:]) -> NDArray:
        # TODO: finish this
        pass

    def cell_length(self, index=np.s_[:]):
        """
        @brief
        """
        pass

    @staticmethod
    def multi_index_matrix(p: int, etype: Union[int, str]=1) -> NDArray:
        ldof = p+1
        multiIndex = np.zeros((ldof, 2), dtype=np.int_)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

    def shape_function(self, bc, p=1) -> NDArray:
        TD = bc.shape[-1] - 1
        multiIndex = self.multi_index_matrix(p)
        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    def grad_shape_function(self, bc: NDArray, p=1, index=np.s_[:]) -> NDArray:
        TD = self.top_dimension()

        multiIndex = self.multi_index_matrix(p)

        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = self.number_of_local_ipoints(p)
        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        Dlambda = self.grad_lambda(index=index)
        gphi = np.einsum('...ij, kjm->...kim', R, Dlambda)
        return gphi

    def grad_lambda(self, index=np.s_[:]) -> NDArray:
        pass

    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        return p + 1

    def number_of_global_ipoints(self, p: int) -> int:
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        return NN + (p-1)*NC

    def interpolation_points(self, p: int) -> NDArray:
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

    def cell_to_ipoint(self, p, index=np.s_[:]):
        """
        @brief 获取网格边与插值点的对应关系
        """
        NC = self.number_of_cells()
        NN = self.number_of_nodes()

        cell = self.entity('cell')
        cell2ipoints = np.zeros((NC, p+1), dtype=np.int_)
        cell2ipoints[:, [0, -1]] = cell
        if p > 1:
            cell2ipoints[:, 1:-1] = NN + np.arange(NC*(p-1)).reshape(NC, p-1)
        return cell2ipoints[index]

    edge_to_ipoint = cell_to_ipoint

    def node_to_ipoint(self, p, index=np.s_[:]):
        return np.arange(self.number_of_nodes())

    face_to_ipoint = node_to_ipoint


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

    def edge_length(self, index=np.s_[:]):
        """
        @brief
        """
        node = self.entity('node')
        edge = self.entity('edge')
        v = node[edge[index,1],:] - node[edge[index,0],:]
        return np.linalg.norm(v, axis=1)

    def cell_area(self, index=np.s_[:]):
        """
        @brief 根据散度定理计算多边形的面积
        @note 请注意下面的计算方式不方便实现部分单元面积的计算
        """
        NC = self.number_of_cells()
        node = self.entity('node')
        edge = self.entity('edge')
        edge2cell = self.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        v =  node[edge[:, 1], :] - node[edge[:, 0], :]
        val = np.einsum('ij, ij->i', v, node[edge[:, 0], :], optimize=True)
        a = np.bincount(edge2cell[:, 0], weights=val, minlength=NC)
        a+= np.bincount(edge2cell[isInEdge, 1], weights=-val[isInEdge], minlength=NC)
        a /=2
        return a[index]

    ## Special Methods in 2D
    def node_size(self):
        """
        @brief
        计算每个网格节点邻接边的长度平均值, 做为节点处的网格尺寸值
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

    def face_normal(self, index=np.s_[:]):
        v = self.face_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def face_unit_normal(self, index=np.s_[:]):
        v = self.face_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def face_tangent(self, index=np.s_[:]):
        node = self.entity('node')
        edge = self.entity('edge')
        v = node[edge[index,1],:] - node[edge[index,0],:]
        return v

    def face_unit_tangent(self, index=np.s_[:]):
        edge = self.entity('edge')
        node = self.entity('node')
        v = node[edge[index,1],:] - node[edge[index,0],:]
        length = np.linalg.norm(v, ord=2, axis=1)
        v /= length.reshape(-1, 1)
        return v

    def edge_frame(self, index=np.s_[:]):
        t = self.edge_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        n = t@w
        return n, t

    def edge_unit_normal(self, index=np.s_[:]):
        #TODO: 3D Case
        v = self.edge_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def edge_unit_tangent(self, index=np.s_[:]):
        node = self.entity('node')
        edge = self.entity('edge')
        v = node[edge[index, -1],:] - node[edge[index, 0],:]
        length = np.linalg.norm(v, axis=1)
        v /= length.reshape(-1, 1)
        return v

    def edge_normal(self, index=np.s_[:]):
        v = self.edge_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def edge_tangent(self, index=np.s_[:]):
        node = self.entity('node')
        edge = self.entity('edge')
        v = node[edge[index, 1],:] - node[edge[index, 0],:]
        return v


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

    def cell_volume(self, index=np.s_[:]):
        pass

    def face_area(self, index=np.s_[:]):
        pass

    def edge_length(self, index=np.s_[:]):
        pass

    def edge_tangent(self):
        edge = self.ds.edge
        node = self.node
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        return v

    def edge_unit_tangent(self):
        edge = self.ds.edge
        node = self.node
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        length = np.sqrt(np.square(v).sum(axis=1))
        return v/length.reshape(-1, 1)
