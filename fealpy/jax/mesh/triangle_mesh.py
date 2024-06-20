
from typing import Any, Callable, Optional, List, Union

import numpy as np
import jax
import jax.numpy as jnp

from .. import logger

from . import functional as F
from .utils import Array
from .mesh_base import SimplexMesh 


class TriangleMesh(SimplexMesh):
    def __init__(self, node: Array, cell: Array):
        """
        @brief TriangleMesh 对象的初始化函数
        """
        self.node = node
        NN = node.shape[0]
        TD = 2
        super().__init__(NN, TD)

        self.localEdge = jnp.array([(1, 2), (2, 0), (0, 1)])
        self.localFace = jnp.array([(1, 2), (2, 0), (0, 1)])
        self.ccw = jnp.array([0, 1, 2])

        self.localCell = np.array([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)])


        logger.info(f"Initialize a {GD}D TriangleMesh instance with {NN} nodes ({node.dtype}) and {NC} cells ({cell.dtype}).")

        self._edge_length = jax.jit(jax.vmap(K.edge_length))

        if GD == 2:
            self._cell_area = jax.jit(jax.vmap(K.tri_area_2d))
            self._cell_area_with_jac = jax.jit(jax.vmap(K.tri_area_2d_with_jac))
            self._grad_lambda = jax.jit(jax.vmap(K.tri_grad_lambda_2d))
        elif GD == 3:
            self._cell_area = jax.jit(jax.vmap(K.tri_area_3d))
            self._cell_area_with_jac = jax.jit(jax.vmap(K.tri_area_3d_with_jac))
            self._grad_lambda = jax.jit(jax.vmap(K.tri_grad_lambda_3d))

        self._quality = jax.jit(jax.vmap(K.tri_quality_radius_ratio))
        self._quality_with_jac = jax.jit(jax.vmap(K.tri_quality_radius_ratio_with_jac))

    def number_of_local_ipoints(self, p: int, iptype: Union[str, int]='cell') -> int:
        """Count the local interpolation points.

        Args:
            p (int): The order of the interpolation points.
            iptype (str | int, optional): The interpolation type. Defaults to 'cell'.

        Returns:
            int: The number of local interpolation points, denoted by 'ldof'.
        """
        if isinstance(iptype, str):
            iptype = entity_str2dim(self.ds, iptype)
        return F.simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int) -> int:
        """Count the global interpolation points.

        Args:
            p (int): The order of the interpolation points.

        Returns:
            int: The number of global interpolation points, denoted by 'gdof'.
        """
        return F.simplex_gdof(p, self)

    def grad_lambda(self, index=np.s_[:]) -> Array:
        return self._grad_lambda(self.node[self.ds.cell[index]])

    def shape_function(self, bcs: Array, p: int=1, variable='u') -> Array:
        TD = bcs.shape[-1] - 1
        mi = self.multi_index_matrix(p, TD)
        phi = K.simplex_shape_function(bcs, mi, p)
        if variable == 'x':
            return phi[None, ...] # (1, NQ, ldof)
        elif variable == 'u':
            return phi # (NQ, ldof)
        else:
            logger.error(f"The variable type {variable} is not correct, which should be `x` or `u`!")

    def grad_shape_function(self, bcs: Array, p: int=1, index=jnp.s_[:], variable='u'):
        """
        @note 注意这里调用的实际上不是形状函数的梯度，而是网格空间基函数的梯度
        """
        TD = bcs.shape[-1] - 1
        mi = self.multi_index_matrix(p, TD)
        R = K.diff_simplex_shape_function(bcs, mi, p, 1) # (NQ, ldof, TD+1)
        if variable == 'u':
            return R
        elif variable == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = jnp.einsum('...ij, kjm->k...im', R, Dlambda, optimize=True)
            return gphi #(NC, NQ, ldof, GD)
        else:
            logger.error(f"The variable type {variable} is not correct, which should be `x` or `u`!")

    cell_grad_shape_function = grad_shape_function

    def hess_shape_function(self, bcs: Array, p: int=1, index=jnp.s_[:], variable='u'):
        """
        @note 注意这里调用的实际上不是形状函数的梯度，而是网格空间基函数的梯度
        """
        TD = bcs.shape[-1] - 1
        mi = self.multi_index_matrix(p, TD)
        R = K.diff_simplex_shape_function(bcs, mi, p, 2) # (NQ, ldof, TD+1, TD+1)
        if variable == 'u':
            return R
        elif variable == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = jnp.einsum('...ijk, cjm, ckn->c...imn', R, Dlambda, Dlambda, optimize=True)
            return gphi #(NC, NQ, ldof, GD, GD)
        else:
            logger.error(f"The variable type {variable} is not correct, which should be `x` or `u`!")


    def edge_unit_normal(self, index=jnp.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_unit_tangent(index=index)
        w = jnp.array([(0,-1),(1,0)])
        return v@w

    face_unit_normal = edge_unit_normal

    def laplace_shape_function(self, bcs: Array, p: int=1, index=jnp.s_[:]) -> Array:
        """
        @brief 计算 p 次 Lagrange 基函数的拉普拉斯
        """
        TD = bcs.shape[-1] - 1
        mi = self.multi_index_matrix(p, TD)
        R = K.grad_simplex_shape_function(bcs, mi, p, 2)
        Dlambda = self.grad_lambda(index=index)
        lphi = jnp.einsum('cjm, ...ijk, ckm->k...i', Dlambda, R, Dlambda, optimize=True)
        return lphi #(NC, NQ, ldof, GD)

    def integrator(self, q: int, etype='cell', qtype: str='legendre'):
        if etype in {'cell', 2}:
            from .quadrature import TriangleQuadrature
            quad = TriangleQuadrature(dtype=self.ftype)
        elif etype in {'edge', 'face', 1}:
            from .quadrature import GaussLegendreQuadrature
            quad = GaussLegendreQuadrature(dtype=self.ftype)
        else:
            raise ValueError(f"No quadrature provided for entity type '{etype}'.")

        quad._latest_order = q
        return quad

    def entity_measure(self, etype: Union[int, str]='cell', index=np.s_[:]) -> Array:
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return 0
        else:
            raise ValueError(f"Invalid entity type '{etype}'.")

    def cell_area(self, index=jnp.s_[:]) -> Array:
        return self._cell_area(self.node[self.ds.cell[index]])

    def edge_length(self, index=jnp.s_[:]) -> Array:
        return self._edge_length(self.node[self.ds.edge[index]])

    def cell_area_with_jac(self, index=jnp.s_[:]):
        return self._cell_area_with_jac(self.node[self.ds.cell[index]])

    def cell_quality(self, index=jnp.s_[:]):
        return  self._quality(self.node[self.ds.cell[index]])

    def cell_quality_with_jac(self, index=jnp.s_[:]):
        return self._quality_with_jac(self.node[self.ds.cell[index]])

    def interpolation_points(self, p: int, index=jnp.s_[:]) -> Array:
        """
        @brief 获取三角形网格上所有 p 次插值点
        """
        GD = self.geo_dimension()
        node = self.entity('node')
        if p == 1:
            return node
        if p > 1:
            edge = self.entity('edge')
            w = self.multi_index_matrix(p, 1)[1:-1]/p
            enode = jnp.einsum('ij, ...jm->...im', w,
                    node[edge,:]).reshape(-1, GD)
            ipoints = jnp.vstack((node, enode))
        if p > 2:
            cell = self.entity('cell')
            mi = self.multi_index_matrix(p, 2)
            flag = (jnp.sum(mi > 0 , axis=1) == 3)
            w = mi[flag, :]/p
            cnode = np.einsum('ij, kj...->ki...', w,
                    node[cell, :]).reshape(-1, GD)
            ipoints = jnp.vstack((ipoints, cnode))

        return ipoints # (gdof, GD)

    def cell_to_ipoint(self, p: int, index=jnp.s_[:]) -> Array:
        """
        @brief  获得 p 次 Lagrange 元的插值点编号
        """
        cell = self.entity('cell')
        if p==1:
            return cell[index]

        mi = self.multi_index_matrix(p, 2)
        idx0, = np.nonzero(mi[:, 0] == 0)
        idx1, = np.nonzero(mi[:, 1] == 0)
        idx2, = np.nonzero(mi[:, 2] == 0)

        edge2cell = self.ds.edge2cell
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        e2p = self.edge_to_ipoint(p)
        ldof = self.number_of_local_ipoints(p)
        c2p = np.zeros((NC, ldof), dtype=self.itype)

        flag = edge2cell[:, 2] == 0
        c2p[edge2cell[flag, 0][:, None], idx0] = e2p[flag]

        flag = edge2cell[:, 2] == 1
        c2p[edge2cell[flag, 0][:, None], idx1[-1::-1]] = e2p[flag]

        flag = edge2cell[:, 2] == 2
        c2p[edge2cell[flag, 0][:, None], idx2] = e2p[flag]

        iflag = edge2cell[:, 0] != edge2cell[:, 1]

        flag = iflag & (edge2cell[:, 3] == 0)
        c2p[edge2cell[flag, 1][:, None], idx0[-1::-1]] = e2p[flag]

        flag = iflag & (edge2cell[:, 3] == 1)
        c2p[edge2cell[flag, 1][:, None], idx1] = e2p[flag]

        flag = iflag & (edge2cell[:, 3] == 2)
        c2p[edge2cell[flag, 1][:, None], idx2[-1::-1]] = e2p[flag]

        cdof = (p-1)*(p-2)//2
        flag = np.sum(mi > 0, axis=1) == 3
        c2p[:, flag] = NN + NE*(p-1) + np.arange(NC*cdof).reshape(NC, cdof)
        return jnp.array(c2p[index])

    @classmethod
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10, threshold=None):
        """
        Generate a triangle mesh for a box domain using jax.numpy, optimizing both node and cell array creation.

        @param box
        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TriangleMesh instance
        """
        X, Y = jnp.mgrid[
                box[0]:box[1]:complex(0, nx+1),
                box[2]:box[3]:complex(0, ny+1)]
        node = jnp.column_stack((X.ravel(), Y.ravel()))

        idx = jnp.arange((nx+1) * (ny+1)).reshape(nx+1, ny+1)

        # Defining cells for the two triangles within each square grid
        cell0 = jnp.column_stack((idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, :-1].ravel()))
        cell1 = jnp.column_stack((idx[:-1, 1:].ravel(), idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()))

        # Concatenating the two sets of cells to form the complete cell array
        cell = jnp.concatenate((cell0, cell1), axis=0)

        if threshold is not None:
            bc = jnp.sum(node[cell, :], axis=1) / 3
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = jnp.zeros(node.shape[0], dtype=jnp.bool_)
            isValidNode = isValidNode.at[cell].set(True)
            node = node[isValidNode]
            idxMap = jnp.zeros(node.shape[0], dtype=jnp.int32)
            idxMap = idxMap.at[isValidNode].set(jnp.arange(isValidNode.sum()))
            cell = idxMap[cell.ravel()].reshape(cell.shape)

        return cls(node, cell)
