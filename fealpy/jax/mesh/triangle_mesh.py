from typing import Any, Callable, Optional, List, Union
import functools

import numpy as np
import jax
import jax.numpy as jnp

from fealpy import logger

from .triangle_mesh_kernel import *
from .mesh_base import MeshBase


class TriangleMeshDataStructure():
    localEdge = jnp.array([(1, 2), (2, 0), (0, 1)])
    localFace = jnp.array([(1, 2), (2, 0), (0, 1)])
    ccw = jnp.array([0, 1, 2])

    localCell = np.array([
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1)])

    def __init__(self, NN, cell):
        self.NN = NN
        self.cell = cell
        self.TD = 2


class TriangleMesh(MeshBase):
    def __init__(self, node, cell):
        """
        @brief TriangleMesh 对象的初始化函数

        """

        assert cell.shape[-1] == 3

        self.node = node
        NN = node.shape[0]
        GD = node.shape[1]
        self.ds = TriangleMeshDataStructure(NN, cell)

        self.shape_function = self._shape_function

        if GD == 2:
            self._cell_area = jax.jit(jax.vmap(tri_area_2d))
            self._cell_area_with_jac = jax.jit(jax.vmap(tri_area_2d_with_jac))
            self._grad_lambda = jax.jit(jax.vmap(tri_grad_lambda_2d))
        elif GD == 3:
            self._cell_area = jax.jit(jax.vmap(tri_area_3d))
            self._cell_area_with_jac = jax.jit(jax.vmap(tri_area_3d_with_jac))
            self._grad_lambda = jax.jit(jax.vmap(tri_grad_lambda_3d))

        self._quality = jax.jit(jax.vmap(tri_quality_radius_ratio))
        self._quality_with_jac = jax.jit(jax.vmap(tri_quality_radius_ratio_with_jac))

    def number_of_nodes(self):
        return len(self.node)

    def number_of_cells(self):
        return len(self.ds.cell)

    def number_of_local_ipoints(self, p, iptype='cell'):
        """
        @brief
        """
        if iptype in {'cell', 2}:
            return (p+1)*(p+2)//2
        elif iptype in {'face', 'edge',  1}: # 包括两个顶点
            return p + 1
        elif iptype in {'node', 0}:
            return 1

    def number_of_global_ipoints(self, p):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        return NN + (p-1)*NE + (p-2)*(p-1)//2*NC

    def grad_lambda(self, index=np.s_[:]):
        return self._grad_lambda(self.node[self.ds.cell[index]])

    def grad_shape_function(self, bc, p=1, index=np.s_[:], variables='x'):
        """
        @note 注意这里调用的实际上不是形状函数的梯度，而是网格空间基函数的梯度
        """
        R = self._grad_shape_function(bc, p)
        if variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = jnp.einsum('...ij, kjm->...kim', R, Dlambda, optimize=True)
            return gphi #(NQ, NC, ldof, GD)
        elif variables == 'u':
            return R #(NQ, ldof, TD+1)

    cell_grad_shape_function = grad_shape_function

    def cell_area(self, index=jnp.s_[:]):
        return self._cell_area(self.node[self.ds.cell[index]]) 

    def cell_area_with_jac(self, index=jnp.s_[:]):
        return self._cell_area_with_jac(self.node[self.ds.cell[index]]) 

    def cell_quality(self, index=jnp.s_[:]):
        return  self._quality(self.node[self.ds.cell[index]])

    def cell_quality_with_jac(self, index=jnp.s_[:]):
        return self._quality_with_jac(self.node[self.ds.cell[index]])

