from typing import Optional, Union, Tuple, Any
from itertools import combinations_with_replacement
from functools import reduce, partial
from math import factorial
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import config, jit

    config.update("jax_enable_x64", True)

except ImportError:
    raise ImportError("Name 'jax' cannot be imported. "
                      'Make sure JAX is installed before using '
                      'the JAX backend in FEALPy. '
                      'See https://github.com/google/jax for installation.')

from .base import Backend, ATTRIBUTE_MAPPING, FUNCTION_MAPPING

Array = jax.Array 
_device = jax.Device

class JAXBackend(Backend[Array], backend_name='jax'):
    DATA_CLASS = Array 

    @staticmethod
    def set_default_device(device: Union[str, _device]) -> None:
        jax.default_device = device 
    
    @staticmethod
    def to_numpy(jax_array: Array, /) -> Any:
        return np.array(jax_array) 

    @staticmethod
    def from_numpy(numpy_array: np.ndarray, /) -> Any:
        """
        
        TODO:
            1. add support to `device` agument
        """
        return jax.device_put(numpy_array)

    ### Tensor creation methods ###
    # NOTE: all copied

    ### Reduction methods ###
    # NOTE: all copied

    ### Unary methods ###
    # NOTE: all copied

    ### Binary methods ###
    # NOTE: all copied

    ### FEALPy functionals ###
    @staticmethod
    def multi_index_matrix(p: int, dim: int, *, dtype=None) -> Array:
        r"""Create a multi-index matrix."""
        dtype = dtype or jnp.int_
        sep = jnp.flip(jnp.array(
            tuple(combinations_with_replacement(range(p+1), dim)),
            dtype=jnp.int_
        ), axis=0)
        z = jnp.zeros((sep.shape[0], 1))
        p = jnp.broadcast_to(p, shape=(sep.shape[0], 1))
        raw = jnp.concatenate((z, sep, p), axis=1)
        return jnp.array(raw[:, 1:] - raw[:, :-1]).astype(dtype)
    
    @staticmethod
    @jit
    def edge_length(edge: Array, node: Array, *, out=None) -> Array:
        return jnp.linalg.norm(node[edge[:, 0]] - node[edge[:, 1]], axis=-1)

    @staticmethod
    @jit
    def edge_normal(edge: Array, node: Array, unit=False, *, out=None) -> Array:
        points = node[edge, :]
        if points.shape[-1] != 2:
            raise ValueError("Only 2D meshes are supported.")
        edges = points[..., 1, :] - points[..., 0, :]
        if [unit]:
            edges /= jnp.linalg.norm(edges, axis=-1, keepdims=True)
        return jnp.stack([edges[..., 1], -edges[...,0]], axis=-1, out=out)    

    @staticmethod
    @jit
    def edge_tangent(edge: Array, node: Array, unit=False, *, out=None) -> Array:
        edges = node[edge[:, 1], :] - node[edge[:, 0], :]
        if [unit]:
            l = jnp.linalg.norm(edges, axis=-1, keepdims=True)
            edges /= l
        return jnp.stack([edges[..., 0], edges[...,1]], axis=-1, out=out)    
    
    @staticmethod
    @jit
    def simplex_measure(entity: Array, node: Array) -> Array:
        points = node[entity, :]
        TD = points.shape[-2] - 1
        if TD != points.size(-1):
            raise RuntimeError("The geometric dimension of points must be NVC-1"
                            "to form a simplex.")
        edges = points[..., 1:, :] - points[..., :-1, :]
        return jnp.linalg.det(edges)/jax.scipy.special.factorial(TD)
    
    @staticmethod
    @jit
    def barycenter(entity: Array, node: Array, loc: Optional[Array]=None) -> Array:
        return jnp.mean(node[entity, :], axis=1) # TODO: polygon mesh case
    
    # Triangle Mesh
    # =============

    #TODO:out
    @staticmethod
    @jit
    def triangle_area_3d(tri: Array, node: Array, out: Optional[Array]=None) -> Array:
        points = node[tri, :]
        return jnp.cross(points[..., 1, :] - points[..., 0, :],
                    points[..., 2, :] - points[..., 0, :], axis=-1) / 2.0
    
    @staticmethod
    def tri_grad_lambda_2d(cell: Array, node: Array) -> Array:
        """grad_lambda function for the triangle mesh in 2D.

        Parameters:
            cell (Tensor[NC, 3]): Indices of vertices of triangles.\n
            node (Tensor[NN, 2]): Node coordinates.

        Returns:
            Tensor[..., 3, 2]:
        """
        e0 = node[cell[:, 2]] - node[cell[:, 1]]
        e1 = node[cell[:, 0]] - node[cell[:, 2]]
        e2 = node[cell[:, 1]] - node[cell[:, 0]]
        nv = jnp.linalg.det(jnp.stack([e0, e1], axis=-2)) # (...)
        e0 = jnp.flip(e0, axis=-1)
        e1 = jnp.flip(e1, axis=-1)
        e2 = jnp.flip(e2, axis=-1)
        result = jnp.stack([e0, e1, e2], axis=-2)
        result = result.at[..., 0].mul(-1)
        return result/(nv[..., None, None])

    @staticmethod
    def triangle_grad_lambda_3d(cell: Array, node: Array) -> Array:
        """
        Parameters:
            cell (Tensor[NC, 3]): Indices of vertices of triangles.\n
            node (Tensor[NN, 3]): Node coordinates.

        Returns:
            Tensor[..., 3, 3]:
        """
        e0 = node[cell[:, 2]] - node[cell[:, 1]]
        e1 = node[cell[:, 0]] - node[cell[:, 2]]
        e2 = node[cell[:, 1]] - node[cell[:, 0]]
        nv = jnp.cross(e0, e1, axis=-1) # (..., 3)
        length = jnp.linalg.norm(nv, axis=-1, keepdims=True) # (..., 1)
        n = nv/length
        return jnp.stack([
            jnp.cross(n, e0, axis=-1),
            jnp.cross(n, e1, axis=-1),
            jnp.cross(n, e2, axis=-1)
        ], axis=-2)/(length.unsqueeze(-2)) # (..., 3, 3)

JAXBackend.attach_attributes(ATTRIBUTE_MAPPING, jnp)
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(tensor='array')
JAXBackend.attach_methods(function_mapping, jnp)
