
from typing import Optional, Sequence, Union
from itertools import combinations_with_replacement
from functools import reduce, partial
from math import factorial, comb

import numpy as np
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from .utils import Array



##################################################
### Mesh
##################################################

def multi_index_matrix(p: int, etype: int, *, dtype=None) -> Array:
    r"""Create a multi-index matrix."""
    dtype = dtype or jnp.int_
    sep = jnp.flip(jnp.array(
        tuple(combinations_with_replacement(range(p+1), etype)),
        dtype=jnp.int_
    ), axis=0)
    raw = jnp.zeros((sep.shape[0], etype+2), dtype=jnp.int_)
    raw = raw.at[:, -1].set(p)
    raw = raw.at[:, 1:-1].set(sep)
    return jnp.array(raw[:, 1:] - raw[:, :-1]).astype(dtype)


### Leangth of edges
def edge_length(edge: Array, node: Array, out=None) -> Array:
    """Edge length.

    Parameters:
        edge (Tensor): Indices of nodes on the two ends of edges, shaped [..., 2].\n
        node (Tensor): Coordinates of nodes, shaped [N, GD].\n
        out (Tensor, optional): The output tensor. Defaults to None.

    Returns:
        Tensor: Length of edges, shaped [...].
    """
    points = node[edge, :]
    return jnp.linalg.norm(points[..., 0, :] - points[..., 1, :], axis=-1)


def edge_normal(edge: Array, node: Array, unit: bool=False, out=None) -> Array:
    """Edge normal for 2D meshes.

    Parameters:
        edge (Tensor): Indices of nodes on the two ends of edges, shaped [..., 2].\n
        node (Tensor): Coordinates of nodes, shaped [N, GD].\n
        unit (bool, optional): Whether to normalize the normal. Defaults to False.\n
        out (Tensor, optional): The output tensor. Defaults to None.

    Returns:
        Tensor: Normal of edges, shaped [..., GD].
    """
    points = node[edge, :]
    if points.shape[-1] != 2:
        raise ValueError("Only 2D meshes are supported.")
    edges = points[..., 1, :] - points[..., 0, :]
    if unit:
        edges /= jnp.linalg.norm(edges, axis=-1, keepdims=True)
    return jnp.stack([edges[..., 1], -edges[..., 0]], axis=-1, out=out)


def entity_barycenter(etn: Array, node: Array) -> Array:
    summary = etn@node
    count = etn@jnp.ones((node.shape[0], 1), dtype=node.dtype)
    return summary/count 


# #################################################
### Homogeneous Mesh
##################################################

def bc_tensor(bcs: Sequence[Array]) -> Array:
    num = len(bcs)
    NVC = reduce(lambda x, y: x * y.shape[-1], bcs, 1)
    desp1 = 'mnopq'
    desp2 = 'abcde'
    string = ", ".join([desp1[i]+desp2[i] for i in range(num)])
    string += " -> " + desp1[:num] + desp2[:num]
    return jnp.einsum(string, *bcs).reshape(-1, NVC)


def bc_to_points(bcs: Union[Array, Sequence[Array]], node: Array,
                 entity:Array, order: Optional[Array]) -> Array:
    r"""Barycentric coordinates to cartesian coordinates in homogeneous meshes."""
    if order is not None:
        entity = entity[:, order]
    points = node[entity, :]

    if not isinstance(bcs, Array):
        bcs = bc_tensor(bcs)
    return jnp.einsum('ijk, ...j -> ...ik', points, bcs)


def homo_entity_barycenter(entity: Array, node: Array) ->Array:
    r"""Entity barycenter in homogeneous meshes."""
    return jnp.mean(node[entity, :], axis=1)


# Interval & Triangle & Tetrahedron
# ================================

def simplex_ldof(p: int, iptype: int) -> int:
    r"""Number of local DoFs of a simplex."""
    if iptype == 0:
        return 1
    return comb(p + iptype, iptype)


def simplex_gdof(p: int, mesh) -> int:
    r"""Number of global DoFs of a mesh with simplex cells."""
    coef = 1
    count = mesh.node.shape[0]

    for i in range(1, mesh.TD + 1):
        coef = (coef * (p-i)) // i
        count += coef * mesh.entity(i).shape[0]
    return count


def simplex_measure(simplex: Array, node: Array) -> Array:
    """Entity measurement of a simplex.

    Parameters:
        simplex (Tensor[..., NVC]): Indices of vertices of the simplex.\n
        node (Tensor[N, GD]): Node coordinates.

    Returns:
        Tensor[...,].
    """
    points = node[simplex, :]
    TD = points.shape[0-2] - 1
    if TD != points.shape[-1]:
        raise RuntimeError("The geometric dimension of points must be NVC-1"
                           "to form a simplex.")
    edges = points[..., 1:, :] - points[..., :-1, :]
    return jnp.linalg.det(edges)/(factorial(TD))


def value_and_jacfwd(f, x):
    pushfwd = partial(jax.jvp, f, (x, ))
    basis = jnp.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
    y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
    return y, jac

# simplex 
def _simplex_shape_function(bc, mi, p):
    """`p`-order shape function values on these barycentry points.

    Parameters:
        bc (Tensor[TD+1, ]):
        p (inr): order of the shape function.
        mi (Tensor): p-order multi-index matrix.

    Returns:
        Tensor[ldof, ]: phi.
    """
    # TD = bc.shape[-1] - 1
    # itype = jnp.int_ # Choose the appropriate integer type
    # shape = (1, TD+1)
    # c = jnp.arange(1, p+1, dtype=itype)
    # P = 1.0 / jnp.cumprod(c, axis=0)
    # t = jnp.arange(0, p, dtype=itype)
    # Ap = p * jnp.expand_dims(bc, -2) - t[..., jnp.newaxis]
    # Ap = jnp.cumprod(Ap, axis=-2)
    # Ap = Ap * P[..., None]
    # A = jnp.concatenate([jnp.ones((1, TD+1), dtype=bc.dtype), Ap], axis=-2)
    # idx = jnp.arange(TD + 1, dtype=itype)
    # phi = jnp.prod(A[mi, idx], axis=-1)
    # print(phi.shape)
    # return phi
    if p == 1:
        return bc
    TD = bc.shape[-1] - 1
    if mi is None:
        mi = multi_index_matrix(p, TD)
    c = jnp.arange(1, p+1, dtype=jnp.int_)
    P = 1.0 / jnp.cumprod(c)
    t = jnp.arange(0, p)
    shape = bc.shape[:-1]+(p+1, TD+1)
    A = jnp.ones(shape, dtype=bc.dtype)
    A = A.at[..., 1:, :].set(p*bc[..., None, :] - t.reshape(-1, 1))
    A = jnp.cumprod(A, axis=-2)
    A = A.at[..., 1:, :].set(A[..., 1:, :] * P.reshape(-1, 1))
    idx = jnp.arange(TD+1)
    phi = jnp.prod(A[..., mi, idx], axis=-1)
    return phi



@partial(jax.jit, static_argnums=(2, ))
def simplex_shape_function(bcs, mi, p):
    fn = jax.vmap(_simplex_shape_function, in_axes=(0, None, None))
    return fn(bcs, mi, p)

def _simplex_diff_shape_function(bc, mi, p, n):
    fn = _simplex_shape_function
    for i in range(n):
        fn = jax.jacfwd(fn)
    return fn(bc, mi, p)


@partial(jax.jit, static_argnums=(2, 3))
def simplex_diff_shape_function(bcs, mi, p, n): 
    return jax.vmap(
            _simplex_diff_shape_function, 
            in_axes=(0, None, None, None)
            )(bcs, mi, p, n)

def simplex_grad_shape_function(bcs, mi, p): 
    # print(simplex_diff_shape_function(bcs, mi, p, n=1))
    return simplex_diff_shape_function(bcs, mi, p, n=1)

# Quadrangle & Hexahedron
# =======================

def tensor_ldof(p: int, iptype: int) -> int:
    r"""Number of local DoFs of a tensor shape."""
    return (p + 1) ** iptype


def tensor_gdof(p: int, mesh) -> int:
    r"""Number of global DoFs of a mesh with tensor cells."""
    coef = 1
    count = mesh.node.shape[0]

    for i in range(1, mesh.TD + 1):
        coef = coef * (p-i)
        count += coef * mesh.entity(i).shape[0]
    return count


##################################################
### Final Mesh
##################################################

# Interval Mesh
# =============

def int_grad_lambda(line: Array, node: Array) -> Array:
    """grad_lambda function on lines.

    Args:
        line (Tensor[..., 2]): Indices of vertices of lines.\n
        node (Tensor[N, GD]): Node coordinates.

    Returns:
        Tensor: grad lambda tensor shaped [..., 2, GD].
    """
    points = node[line, :]
    v = points[..., 1, :] - points[..., 0, :] # (NC, GD)
    h2 = jnp.sum(v**2, axis=-1, keepdim=True)
    v /= h2
    return jnp.stack([-v, v], axis=-2)


# Triangle Mesh
# =============
def tri_area_3d(tri: Array, node: Array, out: Optional[Array]=None) -> Array:
    points = node[tri, :]
    return jnp.cross(points[..., 1, :] - points[..., 0, :],
                 points[..., 2, :] - points[..., 0, :], axis=-1) / 2.0


def tri_grad_lambda_2d(tri: Array, node: Array) -> Array:
    """grad_lambda function for the triangle mesh in 2D.

    Parameters:
        tri (Tensor[..., 3]): Indices of vertices of triangles.\n
        node (Tensor[N, 2]): Node coordinates.

    Returns:
        Tensor[..., 3, 2]:
    """
    points = node[tri, :]
    e0 = points[..., 2, :] - points[..., 1, :]
    e1 = points[..., 0, :] - points[..., 2, :]
    e2 = points[..., 1, :] - points[..., 0, :]
    nv = jnp.linalg.det(jnp.stack([e0, e1], axis=-2)) # (...)
    e0 = jnp.flip(e0, axis=-1)
    e1 = jnp.flip(e1, axis=-1)
    e2 = jnp.flip(e2, axis=-1)
    result = jnp.stack([e0, e1, e2], axis=-2)
    result = result.at[..., 0].mul(-1)
    return result/(nv[..., None, None])

def tri_grad_lambda_3d(tri: Array, node: Array) -> Array:
    """
    Parameters:
        points (Tensor[..., 3]): Indices of vertices of triangles.\n
        node (Tensor[N, 3]): Node coordinates.

    Returns:
        Tensor[..., 3, 3]:
    """
    points = node[tri, :]
    e0 = points[..., 2, :] - points[..., 1, :] # (..., 3)
    e1 = points[..., 0, :] - points[..., 2, :]
    e2 = points[..., 1, :] - points[..., 0, :]
    nv = jnp.cross(e0, e1, axis=-1) # (..., 3)
    length = jnp.linalg.norm(nv, axis=-1, keepdims=True) # (..., 1)
    n = nv/length
    return jnp.stack([
        jnp.cross(n, e0, axis=-1),
        jnp.cross(n, e1, axis=-1),
        jnp.cross(n, e2, axis=-1)
    ], axis=-2)/(length.unsqueeze(-2)) # (..., 3, 3)
