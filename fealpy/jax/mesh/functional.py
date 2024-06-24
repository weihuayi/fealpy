
from typing import Optional, Sequence, Union
from itertools import combinations_with_replacement
from functools import reduce, partial
from math import factorial, comb

import numpy as np
import jax
import jax.numpy as jnp

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
    raw[:, -1] = p
    raw[:, 1:-1] = sep
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
    TD = points.size(-2) - 1
    if TD != points.size(-1):
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
    """
    @brief 给定一组重心坐标点 `bc`, 计算单纯形单元上 `p` 次 Lagrange
    基函数在这一组重心坐标点处的函数值

    @param[in] bc : (TD+1, )
    @param[in] p : 基函数的次数，为正整数
    @param[in] mi : p 次的多重指标矩阵

    @return phi : 形状为 (NQ, ldof)
    """
    TD = bc.shape[-1] - 1
    c = jnp.arange(1, p+1)
    P = 1.0/jnp.cumprod(c)
    t = jnp.arange(0, p)
    A = p*bc - jnp.arange(0, p).reshape(-1, 1)
    A = P.reshape(-1, 1)*jnp.cumprod(A, axis=-2) # (p, TD+1)
    B = jnp.ones((p+1, TD+1), dtype=A.dtype)
    B = B.at[1:, :].set(A)
    idx = jnp.arange(TD+1)
    phi = jnp.prod(B[mi, idx], axis=-1)
    return phi


@partial(jax.jit, static_argnums=(2, ))
def simplex_shape_function(bcs, mi, p):
    fn = jax.vmap(_simplex_shape_function, in_axes=(0, None, None))
    return fn(bcs, mi, p)

def _simplex_grad_shape_function(bc, mi, p, n):
    fn = _simplex_shape_function
    for i in range(n):
        fn = jax.jacfwd(fn)
    return fn(bc, mi, p)


@partial(jax.jit, static_argnums=(2, 3))
def simplex_grad_shape_function(bcs, mi, p, n): 
    return jax.vmap(
            _simplex_grad_shape_function, 
            in_axes=(0, None, None, None)
            )(bcs, mi, p, n)


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
def tri_area_2d(points):
    """
    @brief 给定一个单元的三个顶点的坐标，计算三角形的面积
    """
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    nv = jnp.cross(v1, v2)
    return nv/2.0

def tri_area_3d(points):
    """
    @brief 给定一个单元的三个顶点的坐标，计算三角形的面积

    @params points : (3, 3) 
    """
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    nv = jnp.cross(v1, v2)
    a = jnp.linalg.norm(nv)/2.0
    return nv/2.0

def tri_area_2d_with_jac(points):
    return value_and_jacfwd(tri_area_2d, points)

def tri_area_3d_with_jac(points):
    return value_and_jacfwd(tri_area_3d, points)

def tri_quality_radius_ratio(points):
    v0 = points[2] - points[1]
    v1 = points[0] - points[2]
    v2 = points[1] - points[0]

    l0 = jnp.linalg.norm(v0)
    l1 = jnp.linalg.norm(v1)
    l2 = jnp.linalg.norm(v2)

    p = l0 + l1 + l2
    q = l0*l1*l2
    nv = np.cross(v1, v2)
    a = jnp.linalg.norm(nv)/2.0
    quality = p*q/(16*a**2)
    return quality

def tri_quality_radius_ratio_with_jac(points):
    return value_and_jacfwd(tri_quality_radius_ratio, points)

def tri_grad_lambda_2d(points):
    """
    @brief 计算2D三角形单元的形函数梯度 

    @params points : 形状为  (3, 2), 存储一个三角形单元的坐标，逆时针方向
    """
    v0 = points[2] - points[1]
    v1 = points[0] - points[2]
    v2 = points[1] - points[0]
    nv = jnp.cross(v1, v2) # 三角形面积的 2 倍 
    Dlambda = jnp.array([
        [-v0[1], v0[0]], 
        [-v1[1], v1[0]], 
        [-v2[1], v2[0]]], dtype=jnp.float64)/nv
    return Dlambda 

def tri_grad_lambda_3d(points):
    """
    @brief 计算 3D 三角形单元的形函数梯度 

    @params points : 形状为  (3, 3), 存储一个三角形单元的坐标(逆时针方向)
    """
    v0 = points[2] - points[1]
    v1 = points[0] - points[2]
    v2 = points[1] - points[0]
    nv = jnp.cross(v1, v2) # 三角形面积的 2 倍 
    length = jnp.linalg.norm(nv)
    n = nv/length
    n0 = jnp.cross(n, v0)
    n1 = jnp.cross(n, v1)
    n2 = jnp.cross(n, v2)
    Dlambda = jnp.array([n0, n1, n2], dtype=jnp.float64)/length
    return Dlambda 
