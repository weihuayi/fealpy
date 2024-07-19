
from typing import Union, Optional, Sequence
from functools import reduce

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import det

from math import factorial, comb


##################################################
### Mesh
##################################################

def multi_index_matrix(p: int, TD: int, dtype=np.int_) -> NDArray:
    """
    Create a multi-index matrix. 

    return: multiIndex  ndarray with shape (ldof, TD+1)
    """
    if TD == 3:
        ldof = (p+1)*(p+2)*(p+3)//6
        idx = np.arange(1, ldof)
        idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
        idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
        idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
        idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
        multiIndex = np.zeros((ldof, 4), dtype=dtype)
        multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
        multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
        multiIndex[1:, 1] = idx0 - idx2
        multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
        return multiIndex
    elif TD == 2:
        ldof = (p+1)*(p+2)//2
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 3), dtype=dtype)
        multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
        multiIndex[:,1] = idx0 - multiIndex[:,2]
        multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        return multiIndex
    elif TD == 1:
        ldof = p+1
        multiIndex = np.zeros((ldof, 2), dtype=dtype)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

def simplex_shape_function(bc: NDArray, p: int =1, mi: NDArray=None, dtype=np.int_):
    """
    Create simple shape function.

    Parameters:
        bc:NDArray(TD+1, )
        p: order of the simple shape function.
        mi: p-order multi-index matrix

    Return: phi
    """
    if p == 1:
        return bc
    TD = bc.shape[-1] - 1
    if mi is None:
        mi = multi_index_matrix(p, TD)
    c = np.arange(1, p+1, dtype=np.int_)
    P = 1.0/np.multiply.accumulate(c)
    t = np.arange(0, p)
    shape = bc.shape[:-1]+(p+1, TD+1)
    A = np.ones(shape, dtype=dtype)
    A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)
    np.cumprod(A, axis=-2, out=A)
    A[..., 1:, :] *= P.reshape(-1, 1)
    idx = np.arange(TD+1)
    phi = np.prod(A[..., mi, idx], axis=-1)
    return phi


def simplex_grad_shape_function(bc: NDArray, p: int =1, mi: NDArray=None, dtype=np.int_) -> NDArray:
    """
    Create simple grad shape function.

    """
    TD = bc.shape[-1] - 1
    if mi is not None:
        mi = multi_index_matrix(p, TD)
    ldof = mi.shape[0] # p 次 Lagrange 形函数的个数

    c = np.arange(1, p+1)
    P = 1.0/np.multiply.accumulate(c)

    t = np.arange(0, p)
    shape = bc.shape[:-1]+(p+1, TD+1)
    A = np.ones(shape, dtype=dtype)
    A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)

    FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
    FF[..., range(p), range(p)] = p
    np.cumprod(FF, axis=-2, out=FF)
    F = np.zeros(shape, dtype=dtype)
    F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
    F[..., 1:, :] *= P.reshape(-1, 1)

    np.cumprod(A, axis=-2, out=A)
    A[..., 1:, :] *= P.reshape(-1, 1)

    Q = A[..., mi, range(TD+1)]
    M = F[..., mi, range(TD+1)]

    shape = bc.shape[:-1]+(ldof, TD+1)
    R = np.zeros(shape, dtype=dtype)
    for i in range(TD+1):
        idx = list(range(TD+1))
        idx.remove(i)
        R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
    return R # (..., ldof, TD+1)

### Length of edges
def edge_length(points: NDArray, out=None) -> NDArray:
    """Edge length.

    Parameters:
        points (Tensor): Coordinates of points in two ends of edges, shaped [..., 2, GD].\n
        out (Tensor, optional): The output tensor. Defaults to None.

    Returns:
        Tensor: Length of edges, shaped [...].
    """
    return np.linalg.norm(points[..., 0, :] - points[..., 1, :], axis=-1)

def edge_normal(points: NDArray, unit: bool=False, out=None) -> NDArray:
    """Edge normal for 2D meshes.

    Parameters:
        points (Tensor): Coordinates of points in two ends of edges, shaped [..., 2, GD].\n
        unit (bool, optional): Whether to normalize the normal. Defaults to False.\n
        out (Tensor, optional): The output tensor. Defaults to None.

    Returns:
        Tensor: Normal of edges, shaped [..., GD].
    """
    if points.shape[-1] != 2:
        raise ValueError("Only 2D meshes are supported.")
    edges = points[..., 1, :] - points[..., 0, :]
    if unit:
        edges /= np.linalg.norm(edges, axis=-1, keepdims=True)
    return np.stack([edges[..., 1], -edges[..., 0]], axis=-1)

def edge_tangent(edge: NDArray, node: NDArray, 
        normalize: bool=False) -> NDArray:
    v = node[edge[:, 1], :] - node[edge[:, 0], :]
    if normalize:
        l = np.linalg.norm(v, axis=-1)
        return v/l[:, None]
    return v

def entity_barycenter(etn: NDArray, node: NDArray) -> NDArray:
    summary = etn@node
    count = etn@np.ones((node.shape[0], 1), dtype=node.dtype)
    return summary/count

##################################################
### Homogeneous Mesh
##################################################

def bc_tensor(bcs: Sequence[NDArray]) -> NDArray:
    num = len(bcs)
    NVC = reduce(lambda x, y: x * y.shape[-1], bcs, 1)
    desp1 = 'mnopq'
    desp2 = 'abcde'
    string = ", ".join([desp1[i]+desp2[i] for i in range(num)])
    string += " -> " + desp1[:num] + desp2[:num]
    return np.einsum(string, *bcs).reshape(-1, NVC)


def bc_to_points(bcs: Union[NDArray, Sequence[NDArray]], node: NDArray,
                 entity: NDArray, order: Optional[NDArray]) -> NDArray:
    r"""Barycentric coordinates to cartesian coordinates in homogeneous meshes."""
    if order is not None:
        entity = entity[:, order]
    points = node[entity, :]

    if not isinstance(bcs, np.ndarray):
        bcs = bc_tensor(bcs)
    return np.einsum('ijk, ...j -> i...k', points, bcs)


def homo_entity_barycenter(entity: NDArray, node: NDArray) -> NDArray:
    r"""Entity barycenter in homogeneous meshes."""
    return np.mean(node[entity, :], axis=1)


# Interval  & Triangle  & Tetrahedron 
# ================================================

def simplex_ldof(p: int, iptype: int) -> int:
    """
    Number of local DoFs of a simplex.
    """
    if iptype == 0:
        return 1
    return comb(p + iptype, iptype)


def simplex_gdof(p: int, mesh) -> int:
    """
    Number of global DoFs of a mesh with simplex cells.
    """
    coef = 1
    count = mesh.node.shape[0]

    for i in range(1, mesh.TD + 1):
        coef = (coef * (p-i)) // i
        count += coef * mesh.entity(i).shape[0]
    return count


def simplex_measure(points: NDArray):
    """
    Entity measurement of a simplex.

    Args:
        points: Tensor(..., NVC, GD).
        out: Tensor(...,), optional.

    Returns:
        Tensor(...,).
    """
    TD = points.shape[-2] - 1
    if TD != points.shape[-1]:
        raise RuntimeError("The geometric dimension of points must be NVC-1"
                           "to form a simplex.")
    edges = points[..., 1:, :] - points[..., :-1, :]
    return det(edges)/(factorial(TD))


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

def int_grad_lambda(line: NDArray, node: NDArray) -> NDArray:
    """grad_lambda function on lines.

    Args:
        line (Tensor[..., 2]): Indices of vertices of lines.\n
        node (Tensor[N, GD]): Node coordinates.

    Returns:
        Tensor: grad lambda tensor shaped [..., 2, GD].
    """
    points = node[line, :]
    v = points[..., 1, :] - points[..., 0, :] # (NC, GD)
    h2 = np.sum(v**2, axis=-1, keepdims=True)
    v /= h2
    return np.stack([-v, v], axis=-2)


# Triangle Mesh
# =============

def tri_area_3d(points, out=None):
    """
    Calculate the area of triangles in 3D space given their vertices.
    
    Args:
        points: np.ndarray of shape (..., 3, 3) representing the vertices of triangles.
        out: Optional np.ndarray to store the results.
        
    Returns:
        np.ndarray containing the area of each triangle.
    """
    edge1 = points[..., 1, :] - points[..., 0, :]
    edge2 = points[..., 2, :] - points[..., 0, :]
    cross_product = np.cross(edge1, edge2, axis=-1)
    area = 0.5 * np.linalg.norm(cross_product, axis=-1)
    if out is not None:
        out[:] = area
        return out
    return area

def tri_grad_lambda_2d(points):
    """
    Gradient lambda function for a triangle mesh in 2D.
    
    Args:
        points: np.ndarray of shape (..., 3, 2) representing the vertices of triangles.
        
    Returns:
        np.ndarray of shape (..., 3, 2) representing the gradients.
    """
    e0 = points[..., 2, :] - points[..., 1, :]
    e1 = points[..., 0, :] - points[..., 2, :]
    e2 = points[..., 1, :] - points[..., 0, :]
    nv = det(np.stack([e0, e1], axis=-2)) # Determinant for 2D case, equivalent to np.linalg.det for 2x2 matrix
    e0 = np.flip(e0, axis=-1)
    e1 = np.flip(e1, axis=-1)
    e2 = np.flip(e2, axis=-1)
    result = np.stack([e0, e1, e2], axis=-2)
    result[..., 0] *= -1
    return result / np.expand_dims(nv, axis=(-1, -2))

def tri_grad_lambda_3d(points):
    """
    Gradient lambda function for a triangle mesh in 3D.
    
    Args:
        points: np.ndarray of shape (..., 3, 3) representing the vertices of triangles.
        
    Returns:
        np.ndarray of shape (..., 3, 3) representing the gradients.
    """
    e0 = points[..., 2, :] - points[..., 1, :]  # (..., 3)
    e1 = points[..., 0, :] - points[..., 2, :]
    e2 = points[..., 1, :] - points[..., 0, :]
    nv = np.cross(e0, e1, axis=-1)  # Normal vector, (..., 3)
    length = np.linalg.norm(nv, axis=-1, keepdims=True)  # Length of normal vector, (..., 1)
    n = nv / length  # Unit normal vector
    # Compute the gradient by crossing the unit normal with each edge
    return np.stack([
        np.cross(n, e0, axis=-1),
        np.cross(n, e1, axis=-1),
        np.cross(n, e2, axis=-1)
    ], axis=-2) / length[..., np.newaxis, np.newaxis]  # Scale by inverse length to normalize
