
from typing import Optional, Sequence, Union
from itertools import combinations_with_replacement
from functools import reduce
from math import factorial, comb

import numpy as np
import torch
from torch import Tensor, norm, det, cross


##################################################
### Mesh
##################################################

def multi_index_matrix(p: int, etype: int, *, dtype=None, device=None) -> Tensor:
    r"""Create a multi-index matrix."""
    dtype = dtype or torch.int
    kwargs = {'dtype': dtype, 'device': device}
    sep = np.flip(np.array(
        tuple(combinations_with_replacement(range(p+1), etype)),
        dtype=np.int_
    ), axis=0)
    raw = np.zeros((sep.shape[0], etype+2), dtype=np.int_)
    raw[:, -1] = p
    raw[:, 1:-1] = sep
    return torch.from_numpy(raw[:, 1:] - raw[:, :-1]).to(**kwargs)


def shape_function(bc: Tensor, p: int=1, mi: Optional[Tensor]=None, *,
                   dtype=None, device=None) -> Tensor:
    r"""Shape function"""
    if p <= 0:
        raise ValueError("p must be a positive integer.")
    if p == 1:
        return bc
    TD = bc.shape[-1] - 1
    itype = torch.int
    dtype = dtype or bc.dtype
    shape = bc.shape[:-1] + (p+1, TD+1)
    mi = mi or multi_index_matrix(p, etype=TD, dtype=itype, device=device)
    c = torch.arange(1, p+1, dtype=itype, device=device)
    P = 1.0 / torch.cumprod(c, dim=0)
    t = torch.arange(0, p, dtype=itype, device=device)
    A = torch.ones(shape, dtype=dtype, device=device)
    torch.sub(p*bc.unsqueeze(-2), t.reshape(-1, 1), out=A[..., 1:, :])
    A = torch.cumprod(A, dim=-2).clone()
    A[..., 1:, :].mul_(P.reshape(-1, 1))
    idx = torch.arange(TD + 1, dtype=itype, device=device)
    phi = torch.prod(A[..., mi, idx], dim=-1)
    return phi


### Leangth of edges
def edge_length(points: Tensor, out=None) -> Tensor:
    r"""Edge length.

    Args:
        points (Tensor): Coordinates of points in two ends of edges, shaped [..., 2, GD].
        out (Tensor, optional): The output tensor. Defaults to None.

    Returns:
        Tensor: Length of edges, shaped [...].
    """
    return norm(points[..., 0, :] - points[..., 1, :], dim=-1, out=out)


def edge_normal(points: Tensor, unit: bool=False, out=None) -> Tensor:
    """Edge normal for 2D meshes.

    Args:
        points (Tensor): Coordinates of points in two ends of edges, shaped [..., 2, GD].
        unit (bool, optional): Whether to normalize the normal. Defaults to False.
        out (Tensor, optional): The output tensor. Defaults to None.

    Returns:
        Tensor: Normal of edges, shaped [..., GD].
    """
    if points.shape[-1] != 2:
        raise ValueError("Only 2D meshes are supported.")
    edges = points[..., 1, :] - points[..., 0, :]
    if unit:
        edges = edges.div_(norm(edges, dim=-1, keepdim=True))
    return torch.stack([edges[..., 1], -edges[..., 0]], dim=-1, out=out)


def entity_barycenter(etn: Tensor, node: Tensor) -> Tensor:
    summary = etn@node
    count = etn@torch.ones((node.size(0), 1), dtype=node.dtype)
    return summary.div_(count)


##################################################
### Homogeneous Mesh
##################################################

def bc_tensor(bcs: Sequence[Tensor]):
    num = len(bcs)
    NVC = reduce(lambda x, y: x * y.shape[-1], bcs, 1)
    desp1 = 'mnopq'
    desp2 = 'abcde'
    string = ", ".join([desp1[i]+desp2[i] for i in range(num)])
    string += " -> " + desp1[:num] + desp2[:num]
    return torch.einsum(string, *bcs).reshape(-1, NVC)


def bc_to_points(bcs: Union[Tensor, Sequence[Tensor]], node: Tensor,
                 entity: Tensor, order: Optional[Tensor]) -> Tensor:
    r"""Barycentric coordinates to cartesian coordinates in homogeneous meshes."""
    if order is not None:
        entity = entity[:, order]
    points = node[entity, :]

    if not isinstance(bcs, Tensor):
        bcs = bc_tensor(bcs)
    return torch.einsum('ijk, ...j -> ...ik', points, bcs)


def homo_entity_barycenter(entity: Tensor, node: Tensor):
    r"""Entity barycenter in homogeneous meshes."""
    return torch.mean(node[entity, :], dim=1)


# Interval Mesh & Triangle Mesh & Tetrahedron Mesh
# ================================================

def simplex_ldof(p: int, iptype: int) -> int:
    r"""Number of local DoFs of a simplex."""
    if iptype == 0:
        return 1
    return comb(p + iptype, iptype)


def simplex_gdof(p: int, mesh) -> int:
    r"""Number of global DoFs of a mesh with simplex cells."""
    coef = 1
    count = mesh.node.size(0)

    for i in range(1, mesh.TD + 1):
        coef = (coef * (p-i)) // i
        count += coef * mesh.entity(i).size(0)
    return count


def simplex_measure(points: Tensor):
    r"""Entity measurement of a simplex.

    Args:
        points: Tensor(..., NVC, GD).
        out: Tensor(...,), optional.

    Returns:
        Tensor(...,).
    """
    TD = points.size(-2) - 1
    if TD != points.size(-1):
        raise RuntimeError("The geometric dimension of points must be NVC-1"
                           "to form a simplex.")
    edges = points[..., 1:, :] - points[..., :-1, :]
    return det(edges).div(factorial(TD))


# Quadrangle Mesh & Hexahedron Mesh
# =================================


##################################################
### Final Mesh
##################################################

# Interval Mesh
# =============

def int_grad_lambda(points: Tensor):
    """grad_lambda function for the interval mesh.

    Args:
        points (Tensor[..., 2, GD]): _description_

    Returns:
        Tensor: grad lambda tensor shaped [..., 2, GD].
    """
    v = points[..., 1, :] - points[..., 0, :] # (NC, GD)
    h2 = torch.sum(v**2, dim=-1, keepdim=True)
    v = v.div(h2)
    return torch.stack([-v, v], dim=-2)

# Triangle Mesh
# =============

def tri_area_3d(points: Tensor, out: Optional[Tensor]=None):
    return cross(points[..., 1, :] - points[..., 0, :],
                 points[..., 2, :] - points[..., 0, :], dim=-1, out=out)


def tri_grad_lambda_2d(points: Tensor):
    """grad_lambda function for the triangle mesh in 2D.
    Args:
        points: Tensor(..., 3, 2).

    Returns:
        Tensor(..., 3, 2).
    """
    e0 = points[..., 2, :] - points[..., 1, :]
    e1 = points[..., 0, :] - points[..., 2, :]
    e2 = points[..., 1, :] - points[..., 0, :]
    nv = det(torch.stack([e0, e1], dim=-2)) # (...)
    e0 = e0.flip(-1)
    e1 = e1.flip(-1)
    e2 = e2.flip(-1)
    result = torch.stack([e0, e1, e2], dim=-2)
    result[..., 0].mul_(-1)
    return result.div_(nv[..., None, None])

def tri_grad_lambda_3d(points: Tensor):
    r"""
    Args:
        points: Tensor(..., 3, 3).

    Returns:
        Tensor(..., 3, 3).
    """
    e0 = points[..., 2, :] - points[..., 1, :] # (..., 3)
    e1 = points[..., 0, :] - points[..., 2, :]
    e2 = points[..., 1, :] - points[..., 0, :]
    nv = cross(e0, e1, dim=-1) # (..., 3)
    length = norm(nv, dim=-1, keepdim=True) # (..., 1)
    n = nv.div_(length)
    return torch.stack([
        cross(n, e0, dim=-1),
        cross(n, e1, dim=-1),
        cross(n, e2, dim=-1)
    ], dim=-2).div_(length.unsqueeze(-2)) # (..., 3, 3)
