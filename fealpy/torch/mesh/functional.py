
from typing import Optional
from itertools import combinations_with_replacement
from functools import reduce
from math import factorial

import numpy as np
import torch
from torch import Tensor, norm, det, cross


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
        points: Tensor(..., 2, GD).

    Returns:
        Tensor(...,).
    """
    return norm(points[..., 0, :] - points[..., 1, :], dim=-1, out=out)


### Simplex (2D Triangle, 3D Tetrahegron, 4D...) measure
def simplex_measure(points: Tensor):
    r"""
    Args:
        edges: Tensor(..., NVC, GD).
        out: Tensor(...,), optional.

    Returns:
        Tensor(...,).
    """
    TD = points.size(-2) - 1
    if TD != points.size(-1):
        raise RuntimeError("The geometric dimension of points must be NVC-1"
                           "to form a simplex.")
    edges = points[..., 1:, :] - points[..., :-1, :]
    return det(edges).div_(factorial(TD))


### Triangle
def tri_area_3d(points: Tensor, out: Optional[Tensor]=None):
    return cross(points[..., 1, :] - points[..., 0, :],
                 points[..., 2, :] - points[..., 0, :], dim=-1, out=out)


def tri_grad_lambda_2d(points: Tensor):
    r"""
    Args:
        points: Tensor(..., 3, 2).

    Returns:
        Tensor(..., 3, 2).
    """
    e0 = points[..., 2, :] - points[..., 1, :]
    e1 = points[..., 0, :] - points[..., 2, :]
    e2 = points[..., 1, :] - points[..., 0, :]
    nv = det(torch.stack([e0, e1], dim=-2))
    return torch.tensor([
        [-e0[1], e0[0]],
        [-e1[1], e1[0]],
        [-e2[1], e2[0]]
    ], dtype=points.dtype, device=points.device).div(nv)


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
