
from typing import Optional
from functools import partial

import numpy as np
import torch
from torch import Tensor, vmap
from torch.func import jacfwd, jacrev

# simplex
# @torch.jit.script
def _simplex_shape_function(bc: Tensor, p: int, mi: Tensor):
    """
    @brief `p`-order shape function values on these barycentry points.

    @param bc: Tensor(TD+1, )
    @param p: order of the shape function
    @param mi: p-order multi-index matrix

    @return phi: Tensor(ldof, )
    """
    TD = bc.shape[-1] - 1
    itype = torch.int
    device = bc.device
    shape = (1, TD+1)
    c = torch.arange(1, p+1, dtype=itype, device=device)
    P = 1.0 / torch.cumprod(c, dim=0)
    t = torch.arange(0, p, dtype=itype, device=device)
    Ap = p*bc.unsqueeze(-2) - t.reshape(-1, 1)
    Ap = torch.cumprod(Ap, dim=-2).clone()
    Ap = Ap.mul(P.reshape(-1, 1))
    A = torch.cat([torch.ones(shape, dtype=bc.dtype, device=device), Ap], dim=-2)
    idx = torch.arange(TD + 1, dtype=itype, device=device)
    phi = torch.prod(A[mi, idx], dim=-1)
    return phi


def simplex_shape_function(bcs: Tensor, p: int, mi: Tensor) -> Tensor:
    fn = vmap(
        partial(_simplex_shape_function, p=p, mi=mi)
    )
    return fn(bcs)


def simplex_grad_shape_function(bcs: Tensor, p: int, mi: Tensor) -> Tensor:
    fn = vmap(jacfwd(
        partial(_simplex_shape_function, p=p, mi=mi)
    ))
    return fn(bcs)


def simplex_hess_shape_function(bcs: Tensor, p: int, mi: Tensor) -> Tensor:
    fn = vmap(jacrev(jacfwd(
        partial(_simplex_shape_function, p=p, mi=mi)
    )))
    return fn(bcs)
