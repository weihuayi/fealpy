
import torch
from torch import Tensor, Size


from .tensor import LazyCOOMatrix


def spmm(A: LazyCOOMatrix, b: Tensor) -> Tensor:
    r"""Sparse matrix and dense vector multiplication."""
    source = b[..., A.col] * A.data

def spquad(l: Tensor, A: LazyCOOMatrix, r: Tensor) -> Tensor:
    r"""Quadratric form of sparse matrix and two dense vectors."""
    source = r[..., A.col] * l[..., A.row]
    source.mul_(A.data)
    return source.sum(dim=-1)
