
from typing import Tuple
import torch
from torch import Tensor


def rescale(A_: Tensor, b_: Tensor, amplify=1.0, *, eps=1e-4, inplace=True)\
    -> Tuple[Tensor, Tensor]:
    """
    @brief Apply rescaling to linear equations.

    @param A_: Tensor.
    @param b_: Tensor.
    @param amplify: float.
    @param eps: float.
    @param inplace: bool.

    @return: Tensor.
    """
    assert A_.ndim == 2
    scale = A_.abs().max(dim=-1, keepdim=True)[0] + eps
    ratio: Tensor = amplify / scale
    if inplace:
        A_ *= ratio
        b_ *= ratio.reshape(b_.shape)
        return A_, b_
    else:
        return A_ * ratio, b_ * ratio.reshape(b_.shape)


def ridge(matrix: Tensor, lambda_: float, *, inplace=True):
    """
    @brief Apply to X^TX the preperation of ridge regression.

    $$(X^TX + \\lambda I) \\beta = X^Ty$$

    @param matrix: Tensor. The data matrix $X^TX$.
    @param inplace: bool.

    @return: Tensor.
    """
    A_ = matrix
    assert A_.ndim == 2
    I = torch.eye(A_.shape[0], A_.shape[1], dtype=A_.dtype, device=A_.device)
    if inplace:
        A_ += I * lambda_
        return A_
    else:
        return A_ + I * lambda_
