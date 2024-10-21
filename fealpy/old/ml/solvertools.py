
from typing import Tuple
import torch
from torch import Tensor


def rescale(A_: Tensor, b_: Tensor, amplify=1.0, *, eps=1e-4, inplace=True)\
    -> Tuple[Tensor, Tensor]:
    """
    @brief Apply rescaling to linear equations. For each row of the equations,\
           coefficients will be rescaled by their maximum. Such that\n
           $$A_{ij}^* = amp * A_{ij} / (\max_j |A_{ij}| + eps),$$\n
           $$b_{i}^* = amp * b_{i} / (\max_j |A_{ij}| + eps).$$

    @param A_: Tensor.
    @param b_: Tensor.
    @param amplify: float.
    @param eps: float.
    @param inplace: bool.

    @return: Tensor.
    @example:
    ```
    >>> from fealpy.ml import solvertools as ST
    >>> A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
    >>> b = torch.tensor([5, 6], dtype=torch.float64)
    >>> ST.rescale(A, b)
    (tensor([[0.5000, 1.0000],
             [0.7500, 1.0000]], dtype=torch.float64),
     tensor([2.4999, 1.5000], dtype=torch.float64))
    ```
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


def ridge(matrix: Tensor, lambda_: float, *, inplace=True) -> Tensor:
    """
    @brief Apply to X^TX the preperation of ridge regression.\n
           $$(X^TX + \\lambda I) \\beta = X^Ty$$

    @param matrix: Tensor. The data matrix $X^TX$.
    @param inplace: bool.

    @return: Tensor.
    @example:
    ```
    >>> from fealpy.ml import solvertools as ST
    >>> A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
    >>> ST.ridge(A, 0.1)
    tensor([[1.1000, 2.0000],
            [3.0000, 4.1000]], dtype=torch.float64)
    ```
    """
    A_ = matrix
    assert A_.ndim == 2
    I = torch.eye(A_.shape[0], A_.shape[1], dtype=A_.dtype, device=A_.device)
    if inplace:
        A_ += I * lambda_
        return A_
    else:
        return A_ + I * lambda_


def lasso(matrix: Tensor, lambda_: float, *, inplace=True) -> Tensor:
    pass
