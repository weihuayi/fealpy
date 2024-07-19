
import builtins
from typing import Union, Callable

import numpy as np
from numpy.typing import NDArray

from .utils import is_scalar, is_tensor, get_coef_subscripts


Number = Union[builtins.int, builtins.float]
CoefLike = Union[Number, NDArray, Callable[..., NDArray]]


def integral(value: NDArray, weights: NDArray, measure: NDArray, *,
             entity_type=False) -> NDArray:
    """Numerical integration.

    Args:
        value (Tensor[..., Q, C]): The values on the quadrature points to be integrated.
        weights (Tensor[Q,]): The weights of the quadrature points.
        measure (Tensor[C,]): The measure of the quadrature points.
        entity_type (bool): Whether to return integration in each entity. Defaults to False.

    Returns:
        Tensor[...]: The result of the integration. The shape will be [..., C]\
        if entity_type is True, otherwise [...].
    """
    subs = '...c' if entity_type else '...'
    return np.einsum(f'q, c, ...qc -> {subs}', weights, measure, value)


def linear_integral(input: NDArray, weights: NDArray, measure: NDArray,
                    coef: Union[Number, NDArray, None]=None) -> NDArray:
    r"""Numerical integration.

    Args:
        input (Tensor[Q, C, I]): The values on the quadrature points to be integrated.
        weights (Tensor[Q,]): The weights of the quadrature points.
        measure (Tensor[C,]): The measure of the quadrature points.
        coef (Number, Tensor, optional): The coefficient of the integration. Defaults to None.
        Must be int, float, Tensor, or callable returning Tensor with shape (Q,), (C,) or (Q, C).
        If `batched == True`, the shape of the coef should be (B, Q, C) or (B, C).
        batched (bool, optional): Whether the coef are batched. Defaults to False.

    Returns:
        Tensor[C, I]: The result of the integration.
        If `batched == True`, the shape of the result is (B, C, I).
    """
    if coef is None:
        return np.einsum('q, c, qci -> ci', weights, measure, input)

    NQ = weights.shape[0]
    NC = measure.shape[0]
    if is_scalar(coef):
        return np.einsum('q, c, qci -> ci', weights, measure, input) * coef
    elif is_tensor(coef):
        out_subs = 'ci'
        subs = get_coef_subscripts(coef.shape, NQ, NC)
        return np.einsum(f'q, c, qci, {subs} -> {out_subs}', weights, measure, input, coef)
    else:
        raise TypeError(f"coef should be int, float or Tensor, but got {type(coef)}.")


def bilinear_integral(input1: NDArray, input2: NDArray, weights: NDArray, measure: NDArray,
                      term: NDArray,
                      coef: Union[Number, NDArray, None]=None) -> NDArray:
    r"""Numerical integration.

    Args:
        input1 (Tensor[Q, C, I, ...]): The values on the quadrature points to be integrated.
        input2 (Tensor[Q, C, J, ...]): The values on the quadrature points to be integrated.
        weights (Tensor[Q,]): The weights of the quadrature points.
        measure (Tensor[C,]): The measure of the quadrature points.
        coef (Number, Tensor, optional): The coefficient of the integration. Defaults to None.
        Must be int, float, Tensor, or callable returning Tensor with shape (Q,), (C,) or (Q, C).
        If `batched == True`, the shape of the coef should be (B, Q, C) or (B, C).
        batched (bool, optional): Whether the coef are batched. Defaults to False.

    Returns:
        Tensor[C, I, J]: The result of the integration.
        If `batched == True`, the shape of the output is (B, C, I, J).
    """
    if len(input1.shape)==5:
        input1sub = 'qcidl'
        input2sub = 'qcjdl'
    
    elif len(input1.shape)==4:
        input1sub = 'qcid'
        input2sub = 'qcjd'
    else:
        input1sub = 'qci'
        input2sub = 'qcj'

    NQ = weights.shape[0]
    NC = measure.shape[0]
    
    if term is None:
        if coef is None:
            return np.einsum(f'q, c, {input1sub}, {input2sub} -> cij', weights, measure, input1, input2)
        
        if is_scalar(coef):
            return np.einsum(f'q, c, {input1sub}, {input2sub} -> cij', weights, measure, input1, input2) * coef
        elif is_tensor(coef):
            out_subs = 'cij'
            subs = get_coef_subscripts(coef.shape, NQ, NC)
            return np.einsum(f'q, c, {input1sub}, {input2sub}, {subs} -> {out_subs}', weights, measure, input1, input2, coef)
        else:
            raise TypeError(f"coef should be int, float or Tensor, but got {type(coef)}.")
    else:
        if coef is None:
            return np.einsum(f'q, c, {input1sub}, {input2sub}, ij -> cij', weights, measure, input1, input2, term)

        if is_scalar(coef):
            return np.einsum(f'q, c, {input1sub}, {input2sub}, ij -> cij', weights, measure, input1, input2, term) * coef
        elif is_tensor(coef):
            out_subs = 'ij'
            subs = get_coef_subscripts(coef.shape, NQ, NC)
            return np.einsum(f'q, c, {input1sub}, {input2sub}, ij, {subs} -> {out_subs}', weights, measure, input1, input2, term, coef)
        else:
            raise TypeError(f"coef should be int, float or Tensor, but got {type(coef)}.")
