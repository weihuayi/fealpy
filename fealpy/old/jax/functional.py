
import builtins
from typing import Union, Callable

import jax.numpy as jnp

from .utils import is_scalar, is_tensor, get_coef_subscripts, Array


Number = Union[builtins.int, builtins.float]
CoefLike = Union[Number, Array, Callable[..., Array]]


def integral(value: Array, weights: Array, measure: Array, *,
             entity_type=False) -> Array:
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
    return jnp.einsum(f'q, c, ...qc -> {subs}', weights, measure, value)


def linear_integral(input: Array, weights: Array, measure: Array,
                    coef: Union[Number, Array, None]=None) -> Array:
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
        return jnp.einsum('q, c, qci -> ci', weights, measure, input)

    NQ = weights.shape[0]
    NC = measure.shape[0]
    if is_scalar(coef):
        return jnp.einsum('q, c, qci -> ci', weights, measure, input) * coef
    elif is_tensor(coef):
        out_subs = 'ci'
        subs = get_coef_subscripts(coef.shape, NQ, NC)
        return jnp.einsum(f'q, c, cqi, {subs} -> {out_subs}', weights, measure, input, coef)
    else:
        raise TypeError(f"coef should be int, float or Tensor, but got {type(coef)}.")


def bilinear_integral(input1: Array, input2: Array, weights: Array, measure: Array,
                      coef: Union[Number, Array, None]=None) -> Array:
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
    if len(input1.shape)==4:
        input1sub = 'cqid'
        input2sub = 'cqjd'
    else:
        input1sub = 'cqi'
        input2sub = 'cqj'

    if coef is None:
        return jnp.einsum(f'q, c, {input1sub}, {input2sub} -> cij', weights, measure, input1, input2)

    NQ = weights.shape[0]
    NC = measure.shape[0]

    if is_scalar(coef):
        return jnp.einsum(f'q, c, {input1sub}, {input2sub} -> cij', weights, measure, input1, input2) * coef
    elif is_tensor(coef):
        out_subs = 'cij'
        subs = get_coef_subscripts(coef.shape, NQ, NC)
        return jnp.einsum(f'q, c, {input1sub}, {input2sub}, {subs} -> {out_subs}', weights, measure, input1, input2, coef)
    else:
        raise TypeError(f"coef should be int, float or Tensor, but got {type(coef)}.")
