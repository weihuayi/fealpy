
import builtins
from typing import Union, Callable

from mindspore import Tensor
import mindspore.ops as ops

from .utils import is_scalar, is_tensor, get_coef_subscripts


Number = Union[builtins.int, builtins.float]
CoefLike = Union[Number, Tensor, Callable[..., Tensor]]


def integral(value: Tensor, weights: Tensor, measure: Tensor, *,
             entity_type=False) -> Tensor:
    """Numerical integration.

    Args:
        value (Tensor[..., C, Q]): The values on the quadrature points to be integrated.
        weights (Tensor[Q,]): The weights of the quadrature points.
        measure (Tensor[C,]): The measure of the quadrature points.
        entity_type (bool): Whether to return integration in each entity. Defaults to False.

    Returns:
        Tensor[...]: The result of the integration. The shape will be [..., C]\
        if entity_type is True, otherwise [...].
    """
    subs = '...c' if entity_type else '...'
    return ops.einsum(f'c, q, ...cq -> {subs}', measure, weights, value)


def linear_integral(input: Tensor, weights: Tensor, measure: Tensor,
                    coef: Union[Number, Tensor, None]=None,
                    batched: bool=False) -> Tensor:
    """Numerical integration.

    Args:
        input (Tensor[C, Q, I]): The values on the quadrature points to be integrated.\n
        weights (Tensor[Q,]): The weights of the quadrature points.\n
        measure (Tensor[C,]): The measure of the quadrature points.\n
        coef (Number, Tensor, optional): The coefficient of the integration. Defaults to None.
            Must be int, float, Tensor, or callable returning Tensor with shape (Q,), (C,) or (C, Q).
            If `batched == True`, the shape of the coef should be (B, Q, C) or (B, C).
        batched (bool, optional): Whether the coef are batched. Defaults to False.

    Returns:
        Tensor[C, I]: The result of the integration.
            If `batched == True`, the shape of the result is (B, C, I).
    """
    if coef is None:
        return ops.einsum('c, q, cqi -> ci', measure, weights, input)

    NQ = weights.shape[0]
    NC = measure.shape[0]

    if is_scalar(coef):
        return ops.einsum('c, q, cqi -> ci', measure, weights, input) * coef
    elif is_tensor(coef):
        out_subs = 'bci' if batched else 'ci'
        subs = get_coef_subscripts(coef.shape, NQ, NC, batched)
        return ops.einsum(f'c, q, cqi, {subs} -> {out_subs}', measure, weights, input, coef)
    else:
        raise TypeError(f"coef should be int, float or Tensor, but got {type(coef)}.")


def bilinear_integral(input1: Tensor, input2: Tensor, weights: Tensor, measure: Tensor,
                      coef: Union[Number, Tensor, None]=None,
                      batched: bool=False) -> Tensor:
    """Numerical integration.

    Args:
        input1 (Tensor[C, Q, I, ...]): The values on the quadrature points to be integrated.\n
        input2 (Tensor[C, Q, J, ...]): The values on the quadrature points to be integrated.\n
        weights (Tensor[Q,]): The weights of the quadrature points.\n
        measure (Tensor[C,]): The measure of the quadrature points.\n
        coef (Number, Tensor, optional): The coefficient of the integration. Defaults to None.
            Must be int, float, Tensor, or callable returning Tensor with shape (Q,), (C,) or (C, Q).
            If `batched == True`, the shape of the coef should be (B, C, Q) or (B, C).
        batched (bool, optional): Whether the coef are batched. Defaults to False.

    Returns:
        Tensor[C, I, J]: The result of the integration.
            If `batched == True`, the shape of the output is (B, C, I, J).
    """
    if coef is None:
        return ops.einsum('c, q, cqi..., cqj... -> cij', measure, weights, input1, input2)

    NQ = weights.shape[0]
    NC = measure.shape[0]

    if is_scalar(coef):
        return ops.einsum('c, q, cqi..., cqj... -> cij', measure, weights, input1, input2) * coef
    elif is_tensor(coef):
        out_subs = 'bcij' if batched else 'cij'
        subs = get_coef_subscripts(coef.shape, NQ, NC, batched)
        return ops.einsum(f'c, q, cqi..., cqj..., {subs} -> {out_subs}', measure, weights, input1, input2, coef)
    else:
        raise TypeError(f"coef should be int, float or Tensor, but got {type(coef)}.")
