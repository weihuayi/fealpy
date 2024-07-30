
import builtins
from typing import Union, Callable

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, Number, CoefLike, _S

from .utils import is_scalar, is_tensor, get_coef_subscripts


def integral(value: TensorLike, weights: TensorLike, measure: TensorLike, *,
             entity_type=False) -> TensorLike:
    """Numerical integration.

    Args:
        value (TensorLike[..., C, Q]): The values on the quadrature points to be integrated.
        weights (TensorLike[Q,]): The weights of the quadrature points.
        measure (TensorLike[C,]): The measure of the quadrature points.
        entity_type (bool): Whether to return integration in each entity. Defaults to False.

    Returns:
        TensorLike[...]: The result of the integration. The shape will be [..., C]\
        if entity_type is True, otherwise [...].
    """
    subs = '...c' if entity_type else '...'
    return bm.einsum(f'c, q, ...cq -> {subs}', measure, weights, value)


def linear_integral(input: TensorLike, weights: TensorLike, measure: TensorLike,
                    coef: Union[Number, TensorLike, None]=None,
                    batched: bool=False) -> TensorLike:
    """Numerical integration.

    Args:
        input (TensorLike[C, Q, I]): The values on the quadrature points to be integrated.\n
        weights (TensorLike[Q,]): The weights of the quadrature points.\n
        measure (TensorLike[C,]): The measure of the quadrature points.\n
        coef (Number, TensorLike, optional): The coefficient of the integration. Defaults to None.
            Must be int, float, TensorLike, or callable returning TensorLike with shape (Q,), (C,) or (C, Q).
            If `batched == True`, the shape of the coef should be (B, Q, C) or (B, C).
        batched (bool, optional): Whether the coef are batched. Defaults to False.

    Returns:
        TensorLike[C, I]: The result of the integration.
            If `batched == True`, the shape of the result is (B, C, I).
    """
    if coef is None:
        return bm.einsum('c, q, cqi -> ci', measure, weights, input)

    NQ = weights.shape[0]
    NC = measure.shape[0]

    if is_scalar(coef):
        return bm.einsum('c, q, cqi -> ci', measure, weights, input) * coef
    elif is_tensor(coef):
        out_subs = 'bci' if batched else 'ci'
        subs = get_coef_subscripts(coef.shape, NQ, NC, batched)
        return bm.einsum(f'c, q, cqi, {subs} -> {out_subs}', measure, weights, input, coef)
    else:
        raise TypeError(f"coef should be int, float or TensorLike, but got {type(coef)}.")


def bilinear_integral(input1: TensorLike, input2: TensorLike, weights:
                      TensorLike, measure: TensorLike,
                      coef: Union[Number, TensorLike, None]=None,
                      batched: bool=False) -> TensorLike:
    """Numerical integration.

    Args:
        input1 (TensorLike[C, Q, I, ...]): The values on the quadrature points to be integrated.\n
        input2 (TensorLike[C, Q, J, ...]): The values on the quadrature points to be integrated.\n
        weights (TensorLike[Q,]): The weights of the quadrature points.\n
        measure (TensorLike[C,]): The measure of the quadrature points.\n
        coef (Number, TensorLike, optional): The coefficient of the integration. Defaults to None.
            Must be int, float, TensorLike, or callable returning TensorLike with shape (Q,), (C,) or (C, Q).
            If `batched == True`, the shape of the coef should be (B, C, Q) or (B, C).
        batched (bool, optional): Whether the coef are batched. Defaults to False.

    Returns:
        TensorLike[C, I, J]: The result of the integration.
            If `batched == True`, the shape of the output is (B, C, I, J).
    """
    if coef is None:
        return bm.einsum('c, q, cqi..., cqj... -> cij', measure, weights, input1, input2)

    NQ = weights.shape[0]
    NC = measure.shape[0]

    if is_scalar(coef):
        return bm.einsum('c, q, cqi..., cqj... -> cij', measure, weights, input1, input2) * coef
    elif is_tensor(coef):
        out_subs = 'bcij' if batched else 'cij'
        subs = get_coef_subscripts(coef.shape, NQ, NC, batched)
        return bm.einsum(f'c, q, cqi..., cqj..., {subs} -> {out_subs}', measure, weights, input1, input2, coef)
    else:
        raise TypeError(f"coef should be int, float or TensorLike, but got {type(coef)}.")
