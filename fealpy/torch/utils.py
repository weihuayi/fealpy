
import builtins
from typing import Optional, Union, Callable, TypeGuard

from torch import Tensor, einsum

from .mesh import HomoMesh

Number = Union[builtins.int, builtins.float]
CoefLike = Union[Number, Tensor, Callable[..., Tensor]]
Index = Union[slice, Tensor, int]


def process_coef_func(coef: CoefLike, bcs: Optional[Tensor]=None, mesh: Optional[HomoMesh]=None,
                      index: Optional[Tensor]=None):
    if callable(coef):
        if index is None:
            raise RuntimeError('The index should be provided for coef functions.')
        if getattr(coef, 'coordtype', 'cartesian') == 'cartesian':
            if (mesh is None) or (not isinstance(mesh, HomoMesh)):
                raise RuntimeError('The mesh should be provided for cartesian coef functions.'
                                   'Note that only homogeneous meshes are supported here.')

            ps = mesh.bc_to_point(bcs, index=index)
            coef_val = coef(ps)
        else:
            coef_val = coef(bcs, index=index)
    else:
        coef_val = coef
    return coef_val


def is_scalar(input: Union[int, float, Tensor]):
    if isinstance(input, Tensor):
        return input.numel() == 1
    else:
        return isinstance(input, (int, float))


def is_tensor(input: Union[int, float, Tensor]) -> TypeGuard[Tensor]:
    if isinstance(input, Tensor):
        return input.numel() >= 2
    return False


def get_coef_subscripts(shape: Tensor, nq: int, nc: int, batched: bool):
    if batched:
        coef_shape = shape[1:]
        if coef_shape == (nq, nc):
            subs = "bqc"
        elif coef_shape == (nq, ):
            subs = "bq"
        elif coef_shape == (nc, ):
            subs = "bc"
        else:
            raise RuntimeError(f"The shape of the coef should be (Batch, {nq}, {nc}), "
                               f"(Batch, {nq}) or (Batch, {nc}), but got {tuple(shape)}.")

    else:
        coef_shape = shape
        if coef_shape == (nq, nc):
            subs = "qc"
        elif coef_shape == (nq, ):
            subs = "q"
        elif coef_shape == (nc, ):
            subs = "c"
        else:
            raise RuntimeError(f"The shape of the coef should be ({nq}, {nc}), "
                               f"({nq}) or ({nc}), but got {tuple(shape)}.")

    return subs


def linear_integral(input: Tensor, weights: Tensor, measure: Tensor,
                    coef: Union[Number, Tensor, None]=None,
                    batched: bool=False) -> Tensor:
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
    """
    if coef is None:
        return einsum('q, c, qci -> ci', weights, measure, input)

    NQ, NC = input.shape[:2]

    if is_scalar(coef):
        return einsum('q, c, qci -> ci', weights, measure, input) * coef
    elif is_tensor(coef):
        out_subs = 'bci' if batched else 'ci'
        subs = get_coef_subscripts(coef.shape, NQ, NC, batched)
        return einsum(f'q, c, qci, {subs} -> {out_subs}', weights, measure, input, coef)
    else:
        raise TypeError(f"coef should be int, float, Tensor or callable, but got {type(coef)}.")


def bilinear_integral(input1: Tensor, input2: Tensor, weights: Tensor, measure: Tensor,
                      coef: Union[Number, Tensor, None]=None,
                      batched: bool=False) -> Tensor:
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
    """
    if coef is None:
        return einsum('q, c, qci..., qcj... -> cij', weights, measure, input1, input2)

    NQ, NC = input1.shape[:2]

    if is_scalar(coef):
        return einsum('q, c, qci..., qcj... -> cij', weights, measure, input1, input2) * coef
    elif is_tensor(coef):
        out_subs = 'bcij' if batched else 'cij'
        subs = get_coef_subscripts(coef.shape, NQ, NC, batched)
        return einsum(f'q, c, qci..., qcj..., {subs} -> {out_subs}', weights, measure, input1, input2, coef)
    else:
        raise TypeError(f"coef should be int, float, Tensor or callable, but got {type(coef)}.")
