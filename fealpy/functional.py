
from typing import Optional

from .backend import backend_manager as bm
from .typing import TensorLike, CoefLike
from .utils import is_scalar, is_tensor, fill_axis


def integral(value: TensorLike, weights: TensorLike, measure: TensorLike, *,
             entity_type=False) -> TensorLike:
    """Numerical integration.

    Parameters:
        value (TensorLike[..., C, Q]): The values on the quadrature points to be integrated.
        weights (TensorLike[Q,]): The weights of the quadrature points.
        measure (TensorLike[C,]): The measure of the quadrature points.
        entity_type (bool): Whether to return integration in each entity. Defaults to False.

    Returns:
        TensorLike: The result of the integration. The shape will be [..., C]\
        if `entity_type` is True, otherwise [...].
    """
    subs = '...c' if entity_type else '...'
    return bm.einsum(f'c, q, ...cq -> {subs}', measure, weights, value)


def linear_integral(basis: TensorLike, weights: TensorLike, measure: TensorLike,
                    source: Optional[CoefLike]=None,
                    batched: bool=False) -> TensorLike:
    """Numerical integration.

    Parameters:
        basis (TensorLike[C, Q, I, ...]): The values on the quadrature points to be integrated.
        weights (TensorLike[Q,]): The weights of the quadrature points.
        measure (TensorLike[C,]): The measure of the mesh entity.
        source (Number, TensorLike, optional): The source of the integration. Defaults to None.
            Must be int, float, TensorLike, or callable returning TensorLike with shape (C,), (C, Q) or (C, Q, ...).
            If `batched == True`, there should be a batch dimension as the first axis.
        batched (bool, optional): Whether the source are batched. Defaults to False.

    Returns:
        TensorLike: The result of the integration shaped (C, I) for source (C, Q, ...).
            And (C, I, ...) for source (C, ) and (C, Q).
            If `batched` is True, there will be a batch dimension as the first axis.
    """
    if source is None:
        return bm.einsum('c, q, cq... -> c...', measure, weights, basis)

    if is_scalar(source):
        return bm.einsum('c, q, cq... -> c...', measure, weights, basis) * source

    elif is_tensor(source):
        dof_shape = basis.shape[3:]
        basis = basis.reshape(*basis.shape[:3], -1) # (C, Q, I, dof_numel)

        if source.ndim <= 2 + int(batched):
            source = fill_axis(source, 3 if batched else 2)
            r = bm.einsum(f'c, q, cqid, ...cq -> ...cid', measure, weights, basis, source)
            return bm.reshape(r, r.shape[:-1] + dof_shape)
        else:
            source = fill_axis(source, 4 if batched else 3)
            return bm.einsum(f'c, q, cqid, ...cqd -> ...ci', measure, weights, basis, source)

    else:
        raise TypeError(f"source should be int, float or TensorLike, but got {type(source)}.")


def bilinear_integral(basis1: TensorLike, basis2: TensorLike, weights: TensorLike,
                      measure: TensorLike,
                      coef: Optional[CoefLike]=None,
                      batched: bool=False) -> TensorLike:
    """Numerical integration.

    Parameters:
        basis1 (TensorLike[C, Q, I, ...]): The values on the quadrature points to be integrated.
        basis2 (TensorLike[C, Q, J, ...]): The values on the quadrature points to be integrated.
        weights (TensorLike[Q,]): The weights of the quadrature points.
        measure (TensorLike[C,]): The measure of the mesh entity.
        coef (Number, TensorLike, optional): The coefficient of the integration. Defaults to None.
            Must be int, float, TensorLike, or callable returning TensorLike with shape (C,), (C, Q) or (C, Q, ...).
            If `batched == True`, there should be a batch dimension as the first axis.
        batched (bool, optional): Whether the coef are batched. Defaults to False.

    Returns:
        TensorLike: The result of the integration shaped (C, I, J).
            If `batched` is True, the shape of the output is (B, C, I, J).
    """
    basis1 = basis1.reshape(*basis1.shape[:3], -1) # (C, Q, I, dof_numel)
    basis2 = basis2.reshape(*basis2.shape[:3], -1) # (C, Q, J, dof_numel)

    if coef is None:
        return bm.einsum(f'q, c, cqid, cqjd -> cij', weights, measure, basis1, basis2)

    if is_scalar(coef):
        return bm.einsum(f'q, c, cqid, cqjd -> cij', weights, measure, basis1, basis2) * coef

    elif is_tensor(coef):
        ndim = coef.ndim - int(batched)
        if ndim == 4:
            return  bm.einsum(f'q, c, cqid, cqjn, ...cqdn -> ...cij', weights, measure, basis1, basis2, coef)
        else:
            coef = fill_axis(coef, 4 if batched else 3)
            return bm.einsum(f'q, c, cqid, cqjd, ...cqd -> ...cij', weights, measure, basis1, basis2, coef)
        
    else:
        raise TypeError(f"coef should be int, float or TensorLike, but got {type(coef)}.")

def get_semilinear_coef(value:TensorLike, coef: Optional[CoefLike]=None, batched: bool=False):

    if coef is None:
        return coef * value

    if is_scalar(coef):
        return coef * value

    if is_tensor(coef):
        coef = fill_axis(coef, value.ndim + 1 if batched else value.ndim)
        return coef * value
    else:
        raise TypeError(f"coef should be int, float or TensorLike, but got {type(coef)}.")
