from typing import Tuple

import numpy as np

from ..typing import Array
from .utils import tensor_basis


def generate_tensor_basis(basis: Array, shape: Tuple[int, ...], dof_priority=True) -> Array:
    """Generate tensor basis from scalar basis.

    Parameters:
        basis (Array): Basis of a scalar space, shaped (..., ldof).\n
        shape (Tuple[int, ...]): Shape of the dof.\n
        dof_priority (bool, optional): If True, the degrees of freedom are arranged\
        prior to their components. Defaults to True.

    Returns:
        Array: Basis of the tensor space, shaped (..., ldof*numel, *shape),\
        where numel is the number of elements in the shape.
    """
    factor = tensor_basis(shape, dtype=basis.dtype)
    tb = np.tensordot(basis, factor, axes=0)
    ldof = basis.shape[-1]
    numel = factor.shape[0]

    if dof_priority:
        ndim = len(shape)
        tb = np.moveaxis(tb, -ndim-1, -ndim-2)

    return tb.reshape(basis.shape[:-1] + (numel*ldof,) + shape)


def generate_tensor_grad_basis(grad_basis: Array, shape: Tuple[int, ...], dof_priority=True) -> Array:
    """Generate tensor grad basis from grad basis in scalar space.

    Parameters:
        grad_basis (Array): Gradient of basis of a scalar space, shaped (..., ldof, GD).\n
        shape (Tuple[int, ...]): Shape of the dof.\n
        dof_priority (bool, optional): If True, the degrees of freedom are arranged\
        prior to their components. Defaults to True.

    Returns:
        Array: Basis of the tensor space, shaped (..., ldof*numel, *shape, GD),\
        where numel is the number of elements in the shape.
    """
    factor = tensor_basis(shape, dtype=grad_basis.dtype)
    s0 = "abcde"[:len(shape)]
    tb = np.einsum(f'...jz, n{s0} -> ...jn{s0}z', grad_basis, factor)
    ldof, GD = grad_basis.shape[-2:]
    numel = factor.shape[0]

    if dof_priority:
        ndim = len(shape)
        tb = np.moveaxis(tb, -ndim-2, -ndim-3)

    return tb.reshape(grad_basis.shape[:-2] + (numel*ldof,) + shape + (GD,))
