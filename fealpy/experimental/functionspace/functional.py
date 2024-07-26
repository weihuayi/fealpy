
from typing import Tuple

from ..typing import TensorLike
from ..backend import backend_manager as bm
from .utils import tensor_basis


def generate_tensor_basis(basis: TensorLike, shape: Tuple[int, ...], dof_priority=True) -> TensorLike:
    """Generate tensor basis from scalar basis.

    Parameters:
        basis (Tensor): Basis of a scalar space, shaped (..., ldof).\n
        shape (Tuple[int, ...]): Shape of the dof.\n
        dof_priority (bool, optional): If True, the degrees of freedom are arranged\
        prior to their components. Defaults to True.

    Returns:
        Tensor: Basis of the tensor space, shaped (..., ldof*numel, *shape),\
        where numel is the number of elements in the shape.
    """
    factor = tensor_basis(shape, dtype=basis.dtype)
    tb = bm.tensordot(basis, factor, axes=0)
    ldof = basis.shape[-1]
    numel = factor.shape[0]

    if dof_priority:
        ndim = len(shape)
        tb = bm.swapaxes(tb, -ndim-1, -ndim-2)

    return tb.reshape(basis.shape[:-1] + (numel*ldof,) + shape)


def generate_tensor_grad_basis(grad_basis: TensorLike, shape: Tuple[int, ...], dof_priority=True) -> TensorLike:
    """Generate tensor grad basis from grad basis in scalar space.

    Parameters:
        grad_basis (Tensor): Gradient of basis of a scalar space, shaped (..., ldof, GD).\n
        shape (Tuple[int, ...]): Shape of the dof.\n
        dof_priority (bool, optional): If True, the degrees of freedom are arranged\
        prior to their components. Defaults to True.

    Returns:
        Tensor: Basis of the tensor space, shaped (..., ldof*numel, *shape, GD),\
        where numel is the number of elements in the shape.
    """
    factor = tensor_basis(shape, dtype=grad_basis.dtype)
    s0 = "abcde"[:len(shape)]
    tb = bm.einsum(f'...jz, n{s0} -> ...jn{s0}z', grad_basis, factor)
    ldof, GD = grad_basis.shape[-2:]
    numel = factor.shape[0]

    if dof_priority:
        ndim = len(shape)
        tb = bm.swapaxes(tb, -ndim-2, -ndim-3)

    return tb.reshape(grad_basis.shape[:-2] + (numel*ldof,) + shape + (GD,))
