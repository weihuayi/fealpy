
from typing import Tuple

from torch import tensordot

from ..typing import Tensor
from .utils import tensor_basis


def generate_tensor_basis(basis: Tensor, shape: Tuple[int, ...], dof_priority=True) -> Tensor:
    """Generate tensor basis from scalar basis.

    Parameters:
        basis (Tensor): Basis of a scalar space, shaped (..., ldof).\n
        shape (Tuple[int, ...]): Shape of the dof.\n
        dof_priority (bool, optional): If True, the degrees of freedom are ranked\
        prior to their components. Defaults to True.

    Returns:
        Tensor: Basis of the tensor space, shaped (..., ldof*numel, *shape),\
        where numel is the number of elements in the shape.
    """
    factor = tensor_basis(shape, dtype=basis.dtype, device=basis.device)
    tb = tensordot(basis, factor, dims=0)
    ldof = basis.shape[-1]
    numel = factor.shape[0]

    if dof_priority:
        ndim = len(shape)
        tb = tb.transpose(-ndim-1, -ndim-2)

    return tb.reshape(basis.shape[:-1] + (numel*ldof,) + shape)
