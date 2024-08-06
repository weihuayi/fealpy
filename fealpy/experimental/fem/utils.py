
from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike


def normal_strain(gphi: TensorLike, indices: TensorLike, *, out:
                  Optional[TensorLike]=None) -> TensorLike:
    """Assembly normal strain tensor.

    Parameters:
        gphi (TensorLike): Gradient of the scalar basis functions shaped (..., ldof, GD).\n
        indices (bool, optional): Indices of DoF components in the flattened DoF, shaped (ldof, GD).\n
        out (TensorLike | None, optional): Output tensor. Defaults to None.

    Returns:
        TensorLike: Normal strain shaped (..., GD, GD*ldof).
    """
    kwargs = bm.context(gphi)
    ldof, GD = gphi.shape[-2:]
    new_shape = gphi.shape[:-2] + (GD, GD*ldof)

    if out is None:
        out = bm.zeros(new_shape, **kwargs)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    for i in range(GD):
        out[..., i, indices[:, i]] = gphi[..., :, i]

    return out


def shear_strain(gphi: TensorLike, indices: TensorLike, *, out:
                 Optional[TensorLike]=None) -> TensorLike:
    """Assembly shear strain tensor.

    Parameters:
        gphi (TensorLike): Gradient of the scalar basis functions shaped (..., ldof, GD).\n
        indices (bool, optional): Indices of DoF components in the flattened DoF, shaped (ldof, GD).\n
        out (TensorLike | None, optional): Output tensor. Defaults to None.

    Returns:
        TensorLike: Sheared strain shaped (..., NNZ, GD*ldof) where NNZ = (GD + (GD+1))//2.
    """
    kwargs = bm.context(gphi)
    ldof, GD = gphi.shape[-2:]
    if GD < 2:
        raise ValueError(f"The shear strain requires GD >= 2, but GD = {GD}")
    NNZ = (GD * (GD-1))//2
    new_shape = gphi.shape[:-2] + (NNZ, GD*ldof)

    if out is None:
        out = bm.zeros(new_shape, **kwargs)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    cursor = 0
    for i in range(0, GD-1):
        for j in range(i+1, GD):
            out[..., cursor, indices[:, i]] = gphi[..., :, j]
            out[..., cursor, indices[:, j]] = gphi[..., :, i]
            cursor += 1

    return out
