from typing import Optional

import numpy as np

from numpy.typing import NDArray
Array = np.ndarray

def normal_strain(gphi: NDArray, indices: NDArray, *, out: Optional[NDArray] = None) -> NDArray:
    """Assembly normal strain tensor.

    Parameters:
        gphi (Array): Gradient of the scalar basis functions shaped (..., ldof, GD).
        indices (Array): Indices of DoF components in the flattened DoF, shaped (ldof, GD).
        out (Array | None, optional): Output tensor. Defaults to None.

    Returns:
        Array: Normal strain shaped (..., GD, GD*ldof).
    """
    ldof, GD = gphi.shape[-2:]
    new_shape = gphi.shape[:-2] + (GD, GD * ldof)

    if out is None:
        out = np.zeros(new_shape, dtype=gphi.dtype)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    for i in range(GD):
        out[..., i, indices[:, i]] = gphi[..., :, i]

    return out


def shear_strain(gphi: NDArray, indices: NDArray, *, out: Optional[NDArray] = None) -> NDArray:
    """Assembly shear strain tensor.

    Parameters:
        gphi (Array): Gradient of the scalar basis functions shaped (..., ldof, GD).
        indices (Array): Indices of DoF components in the flattened DoF, shaped (ldof, GD).
        out (Array | None, optional): Output tensor. Defaults to None.

    Returns:
        Array: Sheared strain shaped (..., NNZ, GD*ldof) where NNZ = (GD * (GD-1)) // 2.
    """
    ldof, GD = gphi.shape[-2:]
    if GD < 2:
        raise ValueError(f"The shear strain requires GD >= 2, but GD = {GD}")
    NNZ = (GD * (GD - 1)) // 2
    new_shape = gphi.shape[:-2] + (NNZ, GD * ldof)

    if out is None:
        out = np.zeros(new_shape, dtype=gphi.dtype)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    cursor = 0
    for i in range(0, GD - 1):
        for j in range(i + 1, GD):
            out[..., cursor, indices[:, i]] = gphi[..., :, j]
            out[..., cursor, indices[:, j]] = gphi[..., :, i]
            cursor += 1

    return out
