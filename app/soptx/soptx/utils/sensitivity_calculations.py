from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

def manual_objective_sensitivity(ce: TensorLike, dE: TensorLike) -> TensorLike:
    """
    Compute the sensitivities of the objective function manually.

    Args:
        ce (TensorLike): The computed value of the objective function for each element.
        dE (TensorLike): The derivative of the Young's modulus with respect to density for each element.

    Returns:
        TensorLike: Objective sensitivity (dce).
    """
    NC = ce.shape[0]  # Number of elements
    dce = bm.zeros(NC, dtype=bm.float64)

    dce[:] = -bm.einsum('c, c -> c', dE, ce)  # Element-wise multiplication of dE and ce

    return dce

def manual_volume_sensitivity(ce: TensorLike, dE: TensorLike) -> TensorLike:
    """
    Compute the sensitivities of the volume constraints manually.

    Args:
        ce (TensorLike): The computed value of the objective function for each element.
        dE (TensorLike): The derivative of the Young's modulus with respect to density for each element.

    Returns:
        TensorLike: Volume sensitivity (dve).
    """
    NC = ce.shape[0]  # Number of elements

    dve = bm.ones(NC, dtype=bm.float64)  # Volume sensitivity is usually a vector of ones

    return dve
