
__all__ = [
    "integral_transform",
    "piola_transform_covariant",
    "piola_transform_contravariant"
]

from ..backend import bm, Tensor


def check_jacobi_matrix(J: Tensor) -> None:
    """Check the validity of the Jacobian matrix."""
    if J.ndim < 2:
        raise ValueError("The Jacobian matrix must have at least 2 dimensions.")


def is_square_jacobi_matrix(J: Tensor) -> bool:
    """Check if the Jacobian matrix is square."""
    return J.shape[-2] == J.shape[-1]


def integral_transform(
    value: Tensor,
    J: Tensor,
) -> Tensor:
    """Integral transformation.

    Parameters:
        value (Tensor): Value of tensor-valued function to be transformed,
            with shape (...[, any_dim]).
        J (Tensor): Jacobian matrix, with shape (..., phy_dim, ref_dim).

    Returns:
        Tensor: Transformed value, with shape (...[, any_dim]).
    """
    check_jacobi_matrix(J)

    if is_square_jacobi_matrix(J):
        W = bm.linalg.det(J)
    else:
        W = bm.sqrt(bm.linalg.det(bm.einsum("...xi, ...xj -> ...ij", J, J)))

    return value / W


def piola_transform_covariant(
    value: Tensor,
    J: Tensor,
) -> Tensor:
    """Piola transformation for covariant vectors.

    Parameters:
        value (Tensor): Value of tensor-valued function to be transformed,
            with shape (..., ref_dim).
        J (Tensor): Jacobian matrix, with shape (..., phy_dim, ref_dim).

    Returns:
        Tensor: Transformed value, with shape (..., phy_dim).
    """
    check_jacobi_matrix(J)

    if is_square_jacobi_matrix(J):
        J_inv_T = bm.linalg.inv(J).mT
        return bm.einsum("...xj, ...j -> ...x", J_inv_T, value)
    else:
        G = bm.einsum("...xi, ...xj -> ...ij", J, J)
        G_inv = bm.linalg.inv(G)
        return bm.einsum("...xi, ...ij, ...j -> ...x", J, G_inv, value)


def piola_transform_contravariant(
    value: Tensor,
    J: Tensor,
) -> Tensor:
    """Piola transformation for contravariant vectors.

    Parameters:
        value (Tensor): Value of tensor-valued function to be transformed,
            with shape (..., ref_dim).
        J (Tensor): Jacobian matrix, with shape (..., phy_dim, ref_dim).

    Returns:
        Tensor: Transformed value, with shape (..., phy_dim).
    """
    check_jacobi_matrix(J)

    if is_square_jacobi_matrix(J):
        W = bm.linalg.det(J)
    else:
        W = bm.sqrt(bm.linalg.det(bm.einsum("...xi, ...xj -> ...ij", J, J)))

    return bm.einsum("...xi, ...i -> ...x", J, value) / W
