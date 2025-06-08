from typing import Optional, Protocol

from ..backend import backend_manager as bm
from ..backend import TensorLike

from .. import logger


class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...


def bicg(
    A: SupportsMatmul,
    b: TensorLike,
    x0: Optional[TensorLike] = None,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    maxit: Optional[int] = None,
    M: Optional[SupportsMatmul] = None
) -> tuple[TensorLike, dict]:
    """
    Solve a linear system Ax = b using BiConjugate Gradient (BiCG) method.

    Parameters:
        A (SupportsMatmul): The coefficient matrix of the linear system.
        b (TensorLike): The right-hand side vector of the linear system, can be a 1D or 2D tensor.
        x0 (TensorLike, optional): Initial guess for the solution, a 1D or 2D tensor.
            Must have the same shape as b when reshaped appropriately. Defaults to None.
        atol (float, optional): Absolute tolerance for convergence. Defaults to 1e-8.
        rtol (float, optional): Relative tolerance for convergence. Defaults to 1e-8.
        maxit (int, optional): Maximum number of iterations allowed. Defaults to 5*m.
            If not provided, the method will continue until convergence based on tolerances.
        M (TensorLike): Inverse of the preconditioner of `A`.  `M` should approximate the
            inverse of `A` and be easy to solve. Defaults to None.

    Returns:
        TensorLike: The approximate solution to the system Ax = b.
        dict: Information dictionary containing residual, iteration count, and relative tolerance.

    Raises:
        ValueError: If inputs do not meet specified conditions (e.g., dimensions mismatch).
    """
    assert isinstance(b, TensorLike), "b must be a Tensor"
    if x0 is not None:
        assert isinstance(x0, TensorLike), "x0 must be a Tensor if not None"
    if b.ndim not in {1, 2}:
        raise ValueError("b must be a 1D or 2D dense tensor")
    
    if x0 is None:
        x0 = bm.zeros_like(b)
    elif x0.shape != b.shape:
        raise ValueError("x0 and b must have the same shape")
    
    m, n = A.shape  # Matrix dimensions

    maxit = maxit or 5 * m  # Default maximum iterations
    if x0 is None:
        r = b.copy()
    else:
        r = b - A @ x0 

    info = {}
    b_norm = bm.linalg.norm(b)
    atol = max(float(atol), float(rtol) * float(b_norm))
    
    # Transpose and preconditioner setup
    AT = A.T
    if M is not None:
        MT = M.T 

    r1 = r 
    x = x0
    p1 = p = r

    Ap = bm.empty_like(p)
    Ap1 = bm.empty_like(p1)
    z  = (M @ r)  if M is not None else r
    z1 = (MT @ r1) if M is not None else r1
    a = z @ r1
    
    for niter in range(maxit):
        Ap = A @ p
        Ap1 = AT @ p1

        eta = a / (p1.T @ Ap)
        x = x + eta * p
        r = r - eta * Ap
        r1 = r1 - eta.conj() * Ap1

        z  = (M @ r)  if M is not None else r
        z1 = (MT @ r1) if M is not None else r1
    
        a_pre = a
        a = z @ r1
        mu = a / a_pre
        p = z + mu * p
        p1 = z1 + mu.conj() * p1

        res = bm.linalg.norm(r)
        
        if res <= atol:
            logger.info(
                f"bicg converged in {niter} iterations (absolute tolerance: {atol:.1e})"
            )
            break

        if res <= rtol * b_norm:
            logger.info(
                f"bicg converged in {niter} iterations (relative tolerance: {rtol:.1e})"
            )
            break

        if (maxit is not None) and (niter >= maxit):
            logger.info(f"bicg failed to converge within {maxit} iterations.")
            break

    info['residual'] = res
    info['niter'] = niter+1
    return x, info
