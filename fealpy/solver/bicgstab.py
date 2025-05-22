from typing import Optional, Protocol

from ..backend import backend_manager as bm
from ..backend import TensorLike

from .. import logger


class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...


def bicgstab(
    A: SupportsMatmul,
    b: TensorLike,
    x0: Optional[TensorLike] = None,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    maxit: Optional[int] = None,
    M: Optional[SupportsMatmul] = None
) -> tuple[TensorLike, dict]:
    """
    Solve a linear system Ax = b using BiConjugate Gradient Stabilized (BiCGSTAB) method.

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
    
    if x0 is not None:
        kwags = bm.context(x0)
    else:
        kwags = bm.context(b)
    
    if x0 is None:
        x0 = bm.zeros_like(b,**kwags)
    elif x0.shape != b.shape:
        raise ValueError("x0 and b must have the same shape")
    
    m, _ = A.shape  # Matrix dimensions

    maxit = maxit or 5 * m  # Default maximum iterations
    if x0 is None:
        r = b.copy()
    else:
        r = b - A @ x0 

    info = {}
    b_norm = bm.linalg.norm(b)
    atol = max(float(atol), float(rtol) * float(b_norm))
    rhotol = bm.finfo(x0.dtype).eps**2

    x = x0
    r1 = r
    a = r @ r1
    p = r
    
    q = bm.empty_like(r,**kwags)
    Mp = bm.empty_like(r,**kwags)
    AMp = bm.empty_like(r,**kwags)
    Mq = bm.empty_like(r,**kwags)
    AMq = bm.empty_like(r,**kwags)
    tmp = bm.empty_like(r,**kwags)
    
    for niter in range(maxit):
        if abs(a) < rhotol:
            logger.info(f"break")
            break

        if niter > 0:
            a = r @ r1
            mu = (eta / w) * (a / a_pre)
            # p = mu * (p - w * AMp) + r
            tmp = p - w * AMp
            tmp = mu * tmp
            p = tmp + r
        
        if M is not None:
            Mp = M @ p
        else:
            Mp = p
        AMp = A @ Mp
        c = AMp @ r1
        # c = bm.einsum('i,i->', AMp, r1)
        if abs(c) == 0:
            logger.info(f"break")
            break
        eta = a / c

        q = r - eta * AMp
        qnorm = bm.linalg.norm(q)
        if qnorm <= max(atol,rtol * b_norm):
            x = x + eta * Mp
            res = qnorm
            logger.info(
                f"bicgstab converged in {niter} iterations"
            )
            break

        if M is not None:
            Mq = M @ q
        else:
            Mq = q
        AMq = A @ Mq
        w = (AMq @ Mq) / (AMq @ AMq)

        x += w * Mq
        x += eta * Mp

        r = q - w *AMq
        a_pre = a
        
        res = bm.linalg.norm(r)
        if res <= atol:
            logger.info(
                f"bicgstab converged in {niter} iterations (absolute tolerance: {atol:.1e})"
            )
            break

        if res <= rtol * b_norm:
            logger.info(
                f"bicgstab converged in {niter} iterations (relative tolerance: {rtol:.1e})"
            )
            break      

        if (maxit is not None) and (niter >= maxit):
            logger.info(f"bicgstab failed to converge within {maxit} iterations.")
            break

    info['residual'] = res
    info['niter'] = niter+1
    return x, info
