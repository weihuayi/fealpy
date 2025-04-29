from typing import Optional, Protocol

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse.coo_tensor import COOTensor
from ..sparse.csr_tensor import CSRTensor

from .. import logger


class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...


def minres(A: SupportsMatmul, b: TensorLike, x0: Optional[TensorLike]=None,
       atol: float=1e-12, rtol: float=1e-8,
       maxit: Optional[int]= None) -> TensorLike:
    """
    Solve a linear system Ax = b using MINimum RESidual iteration (minres) method.
    Parameters:
        A (SupportsMatmul): The coefficient matrix of the linear system.
        b (TensorLike): The right-hand side vector of the linear system, can be a 1D or 2D tensor.
        x0 (TensorLike): Initial guess for the solution, a 1D or 2D tensor.
            Must have the same shape as b when reshaped appropriately.
        atol (float, optional): Absolute tolerance for convergence. Default is 1e-12.
        rtol (float, optional): Relative tolerance for convergence. Default is 1e-8.
        maxiter (int, optional): Maximum number of iterations allowed. Default is 5 * m.
            If not provided, the method will continue until convergence based on the given tolerances.

    Returns:
        Tensor: The approximate solution to the system Ax = b.
        info: Including residual, iteration count, relative tolerance.

    Raises:
        ValueError: If inputs do not meet the specified conditions (e.g., A is not sparse, dimensions mismatch).

    Note:
        This implementation assumes that A is a symmetric matrix,
        which is a common requirement for the minres method to work correctly.
    """
    assert isinstance(b, TensorLike), "b must be a Tensor"
    if x0 is not None:
        assert isinstance(x0, TensorLike), "x0 must be a Tensor if not None"
    if b.ndim not in {1, 2}:
        raise ValueError("b must be a 1D or 2D dense tensor")
    
    if x0 is None:
        x0 = bm.zeros_like(b)
    else:
        if x0.shape != b.shape:
            raise ValueError("x0 and b must have the same shape")
    
    m, n = A.shape

    if maxit is None:
        maxit = 5 * m

    if x0 is None:
        r = b.copy()
    else:
        r = b - A @ x0

    beta = bm.linalg.norm(r, ord=2)

    # Ensure A is symmetric
    w = A @ r
    w_1 = A @ w
    s = w @ w
    a = r @ w_1
    eps = bm.finfo(x0.dtype).eps
    epsa = (s + eps) * eps**(1.0/3.0)
    if bm.abs(s - a) > epsa:
        raise ValueError("A must be symmetric matrix")

    if x0 is not None:
        kwags = bm.context(x0)
    else:
        kwags = bm.context(b)

    info = {}

    x = x0
    q_1 = bm.zeros(n, **kwags)
    q_0 = bm.zeros(n, **kwags)
    q_2 = r / beta

    T_diag = []  # Main diagonal elements
    T_offdiag = []  # Off-diagonal elements

    a1 = 0
    a2 = 0
    b1 = 0
    a = 0
    b0 = 0
    d = 0

    t_0 = 0
    t_1 = beta

    p_1 = bm.zeros(n, **kwags)
    p_0 = bm.zeros(n, **kwags)
    p_2 = bm.zeros(n, **kwags)
    w = bm.zeros(m, **kwags)

    Tnorm = 0
    prev_res = float('inf')
    for niter in range(maxit):
        # Lanczos
        q_0 = q_1
        q_1 = q_2
        w = A @ q_1
        T_diag.append(q_1@w)

        if niter == 0:
            w = w - T_diag[niter] * q_1
        else:
            w = w - T_offdiag[niter-1] * q_0 - T_diag[niter] * q_1

        beta_new = bm.linalg.norm(w, ord=2)
        T_offdiag.append(beta_new)
        q_2 = w / beta_new

        if niter == 0:
            oldbeta = beta
        else:
            oldbeta = T_offdiag[-2]

        # Update Tnorm
        Tnorm += T_diag[niter]**2 + T_offdiag[niter]**2 + oldbeta**2

        if niter < 1:
            continue

        # QR step
        c = T_diag[-2] / bm.sqrt(T_offdiag[-2]**2 + T_diag[-2]**2)
        s = T_offdiag[-2] / bm.sqrt(T_offdiag[-2]**2 + T_diag[-2]**2)

        t_0 = t_1
        t_1 = -s * t_0
        t_0 = c * t_0

        # Residual
        res = bm.abs(t_1)
        # Monitor residual growth
        if res > prev_res:
            logger.warning("Residual stagnation, terminating early.")
            break

        # Update T
        a2 = b1
        a1 = a
        b1 = b0
        T_diag = bm.set_at(T_diag, -2, c*T_diag[-2]+s*T_offdiag[-2])
        if niter < 2:
            a = c * T_offdiag[-2] + s * T_diag[-1]
            T_diag = bm.set_at(T_diag, -1, -s*T_offdiag[-2]+c*T_diag[-1])
        else:
            a = c * d + s * T_diag[-1]
            T_diag = bm.set_at(T_diag, -1, -s*d+c*T_diag[-1])

        b0 = s * T_offdiag[-1]
        d = c * T_offdiag[-1]
        T_offdiag[-2] = 0

        p_0 = p_1
        p_1 = p_2
        p_2 = (q_0 - a1 * p_1 - a2 * p_0) / T_diag[-2]
        x = x + t_0 * p_2

        prev_res = res
        Anorm = bm.sqrt(Tnorm)
        xnorm = bm.linalg.norm(x)
        stop_res = res / (Anorm * xnorm)

        if res < atol:
            logger.info(f"minres: converged in {niter} iterations, "
                        "stopped by absolute tolerance.")
            break

        if res < rtol *Anorm * xnorm:
            logger.info(f"minres: converged in {niter} iterations, "
                        "stopped by relative tolerance.")
            break
        
        if (maxit is not None) and (niter >= maxit):
            logger.info(f"minres: failed, stopped by maxiter ({maxit}).")
            break

    info['residual'] = res
    info['niter'] = niter
    info['relative tolerance'] = stop_res
    return x,info
