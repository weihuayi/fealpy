from typing import Optional, Protocol, Tuple

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse.coo_tensor import COOTensor
from ..sparse.csr_tensor import CSRTensor

from .. import logger


class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...


def gmres(
    A: SupportsMatmul,
    b: TensorLike,
    x0: Optional[TensorLike] = None,
    atol: float = 1e-12,
    rtol: float = 1e-8,
    restart: Optional[int] = None,
    maxit: Optional[int] = None,
    M: Optional[SupportsMatmul] = None
) -> Tuple[TensorLike, dict]:
    """
    Solve a linear system Ax = b using Generalized Minimal RESidual iteration (gmres) method.

    Parameters:
        A (SupportsMatmul): The coefficient matrix of the linear system.
        b (TensorLike): The right-hand side vector of the linear system, can be a 1D or 2D tensor.
        x0 (TensorLike, optional): Initial guess for the solution, a 1D or 2D tensor.
            Must have the same shape as b. Defaults to None.
        atol (float, optional): Absolute tolerance for convergence. Defaults to 1e-12.
        rtol (float, optional): Relative tolerance for convergence. Defaults to 1e-8.
        restart (int, optional): Number of iterations between restarts. Larger values increase
            iteration cost but may be necessary for convergence. Defaults to ``min(20, m)``.
        maxit (int, optional): Maximum number of iterations allowed. Defaults to 5*m.
            If not provided, the method will continue until convergence based on tolerances.
        M (TensorLike): Inverse of the preconditioner of `A`.  `M` should approximate the
            inverse of `A` and be easy to solve. In this implementation, left preconditioning 
            is used, and the preconditioned residual is minimized. Defaults to None.

    Returns:
        TensorLike: The approximate solution to the system Ax = b.
        dict: Information dictionary containing residual and iteration count.

    Raises:
        ValueError: If inputs do not meet specified conditions (e.g., dimensions mismatch).
    """
    assert isinstance(b, TensorLike), "b must be a Tensor"
    if x0 is not None:
        assert isinstance(x0, TensorLike), "x0 must be a Tensor if not None"
    single_vector = b.ndim == 1

    if b.ndim not in {1, 2}:
        raise ValueError("b must be a 1D or 2D dense tensor")

    if x0 is None:
        x0 = bm.zeros_like(b)
    else:
        if x0.shape != b.shape:
            raise ValueError("x0 and b must have the same shape")
    
    m, n = A.shape  # Comma spacing
    if restart is None:
        restart = 20
    restart = min(restart, m)

    if maxit is None:
        maxit = 5 * m
    
    if x0 is not None:
        kwags = bm.context(x0)
    else:
        kwags = bm.context(b)
    
    info = {}
    H = bm.zeros((restart, restart + 1), **kwags)
    Q = bm.zeros((restart + 1, n), **kwags)
    givens = bm.zeros((restart, 2), **kwags)
    
    b_norm = bm.linalg.norm(b, ord=2)
    atol = max(float(atol), float(rtol) * float(b_norm))
    eps = bm.finfo(x0.dtype).eps

    all_iter = 0
    x = x0

    if M is not None:
        Mb_norm = bm.linalg.norm(M @ b, ord=2)
    else:
        Mb_norm = b_norm
    ptol_max_factor = 1.
    ptol = Mb_norm * min(ptol_max_factor, atol / b_norm)
    presid = 0.

    for niter in range(maxit):
        if niter == 0:
            r = b - A @ x if x.any() else b
            res = bm.linalg.norm(r)
            if res < atol:  # Early convergence check
                return x, {"residual": res, "niter": niter}
            
        if M is not None:
            r_pre = M @ r
            beta = bm.linalg.norm(r_pre, ord=2)
            Q[0, :] = r_pre / beta
        else:
            beta = bm.linalg.norm(r, ord=2)
            Q[0, :] = r / beta
        t = bm.zeros(restart + 1, **kwags)
        t[0] = beta
        
        breakdown = False
        for i in range(restart):
            all_iter += 1
            w = bm.tensor(A @ Q[i, :])
            if M is not None:
                w = M @ w
            h0 = bm.linalg.norm(w, ord=2)

            # Arnoldi process to get Q, H
            projs = bm.dot(Q[:i + 1, :].conj(), w)
            H[i, :i + 1] = projs
            w -= bm.dot(Q[:i + 1, :].T, projs)
            h1 = bm.linalg.norm(w, ord=2)
            H[i, i + 1] = h1
            
            if h1 <= eps * h0:  # Breakdown check
                H[i, i + 1] = 0
                breakdown = True
            else:
                Q[i + 1, :] = w[:] / h1  # Fixed spacing
            
            # QR decomposition step
            for k in range(i):
                c, s = givens[k, 0], givens[k, 1]
                n0, n1 = H[i, [k, k + 1]]
                H[i, [k, k + 1]] = bm.tensor([c*n0 + s*n1, -s.conj()*n0 + c*n1])
            
            # Compute new Givens rotation
            c, s, sqrt_val = lartg(H[i, i], H[i, i + 1])
            givens[i, :] = bm.tensor([c, s])
            H[i, i] = sqrt_val
            H[i, i + 1] = 0

            # Update t vector
            t = bm.set_at(t, i + 1, -s.conj()*t[i])
            t = bm.set_at(t, i, c*t[i])
            presid = bm.abs(t[i + 1])
            if presid <= ptol or breakdown:  # Tolerance check
                break
        
        # Solve for y using backward substitution
        if H[i, i] == 0:
            t[i] = 0
        y = bm.zeros([i + 1], **kwags)
        y[:] = t[:i + 1]
        for k in range(i, 0, -1):
            if y[k] != 0:
                y[k] /= H[k, k]
                tmp = y[k]
                y[:k] -= tmp * H[k, :k]
        if y[0] != 0:
            y[0] /= H[0, 0]
            
        x = x + y @ Q[:i + 1, :]
        r = b - A @ x
        if M is None:
            res = presid
        else:
            res = bm.linalg.norm(r, ord=2)
        # res = bm.linalg.norm(r, ord=2)
        
        if breakdown:
            logger.info("Breakdown detected in Arnoldi process.")
            break
    
        if res <= atol:
            logger.info(
                f"GMRES converged in {niter} iterations (absolute tolerance: {atol:.1e})"
            )
            break

        if res <= rtol * b_norm:
            logger.info(
                f"GMRES converged in {niter} iterations (relative tolerance: {rtol:.1e})"
            )
            break

        if (maxit is not None) and (niter >= maxit):
            logger.info(f"GMRES failed to converge within {maxit} iterations.")
            break

        if presid <= ptol:
            ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
        else:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)

        ptol = presid * min(ptol_max_factor, atol / res)

    info['residual'] = res
    info['niter'] = all_iter
    return x, info


def lartg(a, b):
    """
    Compute new Givens rotation
    """
    abs_a = bm.abs(a)
    abs_b = bm.abs(b)

    # Handle trivial cases
    if abs_b == 0 and abs_a != 0:
        return bm.tensor(1), bm.tensor(0), abs_a
    if abs_a == 0 and abs_b != 0:
        return bm.tensor(0), b.conj()/abs_b, abs_b
    if abs_a == 0 and abs_b == 0:
        return bm.tensor(1), bm.tensor(0), bm.tensor(0)

    # Thresholds for safe computation
    finfo = bm.finfo(float)
    small_thresh = bm.sqrt(bm.tensor(finfo.min))
    large_thresh = bm.sqrt(bm.tensor(finfo.max))

    t = max(abs_a, abs_b)
    if t >= small_thresh and t <= large_thresh:
        # Normal formula
        r = bm.sqrt(abs_a**2 + abs_b**2)
        c = a / r 
        s = b / r
    else:
        # Scaled formula for stability
        scale = abs_a + abs_b
        norm  = scale * bm.sqrt((abs_a/scale)**2 + (abs_b/scale)**2)
        phi = a / abs_a
        c = abs_a / norm
        s = phi * bm.conj(b) / norm
        r = phi * norm

    return c, s, r