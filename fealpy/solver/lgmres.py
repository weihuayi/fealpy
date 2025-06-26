from typing import Optional, Protocol, Tuple

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse.coo_tensor import COOTensor
from ..sparse.csr_tensor import CSRTensor

from .. import logger


class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...


def lgmres(
    A: SupportsMatmul,
    b: TensorLike,
    x0: Optional[TensorLike] = None,
    atol: float = 1e-12,
    rtol: float = 1e-8,
    inner_m: Optional[int] = None,
    outer_k: Optional[int] = None,
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
        inner_m (int, optional): Number of inner iterations between restarts. Larger values increase
            iteration cost but may be necessary for convergence. Defaults to 20.
        outer_k (int, optional): Number of vectors to carry over between restarts. These vectors
            augment the subspace, potentially improving convergence. Defaults to 3.
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

    if inner_m is None:
        inner_m = 20

    if outer_k is None:
        outer_k = 3

    if maxit is None:
        maxit = 5 * m

    if x0 is not None:
        kwags = bm.context(x0)
    else:
        kwags = bm.context(b)

    info = {}

    dim = inner_m + outer_k
    dim = min(dim, m)
    w = bm.zeros(n, **kwags)
    Q = bm.zeros((dim + 1, n), **kwags)
    H = bm.zeros(dim + 1, **kwags)
    R = bm.zeros((dim + 1, dim + 1), **kwags)
    givens = bm.zeros((dim, 2), **kwags)
    z = []

    b_norm = bm.linalg.norm(b, ord=2)
    atol = max(float(atol), float(rtol) * float(b_norm))
    eps = bm.finfo(x0.dtype).eps

    all_iter = 0
    x = x0

    if M is not None:
        Mb_norm = bm.linalg.norm(M @ b, ord=2)
    else:
        Mb_norm = b_norm
    ptol_max_factor = 1.0
    ptol = Mb_norm * min(ptol_max_factor, atol / b_norm)
    presid = 0

    for niter in range(maxit):
        W = []
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
        t = bm.zeros(dim + 1, **kwags)
        t[0] = beta

        breakdown = False
        
        for i in range(inner_m + len(z)):
            all_iter += 1

            if i < inner_m:
                W.append(Q[i, :])
            else:
                W.append(z[i - inner_m])

            w = bm.tensor(A @ W[-1])
            if M is not None:
                w = M @ w
            h0 = bm.linalg.norm(w, ord=2)

            # Arnoldi process to get Q, H
            projs = Q[: i + 1, :] @ w
            H[: i + 1] = projs
            w -= Q[: i + 1, :].T @ projs
            h1 = bm.linalg.norm(w, ord=2)
            H[i + 1] = h1

            if h1 <= eps * h0:  # Breakdown check
                H[i + 1] = 0
                breakdown = True
            Q[i + 1, :] = w[:] / h1

            # QR decomposition step
            # 待优化(SciPy中使用qr_insert函数）
            for k in range(i):
                c, s = givens[k, 0], givens[k, 1]
                n0, n1 = H[[k, k + 1]]
                H[[k, k + 1]] = bm.tensor([c*n0 + s*n1, -s.conj()*n0 + c*n1])

            # Compute new Givens rotation
            sqrt = bm.sqrt(H[i]**2 + H[i + 1]**2)
            c = H[i] / sqrt
            s = H[i + 1] / sqrt
            givens[i, :] = bm.tensor([c, s])
            H[i] = sqrt
            H[i + 1] = 0
            R[i,:] = H

            # # Update t vector
            t = bm.set_at(t, i + 1, -s.conj()*t[i])
            t = bm.set_at(t, i, c*t[i])
            presid = bm.abs(t[i + 1])
            if presid <= ptol or breakdown:
                break
            
       # Solve for y using backward substitution
        if R[i, i] == 0:
            t[i] = 0
        y = bm.zeros([i + 1], **kwags)
        y[:] = t[:i + 1]
        for k in range(i, 0, -1):
            if y[k] != 0:
                y[k] /= R[k, k]
                tmp = y[k]
                y[:k] -= tmp * R[k,:k]
        if y[0] != 0:
            y[0] /= R[0, 0]

        # dx = y @ W[:]
        # 待优化(SciPy中使用blas函数）
        dx = bm.zeros_like(W[0])
        for k in range(i+1):
            dx += y[k] * W[k]


        if outer_k > 0:
            nx = bm.linalg.norm(dx)
            z.append(dx / nx)
            if len(z) > outer_k:
                z.pop(0)
        x = x + dx

        r = b - A @ x
        if M is None:
            res = presid
        else:
            res = bm.linalg.norm(r, ord=2)

        if breakdown:
            logger.info("Breakdown detected in Arnoldi process.")
            break

        if res <= atol:
            logger.info(
                f"GMRES converged in {all_iter} iterations (absolute tolerance: {atol:.1e})"
            )
            break

        if res <= rtol * b_norm:
            logger.info(
                f"GMRES converged in {all_iter} iterations (relative tolerance: {rtol:.1e})"
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
