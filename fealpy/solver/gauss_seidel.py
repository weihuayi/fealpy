
from typing import Optional, Protocol

from ..backend import backend_manager as bm
from ..backend import TensorLike

from .. import logger

class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...

def gauss_seidel(A: SupportsMatmul, b: TensorLike, x0: Optional[TensorLike]=None,
       atol: float=1e-12, rtol: float=1e-8,
       maxit: Optional[int]=10000,returninfo: bool=False) -> TensorLike:
    """
    Solve a linear system Ax = b using the Gauss-Seidel method.

    Parameters:
        A (SupportsMatmul): The coefficient matrix of the linear system.
        b (TensorLike): The right-hand side vector of the linear system, can be a 1D or 2D tensor.
        x0 (TensorLike, optional): Initial guess for the solution, a 1D or 2D tensor. Must have the same shape as `b`.\
        Defaults to a zero tensor if not provided.
        atol (float, optional): Absolute tolerance for convergence. Default is 1e-12.
        rtol (float, optional): Relative tolerance for convergence. Default is 1e-8.
        maxit (int, optional): Maximum number of iterations allowed. Default is 10000. If not provided, the method will\
        continue until convergence based on the given tolerances.

    Returns:
        Tensor: The approximate solution to the system Ax = b.
        dict: A dictionary containing the residual and the number of iterations taken.

    Raises:
        ValueError: If inputs do not meet the specified conditions (e.g., A is not sparse, dimensions mismatch).

    Note:
        This implementation decomposes the matrix A into its upper triangular part (U) and lower triangular part (L),
        with A = L + D + U. The algorithm solves iteratively by updating the solution with forward substitution on M = L + D,
        until the residual norm meets the specified tolerance or the maximum number of iterations is reached.

        This method assumes that the system is well-conditioned for the Gauss-Seidel iteration.
    """
    from .mumps import spsolve_triangular
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

    info = {}
    U = A.triu(k=1)
    M = A.tril()  #M = D-L，A = D-L-U

    niter = 0
    x = x0
    while True:
        B = b - U.matmul(x)
        x = spsolve_triangular(M, B)   # Use forward substitution to solve the linear system
        a = b - A.matmul(x)
        res = bm.linalg.norm(b-A.matmul(x))
        niter +=1
        if res < rtol :
            logger.info(f"Gauss Seidel: converged in {iter} iterations, "
                        "stopped by relative tolerance.")
            break

        if (maxit is not None) and (niter >= maxit):
            logger.info(f"Gauss Seidel: failed, stopped by maxiter ({maxit}).")
            break
    info['residual'] = res    
    info['niter'] = niter 
    if returninfo is True:
        return x, info
    else:
        return x
