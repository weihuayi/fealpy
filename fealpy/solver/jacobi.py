from typing import Optional, Protocol

from ..backend import backend_manager as bm
from ..backend import TensorLike
from scipy.sparse.linalg import spsolve,spsolve_triangular
#from .mumps import spsolve, spsolve_triangular
from ..sparse.coo_tensor import COOTensor
from ..sparse.csr_tensor import CSRTensor

from .. import logger

class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...

def jacobi(A: SupportsMatmul, b: TensorLike, x0: Optional[TensorLike]=None,
       atol: float=1e-12, rtol: float=1e-8,
       maxit: Optional[int]=10000,returninfo: bool=False) -> TensorLike:
    """
    Solve a linear system Ax = b using the Jacobi iterative method.

    Parameters:
        A (SupportsMatmul): The coefficient matrix of the linear system.
        b (TensorLike): The right-hand side vector of the linear system, can be a 1D or 2D tensor.
        x0 (TensorLike, optional): Initial guess for the solution, a 1D or 2D tensor. Must have the same shape as b.
        atol (float, optional): Absolute tolerance for convergence. Default is 1e-12.
        rtol (float, optional): Relative tolerance for convergence. Default is 1e-8.
        maxit (int, optional): Maximum number of iterations allowed. Default is 10000. If not provided, the method will continue until convergence based on the given tolerances.

    Returns:
        Tensor: The approximate solution to the system Ax = b.

    Raises:
        ValueError: If inputs do not meet the specified conditions (e.g., A is not sparse, dimensions mismatch).

    Note:
        The Jacobi method is an iterative method for solving systems of linear equations. 
        It is a special case of the Gauss-Seidel method and is often used as a preconditioner for other methods.
    """
    kargs = bm.context(b)
    assert isinstance(b, TensorLike), "b must be a Tensor"
    if x0 is not None:
        assert isinstance(x0, TensorLike), "x0 must be a Tensor if not None"
    single_vector = b.ndim == 1

    if b.ndim not in {1, 2}:
        raise ValueError("b must be a 1D or 2D dense tensor")

    if x0 is None:
        x0 = bm.zeros_like(b,**kargs)
    else:
        if x0.shape != b.shape:
            raise ValueError("x0 and b must have the same shape")

    # Tensor splitting
    info = {}
    U = A.triu(k=1)
    L = A.tril(k=-1)
    M = A - L - U
    N = L + U

    err = 1
    niter = 0
    x = x0
    while True:
        B = b - N @ x
        #x = spsolve_triangular(M, B)  # Use forward substitution to solve the linear system
        x += spsolve_triangular(M.to_scipy(), B)
        a = b - A @ x
        res = bm.linalg.norm(a)
        niter += 1
        if res < rtol:
            logger.info(f"Jacobi: converged in {niter} iterations, "
                        "stopped by relative tolerance.")
            break

        if (maxit is not None) and (niter >= maxit):
            logger.info(f"Jacobi: failed, stopped by maxiter ({maxit}).")
            break
    info['residual'] = res    
    info['niter'] = niter 
    if returninfo is True:
        return x, info
    else:
        return x
