from typing import Optional, Protocol

from ..backend import backend_manager as bm
from ..backend import TensorLike
from .mumps import spsolve, spsolve_triangular
from ..sparse.coo_tensor import COOTensor
from ..sparse.csr_tensor import CSRTensor

from .. import logger

class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...

def jacobi(A: SupportsMatmul, b: TensorLike, x0: Optional[TensorLike]=None,
       atol: float=1e-12, rtol: float=1e-8,
       maxit: Optional[int]=10000) -> TensorLike:
    
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

    #张量分裂
    info = {}
    U = A.triu(k=1)
    L = A.tril(k=-1)
    M = A - L- U
    N = L+U
    

    err = 1
    niter = 0
    x = x0
    while True:
        B = b - N@x
        x += spsolve_triangular(M, B)  # 使用前向替换求解线性方程组
        a = b - A@x
        res = bm.linalg.norm(b-A@x)
        niter +=1
        print("n=:", niter, "residual: ", res)
        if res < rtol :
            logger.info(f"Jacobi: converged in {iter} iterations, "
                        "stopped by relative tolerance.")
            break

        if (maxit is not None) and (niter >= maxit):
            logger.info(f"Jacobi: failed, stopped by maxiter ({maxit}).")
            break
    info['residual'] = res    
    info['niter'] = niter 
    return x, info 
