from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor
import numpy as np
def _to_cupy_data(A, b, x0):
    """Convert the input tensors to cupy tensors.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.

    Returns:
        Tuple: The converted tensors.
    """
    import cupy as cp
    if isinstance(A.indices(), np.ndarray): # numpy backend
        A =  A.to_scipy() 
        A = cp.sparse.csr_matrix(A.astype(cp.float64))
    elif bm.device_type(A.indices()) == "cpu": # torch backend
        A = A.device_put("cuda")
        indices = cp.from_dlpack(A.indices())
        data = cp.from_dlpack(A.values())
        A = cp.sparse.csr_matrix((data, (indices[0], indices[1])), shape=A.shape)
    else:
        indices = cp.from_dlpack(A.indices())
        data = cp.from_dlpack(A.values())
        A = cp.sparse.csr_matrix((data, (indices[0], indices[1])), shape=A.shape)

    if isinstance(b, np.ndarray) or bm.device_type(b) == "cpu":
        b = bm.to_numpy(b)
        b = cp.array(b)
        if x0 is not None:
            x0 = bm.to_numpy(x0)
            x0 = cp.array(x0)
    else:
        b = cp.from_dlpack(b)
        if x0 is not None:
            x0 = cp.from_dlpack(x0)
    return A, b, x0


def _cupy_solve(A, b, tol, x0, maxiter ,atol):

    """Solve a linear system using cupy.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.

    Returns:
        Tensor: The solution of the linear system.
    """
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cpx

    iscpu = isinstance(b, np.ndarray) or bm.device_type(b) == "cpu"
    A, b, x0 = _to_cupy_data(A, b, x0)
    x, info = cpx.gmres(A, b, tol=tol, x0=x0, maxiter=maxiter, atol=atol)
    if iscpu:
        x = cp.asnumpy(x)
    return x

def _scipy_solve(A, b, tol, x0, maxiter, atol):
    from scipy.sparse.linalg import gmres 
    from scipy.sparse import csr_matrix

    A = A.to_scipy()
    b = bm.to_numpy(b)
    return gmres(A, b, x0=x0, maxiter=maxiter, atol=atol, rtol=tol)[0]


def gmres(A:[COOTensor, CSRTensor], b, solver:str="scipy", 
          tol=1e-5, x0=None, maxiter=None, atol=0.0):
    """Solve a linear system using a gmres solver.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.
        solver(str): The solver to use. It can be "mumps", "scipy", or "cupy".

    Returns:
        Tensor: The solution of the linear system.
    """
    if solver == "scipy":
        return bm.tensor(_scipy_solve(A, b, tol=tol, x0=x0, maxiter=maxiter, atol=atol))
    elif solver == "cupy":
        A = A.tocoo()
        return bm.tensor(_cupy_solve(A, b, tol=tol, x0=x0, maxiter=maxiter, atol=atol))
    else:
        raise ValueError(f"Unknown solver: {solver}")



