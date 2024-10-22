from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor
import numpy as np
def _to_cupy_data(A, b):
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
    elif A.indices().device.type == "cpu": # torch backend
        A = A.device_put("cuda")
        indices = cp.from_dlpack(A.indices())
        data = cp.from_dlpack(A.values())
        A = cp.sparse.csr_matrix((data, (indices[0], indices[1])), shape=A.shape)
    else:
        indices = cp.from_dlpack(A.indices())
        data = cp.from_dlpack(A.values())
        A = cp.sparse.csr_matrix((data, (indices[0], indices[1])), shape=A.shape)

    if isinstance(b, np.ndarray) or b.device.type == "cpu":
        b = bm.to_numpy(b)
        b = cp.array(b)
    else:
        b = cp.from_dlpack(b)
    return A, b


def _cupy_solve(A, b, atol):

    """Solve a linear system using cupy.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.

    Returns:
        Tensor: The solution of the linear system.
    """
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cpx

    iscpu = isinstance(b, np.ndarray) or b.device.type == "cpu"
    A, b = _to_cupy_data(A, b)
    x, info = cpx.gmres(A, b, tol=atol)
    if iscpu:
        x = cp.asnumpy(x)
    return x

def _mumps_solve(A, b, atol):
    pass
def _scipy_solve(A, b, atol):
    pass


def gmres(A:[COOTensor, CSRTensor], b, solver:str="cupy", atol=1e-18):
    """Solve a linear system using a direct solver.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.
        solver(str): The solver to use. It can be "mumps", "scipy", or "cupy".

    Returns:
        Tensor: The solution of the linear system.
    """
    if solver == "mumps":
        return bm.tensor(_mumps_solve(A, b))
    elif solver == "scipy":
        return bm.tensor(_scipy_solve(A, b))
    elif solver == "cupy":
        A = A.tocoo()
        return bm.tensor(_cupy_solve(A, b, atol=atol))
    else:
        raise ValueError(f"Unknown solver: {solver}")



