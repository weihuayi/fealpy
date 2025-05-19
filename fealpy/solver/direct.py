
from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor
import numpy as np

def _mumps_solve(A, b):
    """Solve a linear system using MUMPS.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.

    Returns:
        Tensor: The solution of the linear system.

    Note:
        The matrix `A` must be on the CPU.
        This function requires the `mumps` package to be installed.
        A simple way to install it is
        ```
        sudo apt install libmumps64-scotch-dev
        pip3 install PyMUMPS
        ```
    """
    from mumps import DMumpsContext
    A = A.to_scipy()
    x = bm.to_numpy(b).copy()

    ctx = DMumpsContext()
    ctx.set_silent()
    ctx.set_centralized_sparse(A)

    ctx.set_rhs(x)
    ctx.run(job=6)
    ctx.destroy()
    return x

def _scipy_solve(A, b):
    """Solve a linear system using cupy.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.

    Returns:
        Tensor: The solution of the linear system.
    """
    from scipy.sparse.linalg import spsolve as spsol 
    from scipy.sparse import csr_matrix

    A = A.to_scipy()
    b = bm.to_numpy(b)
    return spsol(A, b)

def _to_cupy_data(A, b):
    """Convert the input tensors to cupy tensors.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.

    Returns:
        Tuple: The converted tensors.
    """
    import cupy as cp
    if isinstance(A.indices, np.ndarray): # numpy backend
        A =  A.to_scipy() 
        A = cp.sparse.csr_matrix(A.astype(cp.float64))
    elif A.indices.device.type == "cpu": # torch backend
        A = A.device_put("cuda")
        indices = cp.from_dlpack(A.indices)
        data = cp.from_dlpack(A.values)
        A = cp.sparse.csr_matrix((data, (indices[0], indices[1])), shape=A.shape)
    else:
        indices = cp.from_dlpack(A.indices)
        data = cp.from_dlpack(A.values)
        A = cp.sparse.csr_matrix((data, (indices[0], indices[1])), shape=A.shape)

    if isinstance(b, np.ndarray) or b.device.type == "cpu":
        b = bm.to_numpy(b)
        b = cp.array(b)
    else:
        b = cp.from_dlpack(b)
    return A, b

def _cupy_solve(A, b):
    """Solve a linear system using cupy.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.

    Returns:
        Tensor: The solution of the linear system.
    """
    import cupy as cp
    from cupyx.scipy.sparse.linalg import spsolve as spsol

    iscpu = isinstance(b, np.ndarray) or b.device.type == "cpu"
    A, b = _to_cupy_data(A, b)
    x = spsol(A,b)
    if iscpu:
        x = cp.asnumpy(x)
    return x

def spsolve(A:[COOTensor, CSRTensor], b, solver:str="mumps"):
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
        return bm.tensor(_cupy_solve(A, b))
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _cupy_spsolve_triangular(A, b, lower=True):
    """Solve a linear system using cupy.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.

    Returns:
        Tensor: The solution of the linear system.
    """
    import cupy as cp
    from cupyx.scipy.sparse.linalg import spsolve_triangular as spsol_tri

    iscpu = isinstance(b, np.ndarray) or b.device.type == "cpu"
    A, b = _to_cupy_data(A, b)
    x = spsol(A, b, lower=lower)
    if iscpu:
        x = cp.asnumpy(x)
    return x

def _mumps_spsolve_triangular(A:[COOTensor, CSRTensor], b, lower=True):
    """Sovle a triangular system using mumps on cpu
    Parameters:
        A(COOTensor | CSRTensor): The upper or lower matrix of the linear system.
        b(Tensor): The right-hand side.

    Returns:
        Tensor: The solution of the linear system.
    """
    from .mumps import DMumpsContext 

    x = b.copy()
    ctx = DMumpsContext()
    ctx.set_silent()
    ctx.set_centralized_sparse(A)

    ctx.set_rhs(x)
    ctx.run(job=6)
    ctx.destroy()
    return x


def spsolve_triangular(A:[COOTensor, CSRTensor], b, lower=True):
    """Solve a triangular system Ax = b
    Parameters:
        A(COOTensor | CSRTensor): The upper or lower triangular matrix of the linear system.
        b(Tensor): The right-hand side.
    """
    device = bm.device_type(b)
    if device == 'cpu':
        x = _mumps_spsolve_triangular(A, b, lower=lower)
    elif device == 'gpu':
        x = _cupy_spsolve_triangular(A, b, lower=lower)
    else:
        raise ValueError(f"Unsupport device: {device}")

    return x
