
from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor

def _mumps_solve(A, b):
    """Solve a linear system using MUMPS.

    Parameters:
        A(COOTensor | CSRTensor): The matrix of the linear system.
        b(Tensor): The right-hand side.

    Returns:
        Tensor: The solution of the linear system.

    Note:
        This function requires the `mumps` package to be installed.
        A simple way to install it is
        ```
        sudo apt install libmumps64-scotch-dev
        pip3 install PyMUMPS
        ```
    """
    from mumps import DMumpsContext
    from scipy.sparse import coo_matrix

    A = coo_matrix((bm.to_numpy(A.values()), bm.to_numpy(A.indices())), 
                   shape = A.sparse_shape)
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

    A = csr_matrix((bm.to_numpy(A.values()), bm.to_numpy(A.indices())), 
                   shape = A.sparse_shape)
    x = bm.to_numpy(b).copy()
    return spsol(A, b)

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

    indices = cp.from_dlpack(A.indices())
    data = cp.from_dlpack(A.values())

    A = cp.sparse.csr_matrix((data, (indices[0], indices[1])), shape=A.shape)
    b = cp.from_dlpack(b) 
    x = spsol(A,b)
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



