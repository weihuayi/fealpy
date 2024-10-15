
from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor

def _data_transformation(A, b):
    """
    @brief      Transform the data of a sparse matrix and a vector to scipy format. 
    @param      A    The sparse matrix
    @param      b    The vector
    @return     The transformed data.
    """
    A = A.to_scipy()
    b = bm.to_numpy(b)
    return A, b 

def _mumps_solve(A, b):
    """
    @brief      Solve a linear system using MUMPS.
    @param      A    The matrix of the linear system.
    @param      b    The right-hand side
    @return     The solution of the linear system.
    @note       This function requires the `mumps` package to be installed.
                   A simple way to install it is
                   ```
                    sudo apt install libmumps64-scotch-dev
                    pip3 install PyMUMPS
                   ```
    """
    from mumps import DMumpsContext

    ctx = DMumpsContext()
    ctx.set_silent()
    ctx.set_centralized_sparse(A)

    x = b.copy()

    ctx.set_rhs(x)
    ctx.run(job=6)
    ctx.destroy()
    return x

def _scipy_solve(A, b):
    """
    @brief      Solve a linear system using cupy.
    @param      A    The matrix of the linear system
    @param      b    The right-hand side
    @return     The solution of the linear system.
    """
    from scipy.sparse.linalg import spsolve as spsol 
    return spsol(A, b)

def _cupy_solve(A, b):
    """
    @brief      Solve a linear system using cupy.
    @param      A    The matrix of the linear system
    @param      b    The right-hand side
    @return     The solution of the linear system.
    """
    import cupy as cp
    from cupyx.scipy.sparse.linalg import spsolve as spsol

    A = cp.sparse.csr_matrix(A.astype(cp.float64))
    b = cp.array(b, dtype=b.dtype)
    x = spsol(A,b)
    return b

def spsolve(A:[COOTensor, CSRTensor], b, solver:str="mumps"):
    """
    @brief      Solve a linear system using a direct solver.
    @param      A       The matrix of the linear system
    @param      b       The right-hand side
    @param      solver  The solver to use. It can be "mumps", "scipy", or "cupy".
    @return     The solution of the linear system.
    """
    A, b = _data_transformation(A, b)
    if solver == "mumps":
        return _mumps_solve(A, b)
    elif solver == "scipy":
        return _scipy_solver(A, b)
    elif solver == "cupy":
        _cupy_solver(A, b)
    else:
        raise ValueError(f"Unknown solver: {solver}")



