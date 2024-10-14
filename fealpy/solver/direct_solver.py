
from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor

def _mumps_solve(I, J, data, b):
    """
    @brief      Solve a linear system using MUMPS.
    @param      I    The row indices of the matrix
    @param      J    The column indices of the matrix
    @param      data  The data of the matrix
    @param      b    The right-hand side
    @return     The solution of the linear system.
    @note       1. This function requires the `mumps` package to be installed.
                   A simple way to install it is
                   ```
                    sudo apt install libmumps64-scotch-dev
                    pip3 install PyMUMPS
                   ```
                2. The indices in `I` and `J` should be 0-based.
    """
    from mumps import DMumpsContext

    NN = len(b)
    ctx = DMumpsContext()
    ctx.set_silent()
    ctx.set_centralized_assembled(I+1, J+1, data)

    x = b.copy()

    ctx.set_rhs(x)
    ctx.run(job=6)
    ctx.destroy()
    return x

def _scipy_solver(A, b):
    from scipy.sparse.linalg import spsolve 
    return spsolve(A, b)

def _cupy_solver(A, b):
    pass

def spsolve(A:[COOTensor, CSRTensor], b, solver:str="mumps"):
    """
    @brief      Solve a linear system using a direct solver.
    @param      A       The matrix of the linear system
    @param      b       The right-hand side
    @param      solver  The solver to use. It can be "mumps", "scipy", or "cupy".
    @return     The solution of the linear system.
    """
    if solver == "mumps":
        A = A.to_scipy()
        return _mumps_solve(A.row, A.col, A.data, b)
    elif solver == "scipy":
        A = A.to_scipy()
        return _scipy_solver(A, b)
    elif solver == "cupy":
        _cupy_solver(A, b)
    else:
        raise ValueError(f"Unknown solver: {solver}")



