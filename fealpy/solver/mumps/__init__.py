from .dmumps_context import DMumpsContext
from .mumps_solver import MumpsSolver

def spsolve(A, b, silent=True):
    """Sparse solve A\b.
    """

    assert A.dtype == 'd' and b.dtype == 'd', "Now only double precision supported."
    s = MumpsSolver(silent=silent)
    return s.solve(A, b)

def spsolve_triangular(A, b, transpose=False, silent=True):
    """Solve sparse triangular system Ax=b or A^Tx=b.
    """
    assert A.dtype == 'd' and b.dtype == 'd', "Now only double precision supported."
    s = MumpsSolver(silent=silent)
    return s.solve_triangular(A, b, transpose=transpose)
