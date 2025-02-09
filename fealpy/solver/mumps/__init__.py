from .dmumps_context import DMumpsContext
from .mumps_solver import MumpsSolver

def spsolve(A, b, comm=None):
    """Sparse solve A\b."""

    assert A.dtype == 'd' and b.dtype == 'd', "Only double precision supported."
    with DMumpsContext(par=par, sym=0, comm=comm) as ctx:
        if ctx.myid == 0:
            # Set the sparse matrix -- only necessary on
            ctx.set_centralized_sparse(A)
            x = b.copy()
            ctx.set_rhs(x)

        # Silence most messages
        ctx.set_silent()

        # Analysis + Factorization + Solve
        ctx.run(job=6)

        if ctx.myid == 0:
            return x
