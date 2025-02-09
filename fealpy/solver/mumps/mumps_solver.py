
from .dmumps_context import DMumpsContext

_solvers ={
        "D": DMumpsContext,
        }

class MumpsSolver:
    def __init__(self, stype='D', par=1, comm=None, timer=None, logger=None):
        """Initialize the MumpsSolver object.
        """
        self.timer = timer
        self.logger = logger
        self.context = _solvers[stype](par=par, comm=comm)

    def solve(self, A, b, sym=0, job=6, lower=None):
        self.context.id.sym = sym
        x = b.copy()
        self.context.set_matrix(A)
        self.context.set_rhs(x)
        self.context.run(job)
        return x 
