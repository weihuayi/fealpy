
from .dmumps_context import DMumpsContext

_solvers ={
        "D": DMumpsContext,
        }

class MumpsSolver:
    def __init__(self, stype='D', par=1, comm=None, timer=None, logger=None):
        """
        """
        self.stype = stype
        self.context = _solvers[stype](par=par, sym=0, comm=comm)

    def solve(self, A, b):
        self.context.set_matrix(A)
        self.context.set_rhs(b)
        self.context.run()
        return self.context.get_solution()
